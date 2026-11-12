"""Real CUDA gates; these records are never inferred from CPU unit tests."""
import copy
import gc
import platform
import time
import shutil
import importlib.metadata
import torch
import psutil
from .train import rng_state,restore_rng,optimizer,update,payload
from .storage import atomic_json,Recovery
from .protocol import P

def environment(root,require_cuda=True):
    info={'python':platform.python_version(),'ram_available_bytes':psutil.virtual_memory().available,
          'disk_free_bytes':shutil.disk_usage(root).free,
          'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','numpy']},
          'cuda_available':torch.cuda.is_available(),'gpu_smoke_status':'not run'}
    if info['cuda_available']:
        info.update(gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                    free_vram_bytes=torch.cuda.mem_get_info()[0])
    if require_cuda:
        if not info['cuda_available']: raise RuntimeError('CUDA is required: select a single T4 Colab GPU runtime')
        if info['gpu_count']!=1 or 'T4' not in info['gpu']: raise RuntimeError('Mandatory preflight expects one T4; no automatic hardware/protocol substitution')
        if info['free_vram_bytes']<12*2**30: raise RuntimeError('Less than 12 GiB free VRAM; clear other GPU workloads')
        if info['disk_free_bytes']<12*2**30 or info['ram_available_bytes']<5*2**30: raise RuntimeError('Insufficient local disk/RAM for setup')
    return info

def gradient_gate(model,stage):
    connector=[p.grad for p in model.connector.parameters() if p.grad is not None]
    if not connector or not any(torch.count_nonzero(g) for g in connector): raise AssertionError('Connector gradient path broken')
    if not all(torch.isfinite(g).all() for g in connector): raise AssertionError('Nonfinite connector gradients')
    if stage:
        grads=[p.grad for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None]
        if not grads or not any(torch.count_nonzero(g) for g in grads): raise AssertionError('LoRA gradient path broken')
    model.audit(stage)

def smoke_stage(model,features,data,stage,out,identity):
    """Three 8x2 updates, then exact interrupted-vs-continuous recovery check.
    Isolated weights, RNG and optimizer state; no smoke examples count as training.
    """
    model.stage(stage);initial=model.trainable_state();rng=rng_state()
    # Force cap-length content, using real selected image features. Synthetic text
    # here is exclusively a stress fixture, never training/evaluation data.
    base=next(e for e in data['examples'].values() if e['split']=='train2014' and e['kind']=='caption')
    stress=copy.deepcopy(base)
    stress['question_tokens']=(base['question_tokens']*80)[:80]
    stress['answer_tokens']=(base['answer_tokens']*64)[:64]
    examples=[stress]*16
    opt=optimizer(model,stage);scaler=torch.amp.GradScaler('cuda')
    results=[];torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(3):
            torch.cuda.synchronize();start=time.monotonic()
            losses,norm,retries=update(model,opt,scaler,examples,features,stage,i+1)
            torch.cuda.synchronize();gradient_gate(model,stage)
            results.append({'seconds':time.monotonic()-start,'loss':sum(losses)/16,'gradient_norm':norm,'overflow_retries':retries})
        recovery=Recovery(out/'smoke_local',out/'smoke_durable',identity)
        t=time.monotonic();recovery.save(payload(model,opt,scaler,stage,3,'isolated_smoke',[]));copy_seconds=time.monotonic()-t
        state=recovery.load()
        expected_loss=update(model,opt,scaler,examples,features,stage,4)[0]
        expected=model.trainable_state()
        model.restore_trainable(state['trainable']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);restore_rng(state['rng'])
        actual_loss=update(model,opt,scaler,examples,features,stage,4)[0]
        for name,p in model.trainable_state().items():
            torch.testing.assert_close(p,expected[name],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(torch.tensor(actual_loss),torch.tensor(expected_loss),rtol=1e-5,atol=1e-6)
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        if reserved>torch.cuda.get_device_properties(0).total_memory*.95: raise RuntimeError('CUDA smoke lacks 5% memory reserve')
        result={'status':'GPU smoke tested','stage':stage+1,'official_8x2':results,'recovery_equivalence':'PASS',
                'recovery_tolerance':{'rtol':1e-5,'atol':1e-6},'checkpoint_seconds':copy_seconds,
                'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
                'seconds_per_update':sum(x['seconds'] for x in results[1:])/2}
        # Optional effective-batch-equivalent comparison, never used for main training.
        for value in opt.state.values():
            for tensor in value.values():
                if torch.is_tensor(tensor) and tensor.dtype!=torch.float32: raise AssertionError('Optimizer state is not FP32')
        opt.state.clear()
        model.restore_trainable(initial);model.zero_grad(set_to_none=True);restore_rng(rng)
        comparison=optimizer(model,stage);comparison_scaler=torch.amp.GradScaler('cuda')
        try:
            rates=[];torch.cuda.reset_peak_memory_stats()
            for step in range(1,4):
                torch.cuda.synchronize();start=time.monotonic()
                update(model,comparison,comparison_scaler,examples,features,stage,step,microbatch=16)
                torch.cuda.synchronize();rates.append(time.monotonic()-start)
            result['isolated_16x1']={'status':'measured','seconds_per_update':sum(rates[1:])/2,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        except torch.cuda.OutOfMemoryError:
            result['isolated_16x1']={'status':'OOM; mandatory run remains 8x2'}
        finally:
            del comparison,comparison_scaler
        for value in opt.state.values():
            for key,tensor in value.items():
                if torch.is_tensor(tensor) and tensor.dtype!=torch.float32: raise AssertionError('Optimizer state is not FP32')
        atomic_json(out/f'cuda-stage-{stage+1}.json',result)
        return result
    finally:
        model.restore_trainable(initial);model.zero_grad(set_to_none=True);restore_rng(rng)
        del opt,scaler;gc.collect();torch.cuda.empty_cache()
