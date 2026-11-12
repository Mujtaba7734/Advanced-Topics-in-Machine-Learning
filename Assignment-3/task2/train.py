"""Exact update accounting and restartable mixed-precision training."""
import copy
import random
import time
import numpy as np
import torch
from .protocol import P,lr_factor

def seed_all():
    random.seed(P.seed);np.random.seed(P.seed);torch.manual_seed(P.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(P.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

def optimizer(model,stage):
    groups=[{'params':list(model.connector.parameters()),'lr':P.connector_lr[stage],'peak_lr':P.connector_lr[stage]}]
    lora=[p for n,p in model.language.named_parameters() if p.requires_grad]
    if stage: groups.append({'params':lora,'lr':P.lora_lr[stage],'peak_lr':P.lora_lr[stage]})
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    actual={id(p) for g in opt.param_groups for p in g['params']}
    assert actual=={id(p) for p in model.parameters() if p.requires_grad}
    return opt

def update(model,opt,scaler,examples,features,stage,step,microbatch=8):
    if len(examples)!=16 or microbatch not in (8,16): raise ValueError('Official effective batch must be 16')
    model.train(); params=[p for p in model.parameters() if p.requires_grad]
    for group in opt.param_groups: group['lr']=group['peak_lr']*lr_factor(step,P.updates[stage])
    before=rng_state()
    # GradScaler overflow may skip a physical optimizer step. Retry the same draw;
    # never count skipped steps or advance the precommitted sampling plan.
    for attempt in range(8):
        restore_rng(before);opt.zero_grad(set_to_none=True)
        losses=[]
        for j in range(0,16,microbatch):
            batch=examples[j:j+microbatch];f=features.batch(batch)
            with torch.autocast('cuda',dtype=torch.float16): values=model.losses(batch,f)
            losses.append(values.detach())
            scaler.scale(values.sum()/16).backward()
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        scale=scaler.get_scale()
        if not torch.isfinite(norm):
            scaler.step(opt);scaler.update()
            if scaler.get_scale()>=scale: raise FloatingPointError('Nonfinite gradients without scaler backoff')
            continue
        scalar_losses=torch.cat(losses)
        if not torch.isfinite(scalar_losses).all(): raise FloatingPointError('Nonfinite loss')
        scaler.step(opt);scaler.update()
        return scalar_losses.cpu().tolist(),float(norm),attempt
    raise FloatingPointError('Eight consecutive FP16 overflows; no main update counted')

def payload(model,opt,scaler,stage,completed,plan_hash,logs):
    return {'trainable':model.trainable_state(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
            'scheduler':{'stage':stage,'completed_in_stage':completed-sum(P.updates[:stage]),
                         'rule':'ceil 5% warmup, cosine to 0.1'},
            'stage':stage,'update':completed,'plan_hash':plan_hash,'rng':rng_state(),'logs':logs}
