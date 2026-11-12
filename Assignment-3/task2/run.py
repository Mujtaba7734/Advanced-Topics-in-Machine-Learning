"""Single entry point: python -m task2.run [--smoke-only]."""
import argparse
import dataclasses
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter
import torch
from .protocol import P,digest,schedule,stage_at
from .storage import atomic_json,sha256,Recovery
from .data import prepare
from .features import Features
from .model import load_model,tokenize,calibrate
from .train import seed_all,optimizer,update,payload,restore_rng
from .preflight import environment,smoke_stage
from .evaluate import development,final,plots

def source_manifest():
    root=Path(__file__).resolve().parent
    files={p.name:sha256(p) for p in sorted(root.iterdir()) if p.suffix in ('.py','.json','.txt')}
    files['tests/test_task2.py']=sha256(root.parent/'tests/test_task2.py')
    return {'files':files,'hash':digest(files),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'assignment_sha256':sha256(root.parent/'ATML-PA3.pdf'),'protocol':dataclasses.asdict(P)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--drive-root',default='/content/drive/MyDrive/ATML_PA3_Task2')
    parser.add_argument('--local-root',default='/content/atml_pa3_task2')
    parser.add_argument('--session-hours',type=float,default=5.)
    parser.add_argument('--reserve-minutes',type=float,default=15.)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    started=time.monotonic();deadline_at=started+args.session_hours*3600-args.reserve_minutes*60
    deadline=lambda:time.monotonic()>=deadline_at
    root=Path(args.drive_root);local=Path(args.local_root)
    root.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    source=source_manifest();run_id=digest({'source':source['hash'],'protocol':source['protocol']})[:20]
    out=root/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
    final_path=out/'final_verification.json'
    if final_path.exists():
        previous=json.loads(final_path.read_text())
        if previous.get('fully_completed'):
            for name,expected in previous['artifacts'].items():
                if sha256(out/name)!=expected: raise ValueError(f'Completed artifact changed: {name}')
            print(f'[ALREADY COMPLETE] Verified existing artifacts: {out}',flush=True)
            return
    atomic_json(out/'source_manifest.json',source)
    atomic_json(out/'final_verification.json',{'status':'not run','fully_completed':False})
    status={'status':'not run','checks':{}}
    timing=json.loads((out/'timing.json').read_text()) if (out/'timing.json').exists() else {}
    memory=json.loads((out/'memory.json').read_text()) if (out/'memory.json').exists() else {}
    def record():
        atomic_json(out/'preflight.json',status);atomic_json(out/'timing.json',timing);atomic_json(out/'memory.json',memory)
    record()
    try:
        status['environment']=environment(local,require_cuda=False);record()
        environment(local)
        environment_path=out/'environment.json'
        versions=status['environment']['versions']
        if environment_path.exists() and json.loads(environment_path.read_text())['versions']!=versions:
            raise RuntimeError('Dependency versions changed for this run; restore its recorded versions before resuming')
        atomic_json(environment_path,status['environment'])
        # Exercise only isolated Task 2 tests; do not import or mutate Task 1.
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_task2.py'],cwd=Path(__file__).resolve().parent.parent,check=True)
        status['checks']['unit_tests']='PASS';status['status']='CPU tested';record()
        from huggingface_hub import HfApi
        revisions_path=out/'model_revisions.json'
        if revisions_path.exists(): revisions=json.loads(revisions_path.read_text())
        else:
            revisions={key:HfApi().model_info(name,revision=rev).sha for key,name,rev in [('vision',P.vision,P.vision_revision),('language',P.language,P.language_revision)]}
            atomic_json(revisions_path,revisions)
        seed_all();start=time.monotonic()
        data_root=root/'data'/run_id
        data=prepare(data_root,root/'data'/'official_annotations')
        timing['data_setup_seconds']=time.monotonic()-start
        plan=schedule(data);plan_hash=digest(plan)
        atomic_json(out/'sampling_plan.json',{'hash':plan_hash,'updates':plan})
        identity=digest({'source':source['hash'],'protocol':source['protocol'],'data':data['hash'],'plan':plan_hash,'revisions':revisions,'versions':versions})
        atomic_json(out/'run_identity.json',{'identity':identity,'data_hash':data['hash'],'plan_hash':plan_hash,'revisions':revisions})
        atomic_json(out/'selected_data_manifest.json',{'path':str(data_root/'selected_data.json'),'hash':data['hash']})
        status['checks']['data_selection']='PASS';record()
        if deadline(): raise TimeoutError('Safe stop after data selection')
        # HF uses persistent cache: immutable snapshots are reused after reconnect.
        model_cache=root/'models'
        needed_ids={i for batch in plan for i in batch}|set(data['caption_dev'])
        for suite in ('ground_dev','ground_test','natural_dev','natural_test'):
            needed_ids.update(i for pair in data[suite] for i in pair)
        feature_store=Features(local/'features'/run_id,root/'features'/run_id,data,revisions['vision'],source['hash'],deadline,needed_ids)
        feature_store.prepare(model_cache)
        timing['features']=feature_store.stats;memory['features']=feature_store.stats
        status['checks']['feature_cache']='PASS';record()
        model=load_model(revisions['language'],model_cache)
        token_path=out/'tokenized_examples.json';token_marker=out/'tokenized_examples_manifest.json'
        from .storage import valid
        if token_marker.exists() and valid(token_path,json.loads(token_marker.read_text())):
            data['examples']=json.loads(token_path.read_text())
        else:
            data=tokenize(data,model.tokenizer)
            atomic_json(token_path,data['examples'])
            atomic_json(token_marker,{'bytes':token_path.stat().st_size,'sha256':sha256(token_path)})
        recovery=Recovery(local/'recovery'/run_id,out/'checkpoints',identity)
        state=recovery.load()
        benchmarks=[]
        # Isolate every stage, including LoRA, before committing expensive training.
        # Always rerun on a new runtime: the hardware/version allocation can differ.
        for stage in range(3):
            if deadline(): raise TimeoutError('Safe stop before CUDA preflight')
            bench=smoke_stage(model,feature_store,data,stage,out,identity)
            benchmarks.append(bench);print(f'[PASS] CUDA stage {stage+1}: {bench}',flush=True)
        memory['cuda_smoke']=benchmarks;timing['cuda_smoke']=benchmarks
        measured_training=sum(b['seconds_per_update']*n for b,n in zip(benchmarks,P.updates))
        timing['training_only_estimates_seconds']={'best':measured_training*.85,'central':measured_training,'conservative':measured_training*1.3}
        timing['whole_run_estimates_seconds']=None
        timing['unknown_costs']=['full development and final evaluation','future Drive I/O variability']
        timing['five_hour_target_verified']=False
        status['status']='GPU smoke tested';status['checks']['cuda_stages']='PASS';record()
        # Remove smoke-only adapters and restore pristine backbone/connector/RNG.
        del model;gc.collect();torch.cuda.empty_cache();seed_all()
        model=load_model(revisions['language'],model_cache)
        first=[data['examples'][i] for i in data['caption_train'][:64]]
        calibration=calibrate(model,first,feature_store.batch(first))
        atomic_json(out/'calibration.json',calibration)
        current=state['stage'] if state else 0
        model.stage(current);opt=optimizer(model,current);scaler=torch.amp.GradScaler('cuda')
        completed=0;logs=[]
        if state:
            if state['plan_hash']!=plan_hash: raise ValueError('Sampler plan mismatch')
            model.restore_trainable(state['trainable']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng']);completed=state['update'];logs=state['logs']
            if [r['update'] for r in logs]!=list(range(1,completed+1)): raise ValueError('Recovery log/update mismatch')
            if state['scheduler']['stage']!=current or state['scheduler']['completed_in_stage']!=completed-sum(P.updates[:current]):
                raise ValueError('Recovery scheduler position mismatch')
            if completed in (2250,3750,4500):
                from .storage import publish
                torch.save(model.trainable_state(),local/'stage_weights.pt')
                weight_record=publish(local/'stage_weights.pt',out/f'stage-{current+1}-weights.pt')
                atomic_json(out/f'stage-{current+1}-weights.json',weight_record)
        counts_path=out/'trainable_parameters.json'
        counts=json.loads(counts_path.read_text()) if counts_path.exists() else {}
        counts[str(current+1)]=model.audit(current)
        atomic_json(out/'trainable_parameters.json',counts)
        # A real forward/generation evaluation gate in addition to training smoke.
        from .evaluate import evaluate_rows
        fixture=[data['examples'][i] for p in data['ground_dev'][:4] for i in p]
        evaluation_gate_start=time.monotonic()
        for visual in (fixture,None):
            evaluate_rows(model,feature_store,list(map(str,range(8))),fixture,visual,generation=True,deadline=deadline)
        timing['evaluation_gate_seconds_per_answer']=(time.monotonic()-evaluation_gate_start)/16
        status['checks']['generation_and_no_image']='PASS';record()
        if args.smoke_only:
            atomic_json(out/'final_verification.json',{'status':'GPU smoke tested','fully_completed':False,'main_updates':completed})
            return
        torch.cuda.reset_peak_memory_stats()
        if completed==0: development(model,feature_store,data,0,out,deadline)
        baseline_path=out/'dev-0000.json'
        if baseline_path.exists():
            dev_seconds=json.loads(baseline_path.read_text())['seconds']
            final_estimate=1280*timing['evaluation_gate_seconds_per_answer']
            overhead=11*dev_seconds+18*max(b['checkpoint_seconds'] for b in benchmarks)+final_estimate
            setup=time.monotonic()-started
            central=setup+measured_training+overhead
            timing['whole_run_estimates_seconds']={'best':setup+(measured_training+overhead)*.85,'central':central,'conservative':setup+(measured_training+overhead)*1.4}
            print('[TIMING] Whole-run extrapolated seconds (not a completion guarantee):',timing['whole_run_estimates_seconds'],flush=True)
            timing['estimate_caveat']='Extrapolated from isolated measured stages/evaluation; not proof of five-hour completion. Runtime and Drive I/O may vary.'
            record()
        # If interruption occurred after a checkpoint but before its evaluation,
        # finish that exact model's pending evaluation before advancing.
        if completed and (completed%500==0 or completed in (2250,3750,4500)):
            development(model,feature_store,data,completed,out,deadline)
        backup=completed;update_times=[];checkpoint_times=timing.get('checkpoint_seconds',[])
        while completed<4500:
            if deadline():
                recovery.save(payload(model,opt,scaler,current,completed,plan_hash,logs))
                backup=completed;raise TimeoutError('Safe session stop; latest update verified on Drive')
            stage=stage_at(completed)
            if stage!=current:
                current=stage;model.stage(current)
                del opt,scaler
                gc.collect();torch.cuda.empty_cache()
                # Recheck real stage-transition gradients without changing training state.
                smoke_stage(model,feature_store,data,current,out,identity)
                opt=optimizer(model,current);scaler=torch.amp.GradScaler('cuda')
                counts[str(current+1)]=model.audit(current);atomic_json(out/'trainable_parameters.json',counts)
            batch=[data['examples'][i] for i in plan[completed]]
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            losses,norm,retries=update(model,opt,scaler,batch,feature_store,current,completed-sum(P.updates[:current])+1)
            torch.cuda.synchronize();duration=time.monotonic()-tick;update_times.append(duration);completed+=1
            old_peak=memory.get('training',{})
            memory['training']={'peak_allocated_bytes':max(old_peak.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated()),
                                'peak_reserved_bytes':max(old_peak.get('peak_reserved_bytes',0),torch.cuda.max_memory_reserved())}
            log={'update':completed,'stage':current+1,'loss':sum(losses)/16,'example_losses':losses,
                 'example_ids':plan[completed-1],'mixture_examples':dict(Counter(e['kind'] for e in batch)),
                 'gradient_norm':norm,'overflow_retries':retries,'seconds':duration,
                 'learning_rates':[g['lr'] for g in opt.param_groups]}
            logs.append(log)
            # Atomic per-update logs make reentry overwrite only the same replayed update.
            atomic_json(local/'logs'/run_id/f'{completed:04d}.json',log)
            if completed%250==0 or completed in (2250,3750,4500):
                tick=time.monotonic();rec=recovery.save(payload(model,opt,scaler,current,completed,plan_hash,logs))
                checkpoint_times.append(time.monotonic()-tick);backup=rec['update']
                timing['checkpoint_seconds']=checkpoint_times
                timing['training_seconds']=sum(r['seconds'] for r in logs)
                status['status']='partially trained';record()
                if completed in (2250,3750,4500):
                    torch.save(model.trainable_state(),local/'stage_weights.pt')
                    from .storage import publish
                    record_weights=publish(local/'stage_weights.pt',out/f'stage-{current+1}-weights.pt')
                    atomic_json(out/f'stage-{current+1}-weights.json',record_weights)
            if completed%500==0 or completed in (2250,3750,4500): development(model,feature_store,data,completed,out,deadline)
            remaining=sum(max(0,sum(P.updates[:i+1])-max(completed,sum(P.updates[:i])))*benchmarks[i]['seconds_per_update'] for i in range(3))
            heartbeat={'stage':current+1,'completed_updates':completed,'required_updates':4500,'example_presentations':completed*16,
                       'session_elapsed_seconds':time.monotonic()-started,'remaining_training_seconds_measured_estimate':remaining,
                       'whole_run_eta':None,'unknown_costs':['future I/O variability','final evaluation'],
                       'latest_verified_drive_update':backup,'gpu_allocated_bytes':torch.cuda.memory_allocated(),
                       'gpu_reserved_bytes':torch.cuda.memory_reserved(),'safe_stop_planned':True,'safe_stop_in_seconds':max(0,deadline_at-time.monotonic())}
            atomic_json(local/'heartbeat.json',heartbeat)
            if completed%25==0 or completed in (2250,3750,4500): atomic_json(out/'heartbeat.json',heartbeat)
            print(f"[TRAIN] stage={current+1} update={completed}/4500 loss={log['loss']:.5f} elapsed={heartbeat['session_elapsed_seconds']:.0f}s training_ETA~{remaining:.0f}s backup={backup} reserved={heartbeat['gpu_reserved_bytes']/2**30:.2f}GiB safe_stop_in={heartbeat['safe_stop_in_seconds']:.0f}s",flush=True)
        atomic_json(out/'training_logs.json',logs)
        timing['training_seconds']=sum(r['seconds'] for r in logs);timing['checkpoint_seconds']=checkpoint_times
        timing['five_hour_target_verified']=False
        timing['remaining_training_estimates_seconds']={'best':0,'central':0,'conservative':0}
        record();plots(logs,out)
        torch.cuda.reset_peak_memory_stats()
        final_start=time.monotonic()
        final(model,feature_store,data,out,deadline)
        timing['final_evaluation_session_seconds']=time.monotonic()-final_start
        memory['final_evaluation']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        status['status']='fully completed';record()
        verify_final(out,data,logs)
        print(f'[COMPLETE] Verified full protocol: {out}',flush=True)
    except TimeoutError as exc:
        status['safe_stop']=str(exc);record()
        # Latest completed checkpoint was already durable; work since it may replay.
        atomic_json(out/'final_verification.json',{'status':'partially trained' if (out/'checkpoints/latest.json').exists() else status['status'],
                                                 'fully_completed':False,'safe_stop':str(exc)})
        print(f'[SAFE STOP] {exc}. Next Run All resumes verified compatible state: {out}',flush=True)
    except Exception as exc:
        if status['status']=='fully completed': status['status']='partially trained'
        status['failure']={'type':type(exc).__name__,'message':str(exc)};record()
        atomic_json(out/'final_verification.json',{'status':status['status'],'fully_completed':False,'failure':status['failure']})
        raise

def verify_final(out,data,logs):
    expected={0,500,1000,1500,2000,2250,2500,3000,3500,3750,4000,4500}
    if [r['update'] for r in logs]!=list(range(1,4501)): raise ValueError('Missing/duplicate main updates')
    if sum(len(r['example_ids']) for r in logs)!=72000: raise ValueError('Wrong example presentations')
    for u in expected:
        if not (out/f'dev-{u:04d}.json').exists(): raise ValueError(f'Missing development evaluation {u}')
    for suite,n in [('ground_test',128),('natural_test',192)]:
        for condition in ('correct','complementary','mismatched','no_image'):
            rows=json.loads((out/f'{suite}-{condition}.json').read_text())
            expected_ids={i for pair in data[suite] for i in pair}
            if len(rows)!=n or {r['example_id'] for r in rows}!=expected_ids: raise ValueError('Final evaluation incomplete')
            for row in rows:
                if row['reference']!=data['examples'][row['example_id']]['answer']: raise ValueError('Changed original reference')
                if condition=='no_image' and row['input_image_id'] is not None: raise ValueError('No-image control contains an image')
    required=['metrics.json','training_logs.json','learning_curves.png','calibration.json','source_manifest.json',
              'run_identity.json','environment.json','preflight.json','timing.json','memory.json','sampling_plan.json',
              'ground_test-qualitative.html','natural_test-qualitative.html','trainable_parameters.json']
    for name in required:
        if not (out/name).is_file(): raise ValueError(f'Missing required artifact {name}')
    for stage in (1,2,3):
        if not (out/f'stage-{stage}-weights.pt').is_file(): raise ValueError(f'Missing stage {stage} weights')
    rec=json.loads((out/'checkpoints/latest.json').read_text())
    from .storage import valid
    if rec['update']!=4500 or not valid(out/'checkpoints'/rec['file'],rec): raise ValueError('Final recovery not verified')
    artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='final_verification.json'}
    atomic_json(out/'final_verification.json',{'status':'fully completed','fully_completed':True,'main_updates':4500,
                                             'example_presentations':72000,'artifacts':artifacts})

if __name__=='__main__': main()
