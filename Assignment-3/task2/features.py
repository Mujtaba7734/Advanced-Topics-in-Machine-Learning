"""64-image validated FP16 shards, local memory maps for hot training reads."""
from pathlib import Path
import gc
import json
import time
import numpy as np
import torch
from PIL import Image
from .protocol import P,digest
from .storage import atomic_json,publish,valid
from .data import image_path

class Features:
    def __init__(self,local,durable,data,revision,source_hash,deadline=lambda:False,selected_example_ids=None):
        self.local=Path(local);self.durable=Path(durable)
        self.local.mkdir(parents=True,exist_ok=True); self.durable.mkdir(parents=True,exist_ok=True)
        examples=data['examples'].values() if selected_example_ids is None else (data['examples'][i] for i in sorted(selected_example_ids))
        self.images={f"{e['split']}/{e['image_id']}":e for e in examples}
        self.keys=sorted(self.images); self.index={k:(i//64,i%64) for i,k in enumerate(self.keys)}
        self.revision=revision; self.deadline=deadline; self.maps={}
        self.metadata={'model':P.vision,'revision':revision,'source_hash':source_hash,
                       'features':'last_hidden_state, post_layernorm already applied exactly once',
                       'dtype':'float16','shape':[196,768],'order':'14x14 row major','keys':self.keys}
        self.fingerprint=digest(self.metadata); self.stats={}
    def prepare(self,model_cache):
        from transformers import AutoImageProcessor,SiglipVisionModel
        from concurrent.futures import ThreadPoolExecutor
        processor=AutoImageProcessor.from_pretrained(P.vision,revision=self.revision,cache_dir=model_cache,use_fast=False)
        self.metadata['processor']=processor.to_dict(); self.fingerprint=digest(self.metadata)
        model=None; start=time.monotonic(); built=0; copied=0
        downloads=ThreadPoolExecutor(max_workers=8)
        torch.cuda.reset_peak_memory_stats()
        try:
            for shard in range((len(self.keys)+63)//64):
                if self.deadline(): raise TimeoutError('Safe stop during feature caching; completed shards are durable')
                name=f'{self.fingerprint[:16]}-{shard:04d}.npy'
                local=self.local/name; durable=self.durable/name; marker=durable.with_suffix('.json')
                keys=self.keys[shard*64:(shard+1)*64]
                record=json.loads(marker.read_text()) if marker.exists() else None
                if record and record.get('keys')==keys and valid(durable,record):
                    if not valid(local,record): publish(durable,local);copied+=1
                else:
                    if model is None:
                        model=SiglipVisionModel.from_pretrained(P.vision,revision=self.revision,cache_dir=model_cache,
                                                              torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
                    tensors=[];image_records={}
                    for offset in range(0,len(keys),8):
                        ims=[]
                        batch_keys=keys[offset:offset+8]
                        paths=list(downloads.map(lambda k:image_path(self.images[k],self.local/'images'),batch_keys))
                        for k,path in zip(batch_keys,paths):
                            image_records[k]=json.loads(path.with_suffix('.jpg.json').read_text())
                            with Image.open(path) as im: ims.append(im.convert('RGB'))
                        pixels=processor(images=ims,return_tensors='pt')['pixel_values'].cuda().half()
                        if built==0 and offset==0:
                            again=processor(images=ims,return_tensors='pt')['pixel_values']
                            torch.testing.assert_close(pixels.cpu(),again.half(),rtol=0,atol=0)
                            calls=[]
                            hook=model.vision_model.post_layernorm.register_forward_hook(lambda *a:calls.append(1))
                        else: hook=None
                        with torch.no_grad():
                            f=model(pixel_values=pixels).last_hidden_state
                        if hook is not None:
                            hook.remove()
                            if len(calls)!=1: raise AssertionError('SigLIP final layernorm must run exactly once')
                        if f.shape[1:]!=(196,768) or not torch.isfinite(f).all(): raise ValueError('Invalid SigLIP patch features')
                        tensors.append(f.cpu().numpy())
                    array=np.concatenate(tensors).astype(np.float16)
                    with open(local,'wb') as fp: np.save(fp,array,allow_pickle=False)
                    record=publish(local,durable);record.update(keys=keys,metadata=self.metadata,image_sources=image_records)
                    atomic_json(marker,record);built+=len(keys)
                a=np.load(local,mmap_mode='r',allow_pickle=False)
                if a.shape!=(len(keys),196,768) or a.dtype!=np.float16 or not np.isfinite(a).all():
                    raise ValueError(f'Corrupt feature array {local}')
                print(f'[CACHE] shard {shard+1}/{(len(self.keys)+63)//64}, built images {built}',flush=True)
            self.stats={'seconds':time.monotonic()-start,'images_computed':built,'shards_copied':copied,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            atomic_json(self.durable/'feature_manifest.json',dict(self.metadata,fingerprint=self.fingerprint,stats=self.stats))
        finally:
            downloads.shutdown(wait=True)
            del model;gc.collect();torch.cuda.empty_cache()
    def batch(self,examples,device='cuda'):
        arrays=[]
        for e in examples:
            shard,row=self.index[f"{e['split']}/{e['image_id']}"]
            if shard not in self.maps:
                self.maps[shard]=np.load(self.local/f'{self.fingerprint[:16]}-{shard:04d}.npy',mmap_mode='r',allow_pickle=False)
            arrays.append(self.maps[shard][row])
        tensor=torch.from_numpy(np.stack(arrays))
        if device=='cuda': tensor=tensor.pin_memory()
        return tensor.to(device,non_blocking=device=='cuda')
