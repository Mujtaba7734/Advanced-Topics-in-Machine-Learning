"""Checksummed two-phase publication. A partial transfer never becomes latest."""
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with open(tmp,'w') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def valid(path, record):
    path = Path(path)
    return path.is_file() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']

def publish(local, destination):
    local, destination = Path(local), Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    record = {'bytes':local.stat().st_size,'sha256':sha256(local),'file':destination.name}
    if valid(destination,record):
        return record
    tmp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
    shutil.copyfile(local,tmp)
    if not valid(tmp,record):
        raise IOError(f'Copy verification failed: {tmp}')
    os.replace(tmp,destination)
    return record

class Recovery:
    def __init__(self, local, durable, identity):
        self.local,self.durable = Path(local),Path(durable)
        self.identity = identity
        self.local.mkdir(parents=True,exist_ok=True); self.durable.mkdir(parents=True,exist_ok=True)
    def save(self, state):
        import torch
        required = {'trainable','optimizer','scheduler','scaler','rng','update','stage','plan_hash','logs'}
        if not required <= state.keys():
            raise ValueError('Incomplete recovery payload')
        state = dict(state,identity=self.identity)
        name = f'recovery-{state["update"]:04d}-{uuid.uuid4().hex[:8]}.pt'
        local = self.local / name
        torch.save(state, local)
        check = torch.load(local,map_location='cpu',weights_only=False)
        if check.keys() != state.keys() or check['identity'] != self.identity:
            raise IOError('Invalid locally serialized checkpoint')
        rec = publish(local,self.durable/name)
        # Deserialize the destination before publishing its pointer.
        dest = torch.load(self.durable/name,map_location='cpu',weights_only=False)
        if dest['update'] != state['update'] or dest['identity'] != self.identity:
            raise IOError('Invalid durable checkpoint')
        rec.update(update=state['update'],identity=self.identity,stage=state['stage'])
        latest = self.durable/'latest.json'
        if latest.exists():
            old=json.loads(latest.read_text())
            if valid(self.durable/old['file'],old): atomic_json(self.durable/'previous.json',old)
        atomic_json(latest,rec)
        atomic_json(self.durable/'checkpoint_manifest.json',rec)
        # Keep only current and previous verified snapshots, never other runs/files.
        keep={name}
        previous=self.durable/'previous.json'
        if previous.exists(): keep.add(json.loads(previous.read_text())['file'])
        for root in (self.local,self.durable):
            for path in root.glob('recovery-*.pt'):
                if path.name not in keep: path.unlink()
        return rec
    def load(self):
        import torch
        found=False
        for pointer in ('latest.json','previous.json'):
            path=self.durable/pointer
            if not path.exists(): continue
            found=True
            rec=json.loads(path.read_text())
            if rec['identity'] != self.identity: raise ValueError('Source/protocol/data identity mismatch')
            if not valid(self.durable/rec['file'],rec): continue
            publish(self.durable/rec['file'],self.local/rec['file'])
            state=torch.load(self.local/rec['file'],map_location='cpu',weights_only=False)
            if state['identity'] != self.identity: raise ValueError('Recovery identity mismatch')
            return state
        if found: raise IOError('No verified recovery remains; refusing silent restart')
        return None
