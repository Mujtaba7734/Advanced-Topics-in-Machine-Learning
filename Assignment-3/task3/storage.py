"""Verified immutable objects + atomic latest/previous pointers, rooted on Drive."""
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from .protocol import digest

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()

def atomic(path, data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.partial')
 with open(temp,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
 expected=hashlib.sha256(data).hexdigest()
 if sha256(temp)!=expected: raise IOError('Write verification failed')
 os.replace(temp,path)
 if sha256(path)!=expected: raise IOError('Final readback failed')
 return expected

def write_json(path,value):
 return atomic(path,json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode())

class SafeStop(Exception): pass

class Store:
 def __init__(self,root,identity):
  self.root=Path(root);self.identity=identity;self.root.mkdir(parents=True,exist_ok=True)
 def put(self,key,value):
  body=dict(identity=self.identity,value=value)
  sha=digest(body); path=self.root/'objects'/(sha+'.json')
  write_json(path,body)
  record=dict(object=str(path.relative_to(self.root)),sha256=sha256(path),identity=self.identity,key=key)
  pointer=self.root/'pointers'/(key+'.json')
  if pointer.exists():
   # Only promote a verified current pointer to previous.
   try:
    self.get(key); atomic(pointer.with_suffix('.previous.json'),pointer.read_bytes())
   except (ValueError,IOError,json.JSONDecodeError): pass
  write_json(pointer,record)
  if self.get(key)!=value: raise IOError('Checkpoint readback mismatch')
  return str(pointer)
 def get(self,key):
  pointer=self.root/'pointers'/(key+'.json')
  for p in (pointer,pointer.with_suffix('.previous.json')):
   if not p.exists(): continue
   try:
    record=json.loads(p.read_text());obj=self.root/record['object']
    if record['identity']!=self.identity or record['key']!=key or sha256(obj)!=record['sha256']: continue
    body=json.loads(obj.read_text())
    if body['identity']==self.identity: return body['value']
   except (OSError,ValueError,KeyError): continue
  return None

class Monitor:
 def __init__(self,root,hours=5,reserve_minutes=15):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  self.start=time.monotonic();self.deadline=self.start+hours*3600-reserve_minutes*60
  self.lock=threading.RLock();self.current={};self.done=threading.Event()
  self.thread=threading.Thread(target=self._heartbeat,daemon=True);self.thread.start()
 def log(self,kind,message,**fields):
  record=dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),event=kind,message=message,**fields)
  line=f"{record['time']} [{kind}] {message} "+json.dumps(fields,allow_nan=False)
  with self.lock:
   print(line,flush=True)
   logs=[('progress.jsonl',json.dumps(record,allow_nan=False))]
   if os.environ.get('TASK3_NOTEBOOK_TEE')!='1':logs.append(('full_pipeline.log',line))
   for name,text in logs:
    with open(self.root/name,'a') as f:f.write(text+'\n');f.flush()
   self.current=record;write_json(self.root/'heartbeat.json',record)
 def _heartbeat(self):
  while not self.done.wait(30):
   with self.lock:
    self.log('HEARTBEAT','Worker active',last_phase=self.current.get('message'),session_elapsed_s=round(time.monotonic()-self.start,1))
 def check(self,estimated_seconds=0):
  if time.monotonic()+estimated_seconds>=self.deadline: raise SafeStop('Session time budget reached; saved work is reusable')
 def status(self,state,**kwargs): write_json(self.root/'status.json',dict(state=state,**kwargs))
 def close(self):self.done.set();self.thread.join(timeout=2)
