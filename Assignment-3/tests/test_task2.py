"""Task 2 only: CPU tests are not CUDA verification."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from task2.protocol import P,lr_factor,schedule,digest
from task2.metrics import normalize,soft_score,consensus,bootstrap,pair_vectors,summarize
from task2.storage import Recovery,atomic_json,publish,valid
from task2.model import VLM
from task2.evaluate import controlled

class Tokenizer:
    unk_token_id=None
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return 'U'+messages[0]['content']+'A'
    def encode(self,text,**kwargs): return [2] if text=='U' else [3]
    def convert_tokens_to_ids(self,text): return 4

def tiny():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    config=Qwen3Config(vocab_size=32,hidden_size=1024,intermediate_size=32,num_hidden_layers=1,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=16,attention_dropout=0.)
    return VLM(Qwen3ForCausalLM(config),Tokenizer())

def examples():
    return [{'question_tokens':[5,6], 'answer_tokens':[7,8]}, {'question_tokens':[5],'answer_tokens':[9]}]

def test_normalization():
    assert normalize('The TWO cats!')=='2 cats'
    assert normalize('1,000')=='1000'
    assert normalize('3.14')=='3.14'
    assert normalize('dont')=="don't"

@pytest.mark.parametrize('n,expected',[(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)])
def test_soft_consensus(n,expected):
    assert soft_score('yes',['yes']*n+['no']*(10-n))==pytest.approx(expected)
    assert soft_score('',['']*10)==0

def test_consensus_original():
    answer,n=consensus(['Two']*4+['2']*4+['three']*2)
    assert answer=='Two' and n==8

def test_schedule():
    pools={'caption_train':list(range(18000)),'presence':[[18000,18001]],'counting':[[18002,18003]],'natural_train':[[18004,18005]]}
    a=schedule(pools);b=schedule(pools)
    assert a==b and len(a)==4500 and sum(map(len,a))==72000
    flat=[i for batch in a[:2250] for i in batch]
    assert np.array_equal(np.bincount(flat),np.full(18000,2))
    assert sorted(flat[:18000])==list(range(18000))
    for batch in a[2250:]:
        for i in range(0,16,2):
            if batch[i]>=18000: assert batch[i+1]==batch[i]+1
    presence=sum(x==18000 for batch in a[2250:3750] for x in batch)/12000
    assert .48<presence<.52
    natural=sum(x==18004 for batch in a[3750:] for x in batch)/6000
    assert .68<natural<.72

@pytest.mark.parametrize('total',[2250,1500,750])
def test_lr(total):
    import math
    assert lr_factor(math.ceil(total*.05),total)==1
    assert lr_factor(total,total)==pytest.approx(.1)
    assert lr_factor(1,total)>0

def test_pack_masks_and_no_image():
    model=tiny();ex=examples();f=torch.randn(2,196,768)
    inputs,labels=model.pack(ex,f)
    bare,barelabels=model.pack(ex,None)
    assert inputs['inputs_embeds'].shape[1]-bare['inputs_embeds'].shape[1]==196
    assert (labels!=-100).sum().item()==5
    assert labels[0,-3:].tolist()==[7,8,4]
    assert labels[1,-2:].tolist()==[9,4]
    assert (labels[inputs['attention_mask']==0]==-100).all()
    assert inputs['position_ids'][1,0]==0
    _,nlllabels=model.pack(ex,f,include_eot=False)
    assert (nlllabels!=-100).sum().item()==3

def test_loss_matches_full_logits_and_gradient():
    torch.manual_seed(2);model=tiny();model.train();ex=examples();f=torch.randn(2,196,768)
    inputs,labels=model.pack(ex,f)
    output=model.language(**inputs,use_cache=False)
    full=torch.nn.functional.cross_entropy(output.logits[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(2,-1)
    expected=full.sum(1)/(labels[:,1:]!=-100).sum(1)
    actual=model.losses(ex,f)
    torch.testing.assert_close(actual,expected)
    actual.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.connector.parameters())
    assert all(p.grad is None for p in model.language.parameters())

def test_accumulation_example_weighting():
    torch.manual_seed(3);model=tiny();model.eval();ex=examples();f=torch.randn(2,196,768)
    model.losses(ex,f).mean().backward()
    expected=[p.grad.clone() for p in model.connector.parameters()]
    model.zero_grad(set_to_none=True)
    for i in range(2): (model.losses(ex[i:i+1],f[i:i+1]).sum()/2).backward()
    for p,g in zip(model.connector.parameters(),expected): torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)

def test_lora_trainability_and_gradients():
    model=tiny();model.stage(1);model.train()
    model.losses(examples(),torch.randn(2,196,768)).mean().backward()
    assert model.audit(1)['lora']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n)
    assert all(not p.requires_grad for n,p in model.language.named_parameters() if 'lora_' not in n)

def checkpoint(update=1):
    return dict(trainable={'x':torch.tensor(3.)},optimizer={},scheduler={},scaler={},rng={},update=update,stage=0,plan_hash='p',logs=[])

def test_recovery_and_interrupted_copy(tmp_path,monkeypatch):
    r=Recovery(tmp_path/'local',tmp_path/'drive','id');r.save(checkpoint())
    pointer=(tmp_path/'drive/latest.json').read_text()
    import task2.storage as storage
    def broken(src,dst):
        Path(dst).write_bytes(b'partial');raise IOError('Simulated disconnection')
    monkeypatch.setattr(storage.shutil,'copyfile',broken)
    with pytest.raises(IOError): r.save(checkpoint(2))
    assert (tmp_path/'drive/latest.json').read_text()==pointer
    assert r.load()['update']==1

def test_recovery_previous_and_identity(tmp_path):
    r=Recovery(tmp_path/'local',tmp_path/'drive','id');r.save(checkpoint(1));r.save(checkpoint(2))
    latest=json.loads((tmp_path/'drive/latest.json').read_text())
    (tmp_path/'drive'/latest['file']).write_bytes(b'corrupt')
    assert r.load()['update']==1
    with pytest.raises(ValueError): Recovery(tmp_path/'local',tmp_path/'drive','wrong').load()

def test_bootstrap_complete_pairs():
    a=np.array([[1,1,0,0,2,.1],[0,0,1,0,4,.2]])
    result=bootstrap(a)
    assert result['accuracy']['mean']==.5 and result['accuracy']['ci95']==[0.,1.]
    assert result==bootstrap(a)
    with pytest.raises(ValueError): pair_vectors([dict(pair_id='only-one')])

def test_controls_keep_original_references():
    ex={str(i):dict(image_id=i,answer=str(i)) for i in range(4)}
    data={'examples':ex,'test':[['0','1'],['2','3']],'test_mismatch':{'0':'2','1':'3','2':'0','3':'1'}}
    ids,original,visual=controlled(data,'test','complementary')
    assert original[0]['answer']=='0' and visual[0]['image_id']==1
    assert controlled(data,'test','no_image')[2] is None
    assert controlled(data,'test','mismatched')[2][0]['image_id']==2

def test_resume_adam_rng_cpu(tmp_path):
    from task2.train import rng_state,restore_rng
    torch.manual_seed(6304);m=torch.nn.Linear(3,2);opt=torch.optim.AdamW(m.parameters())
    def step():
        opt.zero_grad();loss=m(torch.randn(5,3)).square().mean();loss.backward();opt.step();return loss.item()
    step();saved={'weights':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng_state()}
    expected=step();weights=copy.deepcopy(m.state_dict())
    m.load_state_dict(saved['weights']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    assert step()==expected
    for k,v in m.state_dict().items(): torch.testing.assert_close(v,weights[k],rtol=0,atol=0)

def test_siglip_final_norm_once():
    from transformers import SiglipVisionConfig,SiglipVisionModel
    vision=SiglipVisionModel(SiglipVisionConfig(hidden_size=768,intermediate_size=16,num_hidden_layers=1,
                            num_attention_heads=4,image_size=224,patch_size=16)).eval()
    calls=[]
    hook=vision.vision_model.post_layernorm.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad(): features=vision(pixel_values=torch.zeros(1,3,224,224)).last_hidden_state
    hook.remove()
    assert features.shape==(1,196,768) and len(calls)==1

def test_feature_cache_roundtrip_and_identity(tmp_path):
    array=np.arange(196*768,dtype=np.float32).reshape(1,196,768)/10000
    np.save(tmp_path/'local.npy',array.astype(np.float16))
    rec=publish(tmp_path/'local.npy',tmp_path/'durable.npy')
    assert valid(tmp_path/'durable.npy',rec)
    assert np.array_equal(np.load(tmp_path/'durable.npy'),array.astype(np.float16))
    (tmp_path/'durable.npy').write_bytes(b'corrupted')
    assert not valid(tmp_path/'durable.npy',rec)
    assert digest({'revision':'a','split':'train2014'})!=digest({'revision':'a','split':'val2014'})

def test_natural_real_schema_and_consensus_filter():
    from task2.data import natural_candidates
    qs={'questions':[dict(question_id=i,image_id=i,question='What color?') for i in range(4)]}
    annotations={'annotations':[dict(question_id=i,image_id=i,answer_type='other',
        answers=[{'answer':'red' if i%2==0 else 'blue'}]*n+[{'answer':'green'}]*(10-n)) for i,n in enumerate([8,10,7,10])]}
    pairs=natural_candidates(qs,annotations,[[0,1],[1,0],[2,3]],'train2014')
    assert len(pairs)==1
    assert pairs[0][0]['answer']=='red' and pairs[0][1]['answer']=='blue'
    assert len(pairs[0][0]['answers'])==10

def test_pair_matched_advantage_and_nll_sign():
    rows=[]
    for pid in ['a','b']:
        for i in range(2): rows.append(dict(pair_id=pid,kind='presence',prediction=['yes','no'][i],exact=True,score=1.,nll=2.,seconds=.1))
    swapped=[dict(r,score=0.,exact=False,nll=5.) for r in rows]
    result=summarize({'correct':rows,'complementary':swapped})
    assert result['correct_image_advantages']['accuracy']['mean']==1
    assert result['correct_image_advantages']['reference_nll']['mean']==3
    assert result['correct']['pair_accuracy']['mean']==1

def test_generation_passes_dynamic_positions():
    model=tiny();seen={}
    def fake_generate(**kwargs):
        seen.update(kwargs);return torch.tensor([[7,4],[8,4]])
    model.language.generate=fake_generate
    model.tokenizer.batch_decode=lambda ids,**kwargs:['first','second']
    assert model.generate_answers(examples(),None)==['first','second']
    assert 'position_ids' not in seen and seen['do_sample'] is False and seen['max_new_tokens']==16
    assert seen['attention_mask'].shape[0]==2

def test_content_caps_and_calibration_split():
    from task2.model import tokenize,calibrate
    class LongTokenizer:
        def encode(self,text,**kw): return list(range(1,101))
    data={'examples':{'a':{'question':'q','answer':'a','kind':'caption'},'b':{'question':'q','answer':'a','kind':'presence'}}}
    tokenize(data,LongTokenizer())
    assert len(data['examples']['a']['question_tokens'])==80
    assert len(data['examples']['a']['answer_tokens'])==64
    assert len(data['examples']['b']['answer_tokens'])==16
    with pytest.raises(ValueError,match='Calibration leakage'):
        calibrate(None,[{'split':'val2014'}],None)

def test_grounding_pair_labels():
    from task2.data import grounding
    annotations=[]
    for image,n in enumerate([0,1,2,3,4,1,2,3,4,0]):
        annotations.extend({'image_id':image,'category_id':1} for _ in range(n))
    data={'categories':[{'id':1,'name':'cat'}],'annotations':annotations}
    pools=grounding(data,set(range(10)),'train2014',np.random.default_rng(6304),4)
    for a,b in pools['presence']:
        assert a['answer']=='yes' and b['answer']=='no' and a['question']==b['question']
        assert a['source']['annotated_count']>0 and b['source']['annotated_count']==0
    for a,b in pools['counting']:
        assert a['answer']!=b['answer'] and 1<=int(a['answer'])<=4 and 1<=int(b['answer'])<=4
        assert a['question']=='How many cats can you see?'

def test_feature_image_alignment_and_selection(tmp_path):
    from task2.features import Features
    data={'examples':{'b':{'split':'train2014','image_id':2},'a':{'split':'train2014','image_id':1},'unused':{'split':'train2014','image_id':3}}}
    store=Features(tmp_path/'local',tmp_path/'durable',data,'revision','source',selected_example_ids={'a','b'})
    assert len(store.keys)==2
    np.save(store.local/f'{store.fingerprint[:16]}-0000.npy',np.stack([np.full((196,768),1,np.float16),np.full((196,768),2,np.float16)]))
    batch=store.batch([data['examples']['b'],data['examples']['a']],device='cpu')
    assert batch[:,0,0].tolist()==[2,1]
