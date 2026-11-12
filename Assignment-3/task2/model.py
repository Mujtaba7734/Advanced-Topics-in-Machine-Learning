"""Visual insertion and answer-only, example-weighted causal loss."""
import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from .protocol import P

class Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(768,1024),nn.GELU(),nn.Linear(1024,1024))
        self.register_buffer('scale',torch.tensor(1.,dtype=torch.float32))
    def forward(self,x): return self.net(x)*self.scale

class VLM(nn.Module):
    def __init__(self, language, tokenizer):
        super().__init__(); self.language=language; self.tokenizer=tokenizer
        self.connector=Connector(); self.has_lora=False
        self.language.requires_grad_(False)
        self.language.config.use_cache=False
        self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # Split the native, thinking-disabled template at a unique text marker.
        marker='PA3_VISUAL_INSERTION_MARKER'
        rendered=tokenizer.apply_chat_template([{'role':'user','content':marker}],tokenize=False,
                                               add_generation_prompt=True,enable_thinking=False)
        if rendered.count(marker)!=1: raise ValueError('Unexpected Qwen template')
        before,after=rendered.split(marker)
        self.before=tokenizer.encode(before,add_special_tokens=False)
        self.after=tokenizer.encode(after,add_special_tokens=False)
        self.eot=tokenizer.convert_tokens_to_ids('<|im_end|>')
        if self.eot is None or self.eot==tokenizer.unk_token_id: raise ValueError('Missing Qwen end-of-turn')
    def stage(self,index):
        if index>0 and not self.has_lora:
            from peft import LoraConfig,get_peft_model
            self.language=get_peft_model(self.language,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
                bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
            self.has_lora=True
        for name,p in self.language.named_parameters():
            p.requires_grad_(index>0 and 'lora_' in name)
            if p.requires_grad: p.data=p.data.float()
        self.connector.float().requires_grad_(True)
        self.audit(index)
    def audit(self,index):
        for name,p in self.named_parameters():
            expected=name.startswith('connector.') or (index>0 and 'lora_' in name)
            if p.requires_grad!=expected: raise AssertionError(f'Unexpected trainability {name}')
            if p.requires_grad and p.dtype!=torch.float32: raise AssertionError(f'Non FP32 trainable {name}')
            if not p.requires_grad and p.is_cuda and p.dtype!=torch.float16: raise AssertionError(f'Non FP16 backbone {name}')
        return {k:sum(p.numel() for n,p in self.named_parameters() if p.requires_grad and (n.startswith('connector.') if k=='connector' else 'lora_' in n)) for k in ('connector','lora')}
    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.state_dict().items() if n.startswith('connector.') or 'lora_' in n}
    def restore_trainable(self,state):
        current=self.state_dict()
        if set(state)!={n for n in current if n.startswith('connector.') or 'lora_' in n}: raise ValueError('Trainable state keys mismatch')
        for n,v in state.items(): current[n].copy_(v)
    def pack(self,examples,features=None,generation=False,include_eot=True):
        device=next(self.parameters()).device
        embed=self.language.get_input_embeddings()
        sequences=[]; targets=[]
        for j,e in enumerate(examples):
            q=e['question_tokens']; a=e['answer_tokens']
            prefix=embed(torch.tensor(self.before,device=device))
            if features is not None:
                prefix=torch.cat((prefix,self.connector(features[j]).to(prefix.dtype)),0)
            suffix=embed(torch.tensor(q+self.after,device=device))
            prompt=torch.cat((prefix,suffix),0)
            if generation:
                sequences.append(prompt); targets.append([-100]*len(prompt))
            else:
                answer=a+[self.eot]
                sequences.append(torch.cat((prompt,embed(torch.tensor(answer,device=device))),0))
                targets.append([-100]*len(prompt)+a+([self.eot] if include_eot else [-100]))
        length=max(map(len,sequences)); hidden=sequences[0].shape[-1]
        # Left padding supports batched autoregressive decoding; also valid for causal CE.
        padded=[]; masks=[]; labels=[]
        for seq,tgt in zip(sequences,targets):
            n=length-len(seq)
            padded.append(F.pad(seq,(0,0,n,0)))
            masks.append([0]*n+[1]*len(seq)); labels.append([-100]*n+tgt)
        mask=torch.tensor(masks,device=device)
        positions=(mask.cumsum(-1)-1).clamp_min(0)
        return dict(inputs_embeds=torch.stack(padded),attention_mask=mask,position_ids=positions),torch.tensor(labels,device=device)
    def losses(self,examples,features,include_eot=True):
        inputs,labels=self.pack(examples,features,include_eot=include_eot)
        # Compute transformer states once, project only supervised positions. Avoids
        # B*sequence*151936 vocabulary logits (several GB for visual prefixes).
        base=self.language.get_base_model() if self.has_lora else self.language
        hidden=base.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,:-1]
        labels=labels[:,1:]; keep=labels!=-100
        counts=keep.sum(1)
        if (counts==0).any(): raise ValueError('Empty supervised answer')
        selected=hidden[keep]; gold=labels[keep]
        # Chunk the vocabulary projection with checkpoint recomputation in training.
        # This retains exact CE/gradients without retaining all FP32 logits.
        def token_ce(h,y): return F.cross_entropy(base.lm_head(h).float(),y,reduction='none')
        losses=[]
        for h,y in zip(selected.split(32),gold.split(32)):
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                losses.append(checkpoint(token_ce,h,y,use_reentrant=False))
            else: losses.append(token_ce(h,y))
        values=torch.cat(losses)
        rows=torch.arange(len(examples),device=labels.device)[:,None].expand_as(labels)[keep]
        sums=torch.zeros(len(examples),device=values.device).scatter_add(0,rows,values)
        return sums/counts
    @torch.no_grad()
    def generate_answers(self,examples,features):
        inputs,_=self.pack(examples,features,generation=True)
        # Generation derives and advances positions from the growing attention mask.
        inputs.pop('position_ids')
        ids=self.language.generate(**inputs,max_new_tokens=16,do_sample=False,use_cache=True,
                                   pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.eot)
        return self.tokenizer.batch_decode(ids,skip_special_tokens=True)

def tokenize(data,tokenizer):
    for e in data['examples'].values():
        e['question_tokens']=tokenizer.encode(e['question'],add_special_tokens=False)[:80]
        e['answer_tokens']=tokenizer.encode(e['answer'],add_special_tokens=False)[:64 if e['kind']=='caption' else 16]
        if not e['answer_tokens']: raise ValueError('Empty reference target')
    return data

def load_model(revision,cache_dir,device='cuda'):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(P.language,revision=revision,cache_dir=cache_dir)
    language=AutoModelForCausalLM.from_pretrained(P.language,revision=revision,cache_dir=cache_dir,
                    torch_dtype=torch.float16 if device=='cuda' else torch.float32,attn_implementation='sdpa')
    return VLM(language,tokenizer).to(device)

@torch.no_grad()
def calibrate(model,examples,features):
    # RMS over 64 fixed TRAINING examples, prior to any training, no evaluation data.
    if any(e['split']!='train2014' for e in examples): raise ValueError('Calibration leakage')
    visual=[]; text=[]
    for e,f in zip(examples,features):
        visual.append(model.connector(f.float()).float().square().mean())
        ids=torch.tensor(e['question_tokens']+e['answer_tokens'],device=f.device)
        text.append(model.language.get_input_embeddings()(ids).float().square().mean())
    v=torch.stack(visual).mean().sqrt(); t=torch.stack(text).mean().sqrt()
    scale=t/v
    if not torch.isfinite(scale) or scale<=0: raise ValueError('Invalid calibration')
    model.connector.scale.copy_(scale)
    return {'rule':'training-only ratio of text-embedding RMS to initial connector-output RMS; first 64 selected captions',
            'text_rms':t.item(),'visual_rms':v.item(),'scale':scale.item()}
