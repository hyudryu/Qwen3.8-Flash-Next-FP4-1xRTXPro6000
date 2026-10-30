#!/usr/bin/env python3
"""Compare aligned forced-token full-vocabulary logits in both KL directions."""
import argparse,json,pathlib,torch
torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,nargs='+',required=True);p.add_argument('--candidate',type=pathlib.Path,nargs='+',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if len(a.reference)!=len(a.candidate):raise ValueError('Case counts differ')
records=[]
for ref,mod in zip(a.reference,a.candidate):
 x=torch.load(ref,map_location='cpu',weights_only=False);y=torch.load(mod,map_location='cpu',weights_only=False)
 if x['tokens']!=y['tokens']:raise ValueError('Forced continuations differ')
 if x['logits'].shape!=y['logits'].shape:raise ValueError('Logit shapes differ')
 for start in range(0,len(x['tokens']),8):
  u=x['logits'][start:start+8].double();v=y['logits'][start:start+8].double()
  if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):raise ValueError('Nonfinite raw logits')
  lp=torch.log_softmax(u,dim=-1);lq=torch.log_softmax(v,dim=-1)
  forward=(lp.exp()*(lp-lq)).sum(-1);reverse=(lq.exp()*(lq-lp)).sum(-1);delta=(u-v).abs()
  for j in range(len(u)):
   records.append(dict(case=ref.name,token=start+j,top1_equal=bool(u[j].argmax()==v[j].argmax()),kl_reference_candidate=float(forward[j]),kl_candidate_reference=float(reverse[j]),max_abs_logit_error=float(delta[j].max()),rms_logit_error=float(delta[j].square().mean().sqrt())))
summary=dict(tokens=len(records),top1_agreement=sum(r['top1_equal'] for r in records)/len(records),mean_kl_reference_candidate=sum(r['kl_reference_candidate'] for r in records)/len(records),mean_kl_candidate_reference=sum(r['kl_candidate_reference'] for r in records)/len(records),max_kl_reference_candidate=max(r['kl_reference_candidate'] for r in records),max_abs_logit_error=max(r['max_abs_logit_error'] for r in records),scope='Forced identical continuations; compare to original-versus-original variability before interpreting differences.',records=records)
a.output.write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
