#!/usr/bin/env python3
"""Small standalone geometry/rounding/graph reuse test; no model is loaded."""
import argparse,json,torch
from qwen_v2.fused_shared import swiglu_projection

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cpu-reference',action='store_true');args=parser.parse_args()
 torch.set_num_threads(4)
 torch.manual_seed(231);device='cuda';results=[]
 w=(torch.randn(2560,640,device=device)*0.1).to(torch.float8_e4m3fn)
 scales=torch.randint(119,130,(2560,20),device=device,dtype=torch.uint8)
 decoded=(w.float()*torch.exp2(scales.float()-127).repeat_interleave(32,1)).bfloat16()
 for n in [1,2,4,8,16,32,64]:
  gu=torch.randn(n,1280,device=device,dtype=torch.bfloat16);gate=torch.randn(n,1,device=device,dtype=torch.bfloat16)
  out=torch.empty(n,2560,device=device,dtype=torch.bfloat16);act=torch.empty(n,640,device=device,dtype=torch.bfloat16)
  def run():swiglu_projection[((n+15)//16,160)](gu,w,scales,gate,out,act,n,640,2560,16,16,64,True,True,num_warps=4,enable_fp_fusion=False)
  run();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):run()
  for rep in range(3):
   gu.normal_();gate.normal_();graph.replay()
   g,u=gu[:,:640].float(),gu[:,640:].float();refact=((g*torch.sigmoid(g)).bfloat16().float()*u).bfloat16()
   if args.cpu_reference:
    projection=(refact.float().cpu()@decoded.float().cpu().T).bfloat16().to(device)
   else:projection=refact@decoded.T
   reference=(projection*torch.sigmoid(gate)).bfloat16()
   assert torch.isfinite(out).all()
   ad=(act.float()-refact.float()).abs();diff=(out.float()-reference.float()).abs()
   results.append(dict(rows=n,replay=rep,activation_max_abs=float(ad.max()),output_max_abs=float(diff.max()),output_rms=float(diff.square().mean().sqrt())))
 print(json.dumps({'scope':'Synthetic standalone smoke, not checkpoint parity or serving speed','cases':results},indent=2))
if __name__=='__main__':main()
