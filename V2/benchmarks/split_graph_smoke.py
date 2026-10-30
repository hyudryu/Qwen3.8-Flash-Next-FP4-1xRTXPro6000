#!/usr/bin/env python3
"""Changing-shape graph replay with one shared scratch buffer and CPU reference."""
import json,torch
from qwen_v2.fused_shared import swiglu_projection_split,split_epilogue

def main():
 torch.set_num_threads(4);torch.manual_seed(391);device='cuda'
 w=(torch.randn(2560,640,device=device)*0.1).to(torch.float8_e4m3fn)
 scales=torch.randint(119,130,(2560,20),device=device,dtype=torch.uint8)
 decoded=(w.float()*torch.exp2(scales.float()-127).repeat_interleave(32,1)).bfloat16().float().cpu()
 scratch=torch.empty(8*64*2560,device=device);cases={};records=[]
 for n in [1,2,4,8,16,32,64]:
  gu=torch.randn(n,1280,device=device,dtype=torch.bfloat16);gate=torch.randn(n,1,device=device,dtype=torch.bfloat16)
  out=torch.empty(n,2560,device=device,dtype=torch.bfloat16);act=torch.empty(n,640,device=device,dtype=torch.bfloat16)
  def run():
   swiglu_projection_split[((n+15)//16,40,8)](gu,w,scales,scratch,n,640,2560,16,64,64,8,act,True,num_warps=4,enable_fp_fusion=False)
   split_epilogue[((n*2560+511)//512,)](scratch,gate,out,n,2560,8,512,num_warps=4,enable_fp_fusion=False)
  run();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):run()
  cases[n]=(gu,gate,out,act,graph)
 for n in [1,8,2,64,4,32,8,16,1,4,64,2]:
  gu,gate,out,act,graph=cases[n];gu.normal_();gate.normal_();scratch.fill_(float('nan'));graph.replay()
  g,u=gu[:,:640].float(),gu[:,640:].float();expected_act=((g*torch.sigmoid(g)).bfloat16().float()*u).bfloat16()
  expected_projection=(expected_act.float().cpu()@decoded.T).bfloat16().to(device)
  expected=(expected_projection*torch.sigmoid(gate)).bfloat16();diff=(expected.float()-out.float()).abs()
  assert torch.equal(act,expected_act)
  assert torch.isfinite(out).all()
  records.append(dict(rows=n,activation_exact=True,output_max_abs=float(diff.max()),output_rms=float(diff.square().mean().sqrt())))
 print(json.dumps({'scope':'Synthetic shared-scratch replay and memory-safety smoke, not checkpoint parity or serving performance','cases':records},indent=2))
if __name__=='__main__':main()
