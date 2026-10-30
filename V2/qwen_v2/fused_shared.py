"""Qwen shared-FFN activation/projection/epilogue experiments.

Each eligible path retains the prepared activation precision and rounds the
SwiGLU result, projection output and gate/product at the original BF16 edges.
Unsupported prepared regimes continue through the original module.
"""
import torch,triton
import triton.language as tl

@triton.jit
def swiglu_projection(GU,W,WS,GATE,OUT,ACT,N:tl.constexpr,K:tl.constexpr,O:tl.constexpr,
                      BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GATED:tl.constexpr,STORE_ACT:tl.constexpr):
 rows=tl.program_id(0)*BM+tl.arange(0,BM)
 cols=tl.program_id(1)*BN+tl.arange(0,BN)
 kk=tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.float32)
 for base in range(tl.cdiv(K,BK)):
  k=base*BK+kk
  gate=tl.load(GU+rows[:,None]*(2*K)+k[None,:],(rows[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
  up=tl.load(GU+rows[:,None]*(2*K)+K+k[None,:],(rows[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
  x=((gate*tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)*up).to(tl.bfloat16)
  if STORE_ACT and tl.program_id(1)==0:tl.store(ACT+rows[:,None]*K+k[None,:],x,(rows[:,None]<N)&(k[None,:]<K))
  weight=tl.load(W+cols[:,None]*K+k[None,:],(cols[:,None]<O)&(k[None,:]<K),0.).to(tl.float32)
  scale_byte=tl.load(WS+cols[:,None]*(K//32)+k[None,:]//32,(cols[:,None]<O)&(k[None,:]<K),127).to(tl.int32)
  scale=(scale_byte<<23).to(tl.float32,bitcast=True)
  b=(weight*scale).to(tl.bfloat16)
  acc=tl.dot(x,tl.trans(b),acc)
 result=acc.to(tl.bfloat16)
 if GATED:
  gate_logits=tl.load(GATE+rows,rows<N,0).to(tl.float32)
  gate_factor=tl.sigmoid(gate_logits).to(tl.bfloat16)
  result=(result.to(tl.float32)*gate_factor[:,None].to(tl.float32)).to(tl.bfloat16)
 tl.store(OUT+rows[:,None]*O+cols[None,:],result,(rows[:,None]<N)&(cols[None,:]<O))

@triton.jit
def swiglu_projection_split(GU,W,WS,PART,N:tl.constexpr,K:tl.constexpr,O:tl.constexpr,
                            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLITS:tl.constexpr,ACT,STORE_ACT:tl.constexpr):
 rows=tl.program_id(0)*BM+tl.arange(0,BM)
 cols=tl.program_id(1)*BN+tl.arange(0,BN)
 split=tl.program_id(2)
 kk=tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.float32)
 blocks=tl.cdiv(tl.cdiv(K,BK),SPLITS)
 for local in range(blocks):
  k=(split*blocks+local)*BK+kk
  mask=(rows[:,None]<N)&(k[None,:]<K)
  gate=tl.load(GU+rows[:,None]*(2*K)+k[None,:],mask,0).to(tl.float32)
  up=tl.load(GU+rows[:,None]*(2*K)+K+k[None,:],mask,0).to(tl.float32)
  x=((gate*tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)*up).to(tl.bfloat16)
  if STORE_ACT and tl.program_id(1)==0:
   tl.store(ACT+rows[:,None]*K+k[None,:],x,mask)
  weight=tl.load(W+cols[:,None]*K+k[None,:],(cols[:,None]<O)&(k[None,:]<K),0.).to(tl.float32)
  sb=tl.load(WS+cols[:,None]*(K//32)+k[None,:]//32,(cols[:,None]<O)&(k[None,:]<K),127).to(tl.int32)
  scale=(sb<<23).to(tl.float32,bitcast=True)
  b=(weight*scale).to(tl.bfloat16)
  acc=tl.dot(x,tl.trans(b),acc)
 tl.store(PART+split*N*O+rows[:,None]*O+cols[None,:],acc,(rows[:,None]<N)&(cols[None,:]<O))

@triton.jit
def split_epilogue(PART,GATE,OUT,N:tl.constexpr,O:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 acc=tl.full((B,),0,tl.float32)
 for split in range(SPLITS):
  acc+=tl.load(PART+split*N*O+i,i<N*O,0)
 projection=acc.to(tl.bfloat16).to(tl.float32)
 gate=tl.sigmoid(tl.load(GATE+i//O,i<N*O,0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
 tl.store(OUT+i,projection*gate,i<N*O)
