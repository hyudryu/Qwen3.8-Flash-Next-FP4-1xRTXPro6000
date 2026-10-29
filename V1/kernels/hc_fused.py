"""Fuse TP1 HC up-projection, BF16 gate rounding and stream mean.

Prototype only: the output follows the original BF16 intermediate semantics.
The model router is unrelated to these residual gates.
"""
import triton
import triton.language as tl
@triton.jit
def hc_up_gate(x,w,norm,out,N:tl.constexpr,H:tl.constexpr,R:tl.constexpr,S:tl.constexpr,BH:tl.constexpr):
 t=tl.program_id(0)*16+tl.arange(0,16)
 h=tl.program_id(1)*BH+tl.arange(0,BH)
 c=(tl.arange(0,S)[:,None]*H+h[None,:]).reshape((S*BH,))
 k=tl.arange(0,32)
 acc=tl.zeros((16,S*BH),tl.float32)
 for base in range(tl.cdiv(R,32)):
  r=base*32+k
  a=tl.load(x+t[:,None]*R+r[None,:],(t[:,None]<N)&(r[None,:]<R),0)
  b=tl.load(w+c[None,:]*R+r[:,None],(c[None,:]<S*H)&(r[:,None]<R),0)
  acc=tl.dot(a,b,acc)
 logits=acc.to(tl.bfloat16).to(tl.float32)
 gate=tl.sigmoid(logits).to(tl.bfloat16)
 v=tl.load(norm+t[:,None]*(S*H)+c[None,:],(t[:,None]<N)&(c[None,:]<S*H),0)
 product=(gate*v).to(tl.bfloat16).to(tl.float32).reshape((16,S,BH))
 # Original accumulates stream order, rather than a tree reduction.
 total=tl.sum(tl.where(tl.arange(0,S)[None,:,None]==0,product,0.),1)
 for stream in tl.static_range(1,S):
  total+=tl.sum(tl.where(tl.arange(0,S)[None,:,None]==stream,product,0.),1)
 tl.store(out+t[:,None]*H+h[None,:],total/S,(t[:,None]<N)&(h[None,:]<H))
