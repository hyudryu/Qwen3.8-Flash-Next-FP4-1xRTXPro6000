"""GPU compaction of authoritative routed expert IDs; no host counts."""
import triton
import triton.language as tl
@triton.jit
def compact_experts(ids,compact,N:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
 r=tl.arange(0,B);raw=tl.load(ids+r,r<N,other=-1)
 # Validate in the original ID width: an invalid int64 ID must not wrap to
 # a valid expert when narrowed for the histogram.
 valid=(r<N)&(raw>=0)&(raw<E)
 counts=tl.histogram(tl.where(valid,raw,0).to(tl.int32),E)
 e=tl.arange(0,E)
 counts-=tl.where(e==0,tl.sum((~valid).to(tl.int32)),0)
 present=counts>0;rank=tl.cumsum(present.to(tl.int32));total=tl.sum(present.to(tl.int32))
 capacity=tl.minimum(N,E)
 tl.store(compact+e,-1,(e>=total)&(e<capacity))
 tl.store(compact+rank-1,e,present)
