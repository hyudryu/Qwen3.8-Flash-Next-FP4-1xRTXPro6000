"""CuTe-DSL CSF decoder with early exception metadata loading.

For aligned native NVFP4 planes, overlap sparse exception index/first-record
loads with the base/nibble expansion. Keep only compact bounds and the first
record in registers; patch remaining exceptions cooperatively in shared memory.
The authoritative codec/layout helpers are unchanged. No prediction is used.
"""
import hashlib,pathlib
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32,Int64,Uint32
from b12x._lib.quant.csf_blocks import CompactBlockStoragePlane
from b12x._lib.quant.nvfp4_csf import _Pair
from b12x._lib.intrinsics import st_shared_v4_u32,ld_shared_v4_u32
from b12x._lib.compiler import KernelCompileSpec,compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.utils import make_ptr,current_cuda_stream
SOURCE_SHA=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

class MetadataPrefetchPlane(CompactBlockStoragePlane):
    @cute.jit
    def decode(self,tensors,expert,task,tid):
        if cutlass.const_expr(self.shared_exceptions and self.group_record_bytes!=0 and self.destination==0 and self.storage.rows%128==0 and self.storage.columns%4==0 and self.sections==1):
            self.native_pipeline(tensors,expert,task,tid)
        elif cutlass.const_expr(not self.shared_exceptions and self.group_record_bytes!=0 and self.destination==0 and self.storage.rows%128==0 and self.storage.columns%4==0 and self.sections==1):
            self.native_global_pipeline(tensors,expert,task,tid)
        else:
            self.decode_staged(tensors,expert,task,tid)

    @cute.jit
    def native_global_pipeline(self,tensors,expert,task,tid):
        storage,output=tensors
        warp,lane=tid//Int32(32),tid%Int32(32)
        chunks=self.groups_per_task//self.warps
        starts=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint32)
        ends=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint32)
        positions=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Int32)
        exceptions=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint8)
        starts.fill(Uint32(0));ends.fill(Uint32(0));positions.fill(Int32(0));exceptions.fill(cutlass.Uint8(0))
        value_offset=self.group_value_offset(storage)
        for chunk in cutlass.range_constexpr(chunks):
            group=task*Int32(self.groups_per_task)+Int32(chunk*self.warps)+warp
            if warp<Int32(self.warps) and group<Int32(self.groups):
                first,end=self.group_record_bounds(storage,expert,group)
                starts[chunk],ends[chunk]=first,end
                entry=first+lane.to(Uint32)
                if entry<end:
                    pos,value=self.load_group_record(storage,entry,value_offset)
                    positions[chunk],exceptions[chunk]=pos,value
        expert_output=output+Int64(expert)*Int64(self.rows*self.columns)
        for chunk in cutlass.range_constexpr(chunks):
            group=task*Int32(self.groups_per_task)+Int32(chunk*self.warps)+warp
            if warp<Int32(self.warps) and group<Int32(self.groups):
                v=self.group_words(storage,expert,group,lane)
                self.store_group(expert_output,group,lane,v[0],v[1],v[2],v[3])
        # The patch may target a byte written by another lane of this warp.
        cute.arch.sync_warp()
        for chunk in cutlass.range_constexpr(chunks):
            group=task*Int32(self.groups_per_task)+Int32(chunk*self.warps)+warp
            if warp<Int32(self.warps) and group<Int32(self.groups):
                target=expert_output+self.group_output_offset(group)
                if starts[chunk]+lane.to(Uint32)<ends[chunk]:
                    self.write_group_record(group,target,positions[chunk],exceptions[chunk])
                self.patch_group_records(storage,group,lane,target,starts[chunk]+Uint32(32),ends[chunk])

    @cute.jit
    def native_pipeline(self,tensors,expert,task,tid):
        storage,output=tensors
        smem=cutlass.utils.SmemAllocator()
        shared=smem.allocate_tensor(cutlass.Uint32,cute.make_layout(self.groups_per_task*128),byte_alignment=16)
        warp,lane=tid//Int32(32),tid%Int32(32)
        chunks=self.groups_per_task//self.warps
        starts=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint32)
        ends=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint32)
        positions=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Int32)
        exceptions=cute.make_rmem_tensor(cute.make_layout(chunks),cutlass.Uint8)
        starts.fill(Uint32(0));ends.fill(Uint32(0));positions.fill(Int32(0));exceptions.fill(cutlass.Uint8(0))
        value_offset=self.group_value_offset(storage)
        # Stage sparse metadata first, before independent base/code loads.
        for chunk in cutlass.range_constexpr(chunks):
            group=task*Int32(self.groups_per_task)+Int32(chunk*self.warps)+warp
            if warp<Int32(self.warps) and group<Int32(self.groups):
                first,end=self.group_record_bounds(storage,expert,group)
                starts[chunk],ends[chunk]=first,end
                entry=first+lane.to(Uint32)
                if entry<end:
                    pos,value=self.load_group_record(storage,entry,value_offset)
                    positions[chunk],exceptions[chunk]=pos,value
        for chunk in cutlass.range_constexpr(chunks):
            local=Int32(chunk*self.warps)+warp
            group=task*Int32(self.groups_per_task)+local
            if warp<Int32(self.warps) and group<Int32(self.groups):
                v=self.group_words(storage,expert,group,lane)
                address=shared.iterator.toint().to(Int32)+local*Int32(512)+lane*Int32(16)
                st_shared_v4_u32(address,v[0],v[1],v[2],v[3])
        cute.arch.sync_warp()
        for chunk in cutlass.range_constexpr(chunks):
            local=Int32(chunk*self.warps)+warp
            group=task*Int32(self.groups_per_task)+local
            if warp<Int32(self.warps) and group<Int32(self.groups):
                target=cute.recast_ptr(shared.iterator,dtype=cutlass.Uint8)+local*Int32(512)
                if starts[chunk]+lane.to(Uint32)<ends[chunk]:
                    self.write_group_record(group,target,positions[chunk],exceptions[chunk])
                self.patch_group_records(storage,group,lane,target,starts[chunk]+Uint32(32),ends[chunk])
        cute.arch.sync_warp()
        for chunk in cutlass.range_constexpr(chunks):
            local=Int32(chunk*self.warps)+warp
            group=task*Int32(self.groups_per_task)+local
            if warp<Int32(self.warps) and group<Int32(self.groups):
                address=shared.iterator.toint().to(Int32)+local*Int32(512)+lane*Int32(16)
                a,b,c,d=ld_shared_v4_u32(address)
                self.store_group(output+Int64(expert)*Int64(self.rows*self.columns),group,lane,a,b,c,d)

class MetadataPrefetchPair(_Pair):
    def __init__(self,first,second,indexed=False):
        if indexed:raise ValueError('Indexed layout is not supported by this prototype')
        self.first=MetadataPrefetchPlane(first);self.second=MetadataPrefetchPlane(second)
        self.threads=max(self.first.threads,self.second.threads)

@program_cache
def compile_pipeline(first,second,ids64=False):
    launch=MetadataPrefetchPair(first,second)
    plane=(make_ptr(cutlass.Uint8,16,cute.AddressSpace.gmem,assumed_align=16),
           make_ptr(cutlass.Uint8,16,cute.AddressSpace.gmem,assumed_align=16),
           make_ptr(cutlass.Int64,8,cute.AddressSpace.gmem,assumed_align=8),
           make_ptr(cutlass.Uint8,16,cute.AddressSpace.gmem,assumed_align=16),
           make_ptr(cutlass.Uint8,16,cute.AddressSpace.gmem,assumed_align=16))
    key=(first,second,bool(ids64),SOURCE_SHA)
    return b12x_compile(launch,*plane,*plane,
        make_ptr(cutlass.Int64 if ids64 else cutlass.Int32,8 if ids64 else 4,cute.AddressSpace.gmem,assumed_align=8 if ids64 else 4),
        1,1,1,1,0,make_ptr(cutlass.Int32,4,cute.AddressSpace.gmem,assumed_align=4),make_ptr(cutlass.Int32,4,cute.AddressSpace.gmem,assumed_align=4),0,current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key('qwen38.quant.csf_pipeline',1,key))

def pipeline_geometries(first,second):
    """Validated model geometry, independent of the tuner's small-route layout."""
    if not all(g[2]==4 and g[3]==9 and g[6]==512 and g[9]==0 and g[10]==1
               and g[0]%128==0 and g[1]%4==0 for g in (first,second)):
        return None
    return tuple((*g[:7],32,0,*g[9:]) for g in (first,second))
