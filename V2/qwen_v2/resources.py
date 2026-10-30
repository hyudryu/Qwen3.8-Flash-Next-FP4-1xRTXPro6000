"""Setup-time compiled resource and driver occupancy inspection."""
import ctypes
import torch

def inspect(kernel,device,threads=128):
    kernel._init_handles()
    result={'registers_per_thread':kernel.n_regs,'shared_memory_bytes':kernel.metadata.shared,'threads_per_cta':threads}
    driver=ctypes.CDLL('libcuda.so.1')
    blocks=ctypes.c_int();maximum_threads=ctypes.c_int()
    occupancy=driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    occupancy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
    occupancy.restype=ctypes.c_int
    status=occupancy(ctypes.byref(blocks),ctypes.c_void_p(kernel.function),threads,kernel.metadata.shared)
    maximum=driver.cuDeviceGetAttribute
    maximum.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    maximum.restype=ctypes.c_int
    attr_status=maximum(ctypes.byref(maximum_threads),39,device.index or 0)
    result['occupancy_query_status']=status
    if status==0:
        result['theoretical_blocks_per_sm']=blocks.value
        result['sm_count']=torch.cuda.get_device_properties(device).multi_processor_count
        if attr_status==0:result['theoretical_thread_occupancy']=blocks.value*threads/maximum_threads.value
    result['scope']='Driver resource bound, not measured achieved occupancy'
    return result
