# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_kamei/xe/cxeorct76b6d6xzfwzyttr6zc2nb7x4vg5esuvatvvhtiqagrrsm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid_backward, aten.threshold_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %mul_1), kwargs = {})
triton_poi_fused_sigmoid_backward_threshold_backward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_backward_threshold_backward_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp2 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp0, tmp7, tmp6)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    sigmoid, le, tangents_1 = args
    args.clear()
    assert_size_stride(sigmoid, (100, ), (1, ))
    assert_size_stride(le, (100, ), (1, ))
    assert_size_stride(tangents_1, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((100, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid_backward, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_backward_threshold_backward_0.run(le, tangents_1, sigmoid, buf0, 100, grid=grid(100), stream=stream0)
        del le
        del sigmoid
        del tangents_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    sigmoid = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    le = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([sigmoid, le, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
