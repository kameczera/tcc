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


# kernel path: /tmp/torchinductor_kamei/wc/cwclii2semmnsj4lzqkmmtcngyzxlpbfz2iimkosbqoqkz5k2kri.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_1 : [num_users=99] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%tangents_1, %full_default_1, 0, 99), kwargs = {})
#   %select_scatter_default_2 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %full_default_1, 0, 98), kwargs = {})
#   %select_scatter_default_4 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %full_default_1, 0, 97), kwargs = {})
#   %select_scatter_default_6 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %full_default_1, 0, 96), kwargs = {})
#   %select_scatter_default_8 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_6, %full_default_1, 0, 95), kwargs = {})
#   %select_scatter_default_10 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %full_default_1, 0, 94), kwargs = {})
#   %select_scatter_default_12 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_10, %full_default_1, 0, 93), kwargs = {})
#   %select_scatter_default_14 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %full_default_1, 0, 92), kwargs = {})
#   %select_scatter_default_16 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_14, %full_default_1, 0, 91), kwargs = {})
#   %select_scatter_default_18 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_16, %full_default_1, 0, 90), kwargs = {})
#   %select_scatter_default_20 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_18, %full_default_1, 0, 89), kwargs = {})
#   %select_scatter_default_22 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_20, %full_default_1, 0, 88), kwargs = {})
#   %select_scatter_default_24 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_22, %full_default_1, 0, 87), kwargs = {})
#   %select_scatter_default_26 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_24, %full_default_1, 0, 86), kwargs = {})
#   %select_scatter_default_28 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_26, %full_default_1, 0, 85), kwargs = {})
#   %select_scatter_default_30 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_28, %full_default_1, 0, 84), kwargs = {})
#   %select_scatter_default_32 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_30, %full_default_1, 0, 83), kwargs = {})
#   %select_scatter_default_34 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_32, %full_default_1, 0, 82), kwargs = {})
#   %select_scatter_default_36 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_34, %full_default_1, 0, 81), kwargs = {})
#   %select_scatter_default_38 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_36, %full_default_1, 0, 80), kwargs = {})
#   %select_scatter_default_40 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_38, %full_default_1, 0, 79), kwargs = {})
#   %select_scatter_default_42 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_40, %full_default_1, 0, 78), kwargs = {})
#   %select_scatter_default_44 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_42, %full_default_1, 0, 77), kwargs = {})
#   %select_scatter_default_46 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_44, %full_default_1, 0, 76), kwargs = {})
#   %select_scatter_default_48 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_46, %full_default_1, 0, 75), kwargs = {})
#   %select_scatter_default_50 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_48, %full_default_1, 0, 74), kwargs = {})
#   %select_scatter_default_52 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_50, %full_default_1, 0, 73), kwargs = {})
#   %select_scatter_default_54 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_52, %full_default_1, 0, 72), kwargs = {})
#   %select_scatter_default_56 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_54, %full_default_1, 0, 71), kwargs = {})
#   %select_scatter_default_58 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_56, %full_default_1, 0, 70), kwargs = {})
#   %select_scatter_default_60 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_58, %full_default_1, 0, 69), kwargs = {})
#   %select_scatter_default_62 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_60, %full_default_1, 0, 68), kwargs = {})
#   %select_scatter_default_64 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_62, %full_default_1, 0, 67), kwargs = {})
#   %select_scatter_default_66 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_64, %full_default_1, 0, 66), kwargs = {})
#   %select_scatter_default_68 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_66, %full_default_1, 0, 65), kwargs = {})
#   %select_scatter_default_70 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_68, %full_default_1, 0, 64), kwargs = {})
#   %select_scatter_default_72 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_70, %full_default_1, 0, 63), kwargs = {})
#   %select_scatter_default_74 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_72, %full_default_1, 0, 62), kwargs = {})
#   %select_scatter_default_76 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_74, %full_default_1, 0, 61), kwargs = {})
#   %select_scatter_default_78 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_76, %full_default_1, 0, 60), kwargs = {})
#   %select_scatter_default_80 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_78, %full_default_1, 0, 59), kwargs = {})
#   %select_scatter_default_82 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_80, %full_default_1, 0, 58), kwargs = {})
#   %select_scatter_default_84 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_82, %full_default_1, 0, 57), kwargs = {})
#   %select_scatter_default_86 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_84, %full_default_1, 0, 56), kwargs = {})
#   %select_scatter_default_88 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_86, %full_default_1, 0, 55), kwargs = {})
#   %select_scatter_default_90 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_88, %full_default_1, 0, 54), kwargs = {})
#   %select_scatter_default_92 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_90, %full_default_1, 0, 53), kwargs = {})
#   %select_scatter_default_94 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_92, %full_default_1, 0, 52), kwargs = {})
#   %select_scatter_default_96 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_94, %full_default_1, 0, 51), kwargs = {})
#   %select_scatter_default_98 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_96, %full_default_1, 0, 50), kwargs = {})
#   %select_scatter_default_100 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_98, %full_default_1, 0, 49), kwargs = {})
#   %select_scatter_default_102 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_100, %full_default_1, 0, 48), kwargs = {})
#   %select_scatter_default_104 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_102, %full_default_1, 0, 47), kwargs = {})
#   %select_scatter_default_106 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_104, %full_default_1, 0, 46), kwargs = {})
#   %select_scatter_default_108 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_106, %full_default_1, 0, 45), kwargs = {})
#   %select_scatter_default_110 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_108, %full_default_1, 0, 44), kwargs = {})
#   %select_scatter_default_112 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_110, %full_default_1, 0, 43), kwargs = {})
#   %select_scatter_default_114 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_112, %full_default_1, 0, 42), kwargs = {})
#   %select_scatter_default_116 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_114, %full_default_1, 0, 41), kwargs = {})
#   %select_scatter_default_118 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_116, %full_default_1, 0, 40), kwargs = {})
#   %select_scatter_default_120 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_118, %full_default_1, 0, 39), kwargs = {})
#   %select_scatter_default_122 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_120, %full_default_1, 0, 38), kwargs = {})
#   %select_scatter_default_124 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_122, %full_default_1, 0, 37), kwargs = {})
#   %select_scatter_default_126 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_124, %full_default_1, 0, 36), kwargs = {})
#   %select_scatter_default_128 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_126, %full_default_1, 0, 35), kwargs = {})
#   %select_scatter_default_130 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_128, %full_default_1, 0, 34), kwargs = {})
#   %select_scatter_default_132 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_130, %full_default_1, 0, 33), kwargs = {})
#   %select_scatter_default_134 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_132, %full_default_1, 0, 32), kwargs = {})
#   %select_scatter_default_136 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_134, %full_default_1, 0, 31), kwargs = {})
#   %select_scatter_default_138 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_136, %full_default_1, 0, 30), kwargs = {})
#   %select_scatter_default_140 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_138, %full_default_1, 0, 29), kwargs = {})
#   %select_scatter_default_142 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_140, %full_default_1, 0, 28), kwargs = {})
#   %select_scatter_default_144 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_142, %full_default_1, 0, 27), kwargs = {})
#   %select_scatter_default_146 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_144, %full_default_1, 0, 26), kwargs = {})
#   %select_scatter_default_148 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_146, %full_default_1, 0, 25), kwargs = {})
#   %select_scatter_default_150 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_148, %full_default_1, 0, 24), kwargs = {})
#   %select_scatter_default_152 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_150, %full_default_1, 0, 23), kwargs = {})
#   %select_scatter_default_154 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_152, %full_default_1, 0, 22), kwargs = {})
#   %select_scatter_default_156 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_154, %full_default_1, 0, 21), kwargs = {})
#   %select_scatter_default_158 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_156, %full_default_1, 0, 20), kwargs = {})
#   %select_scatter_default_160 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_158, %full_default_1, 0, 19), kwargs = {})
#   %select_scatter_default_162 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_160, %full_default_1, 0, 18), kwargs = {})
#   %select_scatter_default_164 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_162, %full_default_1, 0, 17), kwargs = {})
#   %select_scatter_default_166 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_164, %full_default_1, 0, 16), kwargs = {})
#   %select_scatter_default_168 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_166, %full_default_1, 0, 15), kwargs = {})
#   %select_scatter_default_170 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_168, %full_default_1, 0, 14), kwargs = {})
#   %select_scatter_default_172 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_170, %full_default_1, 0, 13), kwargs = {})
#   %select_scatter_default_174 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_172, %full_default_1, 0, 12), kwargs = {})
#   %select_scatter_default_176 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_174, %full_default_1, 0, 11), kwargs = {})
#   %select_scatter_default_178 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_176, %full_default_1, 0, 10), kwargs = {})
triton_poi_fused_zeros_like_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp21 = tl.load(in_ptr0 + (x0), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 90, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 91, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 92, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 93, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 94, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 95, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 96, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 97, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 98, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp19 = tl.full([1], 99, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp22 = 0.0
    tmp23 = tl.where(tmp20, tmp22, tmp21)
    tmp24 = tl.where(tmp18, tmp22, tmp23)
    tmp25 = tl.where(tmp16, tmp22, tmp24)
    tmp26 = tl.where(tmp14, tmp22, tmp25)
    tmp27 = tl.where(tmp12, tmp22, tmp26)
    tmp28 = tl.where(tmp10, tmp22, tmp27)
    tmp29 = tl.where(tmp8, tmp22, tmp28)
    tmp30 = tl.where(tmp6, tmp22, tmp29)
    tmp31 = tl.where(tmp4, tmp22, tmp30)
    tmp32 = tl.where(tmp2, tmp22, tmp31)
    tmp33 = tl.full([1], 80, tl.int32)
    tmp34 = tmp0 == tmp33
    tmp35 = tl.full([1], 81, tl.int32)
    tmp36 = tmp0 == tmp35
    tmp37 = tl.full([1], 82, tl.int32)
    tmp38 = tmp0 == tmp37
    tmp39 = tl.full([1], 83, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp41 = tl.full([1], 84, tl.int32)
    tmp42 = tmp0 == tmp41
    tmp43 = tl.full([1], 85, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp45 = tl.full([1], 86, tl.int32)
    tmp46 = tmp0 == tmp45
    tmp47 = tl.full([1], 87, tl.int32)
    tmp48 = tmp0 == tmp47
    tmp49 = tl.full([1], 88, tl.int32)
    tmp50 = tmp0 == tmp49
    tmp51 = tl.full([1], 89, tl.int32)
    tmp52 = tmp0 == tmp51
    tmp53 = tl.where(tmp52, tmp22, tmp32)
    tmp54 = tl.where(tmp50, tmp22, tmp53)
    tmp55 = tl.where(tmp48, tmp22, tmp54)
    tmp56 = tl.where(tmp46, tmp22, tmp55)
    tmp57 = tl.where(tmp44, tmp22, tmp56)
    tmp58 = tl.where(tmp42, tmp22, tmp57)
    tmp59 = tl.where(tmp40, tmp22, tmp58)
    tmp60 = tl.where(tmp38, tmp22, tmp59)
    tmp61 = tl.where(tmp36, tmp22, tmp60)
    tmp62 = tl.where(tmp34, tmp22, tmp61)
    tmp63 = tl.full([1], 70, tl.int32)
    tmp64 = tmp0 == tmp63
    tmp65 = tl.full([1], 71, tl.int32)
    tmp66 = tmp0 == tmp65
    tmp67 = tl.full([1], 72, tl.int32)
    tmp68 = tmp0 == tmp67
    tmp69 = tl.full([1], 73, tl.int32)
    tmp70 = tmp0 == tmp69
    tmp71 = tl.full([1], 74, tl.int32)
    tmp72 = tmp0 == tmp71
    tmp73 = tl.full([1], 75, tl.int32)
    tmp74 = tmp0 == tmp73
    tmp75 = tl.full([1], 76, tl.int32)
    tmp76 = tmp0 == tmp75
    tmp77 = tl.full([1], 77, tl.int32)
    tmp78 = tmp0 == tmp77
    tmp79 = tl.full([1], 78, tl.int32)
    tmp80 = tmp0 == tmp79
    tmp81 = tl.full([1], 79, tl.int32)
    tmp82 = tmp0 == tmp81
    tmp83 = tl.where(tmp82, tmp22, tmp62)
    tmp84 = tl.where(tmp80, tmp22, tmp83)
    tmp85 = tl.where(tmp78, tmp22, tmp84)
    tmp86 = tl.where(tmp76, tmp22, tmp85)
    tmp87 = tl.where(tmp74, tmp22, tmp86)
    tmp88 = tl.where(tmp72, tmp22, tmp87)
    tmp89 = tl.where(tmp70, tmp22, tmp88)
    tmp90 = tl.where(tmp68, tmp22, tmp89)
    tmp91 = tl.where(tmp66, tmp22, tmp90)
    tmp92 = tl.where(tmp64, tmp22, tmp91)
    tmp93 = tl.full([1], 60, tl.int32)
    tmp94 = tmp0 == tmp93
    tmp95 = tl.full([1], 61, tl.int32)
    tmp96 = tmp0 == tmp95
    tmp97 = tl.full([1], 62, tl.int32)
    tmp98 = tmp0 == tmp97
    tmp99 = tl.full([1], 63, tl.int32)
    tmp100 = tmp0 == tmp99
    tmp101 = tl.full([1], 64, tl.int32)
    tmp102 = tmp0 == tmp101
    tmp103 = tl.full([1], 65, tl.int32)
    tmp104 = tmp0 == tmp103
    tmp105 = tl.full([1], 66, tl.int32)
    tmp106 = tmp0 == tmp105
    tmp107 = tl.full([1], 67, tl.int32)
    tmp108 = tmp0 == tmp107
    tmp109 = tl.full([1], 68, tl.int32)
    tmp110 = tmp0 == tmp109
    tmp111 = tl.full([1], 69, tl.int32)
    tmp112 = tmp0 == tmp111
    tmp113 = tl.where(tmp112, tmp22, tmp92)
    tmp114 = tl.where(tmp110, tmp22, tmp113)
    tmp115 = tl.where(tmp108, tmp22, tmp114)
    tmp116 = tl.where(tmp106, tmp22, tmp115)
    tmp117 = tl.where(tmp104, tmp22, tmp116)
    tmp118 = tl.where(tmp102, tmp22, tmp117)
    tmp119 = tl.where(tmp100, tmp22, tmp118)
    tmp120 = tl.where(tmp98, tmp22, tmp119)
    tmp121 = tl.where(tmp96, tmp22, tmp120)
    tmp122 = tl.where(tmp94, tmp22, tmp121)
    tmp123 = tl.full([1], 50, tl.int32)
    tmp124 = tmp0 == tmp123
    tmp125 = tl.full([1], 51, tl.int32)
    tmp126 = tmp0 == tmp125
    tmp127 = tl.full([1], 52, tl.int32)
    tmp128 = tmp0 == tmp127
    tmp129 = tl.full([1], 53, tl.int32)
    tmp130 = tmp0 == tmp129
    tmp131 = tl.full([1], 54, tl.int32)
    tmp132 = tmp0 == tmp131
    tmp133 = tl.full([1], 55, tl.int32)
    tmp134 = tmp0 == tmp133
    tmp135 = tl.full([1], 56, tl.int32)
    tmp136 = tmp0 == tmp135
    tmp137 = tl.full([1], 57, tl.int32)
    tmp138 = tmp0 == tmp137
    tmp139 = tl.full([1], 58, tl.int32)
    tmp140 = tmp0 == tmp139
    tmp141 = tl.full([1], 59, tl.int32)
    tmp142 = tmp0 == tmp141
    tmp143 = tl.where(tmp142, tmp22, tmp122)
    tmp144 = tl.where(tmp140, tmp22, tmp143)
    tmp145 = tl.where(tmp138, tmp22, tmp144)
    tmp146 = tl.where(tmp136, tmp22, tmp145)
    tmp147 = tl.where(tmp134, tmp22, tmp146)
    tmp148 = tl.where(tmp132, tmp22, tmp147)
    tmp149 = tl.where(tmp130, tmp22, tmp148)
    tmp150 = tl.where(tmp128, tmp22, tmp149)
    tmp151 = tl.where(tmp126, tmp22, tmp150)
    tmp152 = tl.where(tmp124, tmp22, tmp151)
    tmp153 = tl.full([1], 40, tl.int32)
    tmp154 = tmp0 == tmp153
    tmp155 = tl.full([1], 41, tl.int32)
    tmp156 = tmp0 == tmp155
    tmp157 = tl.full([1], 42, tl.int32)
    tmp158 = tmp0 == tmp157
    tmp159 = tl.full([1], 43, tl.int32)
    tmp160 = tmp0 == tmp159
    tmp161 = tl.full([1], 44, tl.int32)
    tmp162 = tmp0 == tmp161
    tmp163 = tl.full([1], 45, tl.int32)
    tmp164 = tmp0 == tmp163
    tmp165 = tl.full([1], 46, tl.int32)
    tmp166 = tmp0 == tmp165
    tmp167 = tl.full([1], 47, tl.int32)
    tmp168 = tmp0 == tmp167
    tmp169 = tl.full([1], 48, tl.int32)
    tmp170 = tmp0 == tmp169
    tmp171 = tl.full([1], 49, tl.int32)
    tmp172 = tmp0 == tmp171
    tmp173 = tl.where(tmp172, tmp22, tmp152)
    tmp174 = tl.where(tmp170, tmp22, tmp173)
    tmp175 = tl.where(tmp168, tmp22, tmp174)
    tmp176 = tl.where(tmp166, tmp22, tmp175)
    tmp177 = tl.where(tmp164, tmp22, tmp176)
    tmp178 = tl.where(tmp162, tmp22, tmp177)
    tmp179 = tl.where(tmp160, tmp22, tmp178)
    tmp180 = tl.where(tmp158, tmp22, tmp179)
    tmp181 = tl.where(tmp156, tmp22, tmp180)
    tmp182 = tl.where(tmp154, tmp22, tmp181)
    tmp183 = tl.full([1], 30, tl.int32)
    tmp184 = tmp0 == tmp183
    tmp185 = tl.full([1], 31, tl.int32)
    tmp186 = tmp0 == tmp185
    tmp187 = tl.full([1], 32, tl.int32)
    tmp188 = tmp0 == tmp187
    tmp189 = tl.full([1], 33, tl.int32)
    tmp190 = tmp0 == tmp189
    tmp191 = tl.full([1], 34, tl.int32)
    tmp192 = tmp0 == tmp191
    tmp193 = tl.full([1], 35, tl.int32)
    tmp194 = tmp0 == tmp193
    tmp195 = tl.full([1], 36, tl.int32)
    tmp196 = tmp0 == tmp195
    tmp197 = tl.full([1], 37, tl.int32)
    tmp198 = tmp0 == tmp197
    tmp199 = tl.full([1], 38, tl.int32)
    tmp200 = tmp0 == tmp199
    tmp201 = tl.full([1], 39, tl.int32)
    tmp202 = tmp0 == tmp201
    tmp203 = tl.where(tmp202, tmp22, tmp182)
    tmp204 = tl.where(tmp200, tmp22, tmp203)
    tmp205 = tl.where(tmp198, tmp22, tmp204)
    tmp206 = tl.where(tmp196, tmp22, tmp205)
    tmp207 = tl.where(tmp194, tmp22, tmp206)
    tmp208 = tl.where(tmp192, tmp22, tmp207)
    tmp209 = tl.where(tmp190, tmp22, tmp208)
    tmp210 = tl.where(tmp188, tmp22, tmp209)
    tmp211 = tl.where(tmp186, tmp22, tmp210)
    tmp212 = tl.where(tmp184, tmp22, tmp211)
    tmp213 = tl.full([1], 20, tl.int32)
    tmp214 = tmp0 == tmp213
    tmp215 = tl.full([1], 21, tl.int32)
    tmp216 = tmp0 == tmp215
    tmp217 = tl.full([1], 22, tl.int32)
    tmp218 = tmp0 == tmp217
    tmp219 = tl.full([1], 23, tl.int32)
    tmp220 = tmp0 == tmp219
    tmp221 = tl.full([1], 24, tl.int32)
    tmp222 = tmp0 == tmp221
    tmp223 = tl.full([1], 25, tl.int32)
    tmp224 = tmp0 == tmp223
    tmp225 = tl.full([1], 26, tl.int32)
    tmp226 = tmp0 == tmp225
    tmp227 = tl.full([1], 27, tl.int32)
    tmp228 = tmp0 == tmp227
    tmp229 = tl.full([1], 28, tl.int32)
    tmp230 = tmp0 == tmp229
    tmp231 = tl.full([1], 29, tl.int32)
    tmp232 = tmp0 == tmp231
    tmp233 = tl.where(tmp232, tmp22, tmp212)
    tmp234 = tl.where(tmp230, tmp22, tmp233)
    tmp235 = tl.where(tmp228, tmp22, tmp234)
    tmp236 = tl.where(tmp226, tmp22, tmp235)
    tmp237 = tl.where(tmp224, tmp22, tmp236)
    tmp238 = tl.where(tmp222, tmp22, tmp237)
    tmp239 = tl.where(tmp220, tmp22, tmp238)
    tmp240 = tl.where(tmp218, tmp22, tmp239)
    tmp241 = tl.where(tmp216, tmp22, tmp240)
    tmp242 = tl.where(tmp214, tmp22, tmp241)
    tmp243 = tl.full([1], 10, tl.int32)
    tmp244 = tmp0 == tmp243
    tmp245 = tl.full([1], 11, tl.int32)
    tmp246 = tmp0 == tmp245
    tmp247 = tl.full([1], 12, tl.int32)
    tmp248 = tmp0 == tmp247
    tmp249 = tl.full([1], 13, tl.int32)
    tmp250 = tmp0 == tmp249
    tmp251 = tl.full([1], 14, tl.int32)
    tmp252 = tmp0 == tmp251
    tmp253 = tl.full([1], 15, tl.int32)
    tmp254 = tmp0 == tmp253
    tmp255 = tl.full([1], 16, tl.int32)
    tmp256 = tmp0 == tmp255
    tmp257 = tl.full([1], 17, tl.int32)
    tmp258 = tmp0 == tmp257
    tmp259 = tl.full([1], 18, tl.int32)
    tmp260 = tmp0 == tmp259
    tmp261 = tl.full([1], 19, tl.int32)
    tmp262 = tmp0 == tmp261
    tmp263 = tl.where(tmp262, tmp22, tmp242)
    tmp264 = tl.where(tmp260, tmp22, tmp263)
    tmp265 = tl.where(tmp258, tmp22, tmp264)
    tmp266 = tl.where(tmp256, tmp22, tmp265)
    tmp267 = tl.where(tmp254, tmp22, tmp266)
    tmp268 = tl.where(tmp252, tmp22, tmp267)
    tmp269 = tl.where(tmp250, tmp22, tmp268)
    tmp270 = tl.where(tmp248, tmp22, tmp269)
    tmp271 = tl.where(tmp246, tmp22, tmp270)
    tmp272 = tl.where(tmp244, tmp22, tmp271)
    tl.store(out_ptr0 + (x0), tmp32, xmask)
    tl.store(out_ptr1 + (x0), tmp62, xmask)
    tl.store(out_ptr2 + (x0), tmp92, xmask)
    tl.store(out_ptr3 + (x0), tmp122, xmask)
    tl.store(out_ptr4 + (x0), tmp152, xmask)
    tl.store(out_ptr5 + (x0), tmp182, xmask)
    tl.store(out_ptr6 + (x0), tmp212, xmask)
    tl.store(out_ptr7 + (x0), tmp242, xmask)
    tl.store(out_ptr8 + (x0), tmp272, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/nl/cnlpp6c2fkz3tn7nmzopux4z7dvqrc4xwvcwdd2pm4hchuo47hxs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_557, 2), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (80))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 80, tl.int32)
    tmp1 = tl.full([1], 81, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 82, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 83, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 84, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 85, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 86, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 87, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 88, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 89, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/jx/cjxy5vayhoxexcnu4dosxovgjnyutt6wp6dixdqdmtv3ehbs26dy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_587, 2), kwargs = {})
triton_poi_fused_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (70))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 70, tl.int32)
    tmp1 = tl.full([1], 71, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 72, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 73, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 74, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 75, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 76, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 77, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 78, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 79, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/g6/cg6yuameepyued4iojoil2d5jwwf4qihfwd246jmadjfpkyorqck.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_617, 2), kwargs = {})
triton_poi_fused_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (60))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 60, tl.int32)
    tmp1 = tl.full([1], 61, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 62, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 63, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 64, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 65, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 66, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 67, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 68, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 69, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/wm/cwmyjrlegb7fcnw4ptlq4uzgbnulb7qarpvvbwzsex2kkcwxqc5p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_647, 2), kwargs = {})
triton_poi_fused_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (50))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 50, tl.int32)
    tmp1 = tl.full([1], 51, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 52, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 53, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 54, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 55, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 56, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 57, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 58, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 59, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/u2/cu2jmryb5fpz26zlyhovhgtrhjq2dkpu6crrlwwn7okfuit26ufc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_677, 2), kwargs = {})
triton_poi_fused_mul_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (40))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 40, tl.int32)
    tmp1 = tl.full([1], 41, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 42, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 43, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 44, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 45, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 46, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 47, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 48, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 49, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/hv/chvvqae2g4orayfosihs7q5pikuep5x2vlwqj7xt3vt6mwpmpujn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_707, 2), kwargs = {})
triton_poi_fused_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (30))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 30, tl.int32)
    tmp1 = tl.full([1], 31, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 32, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 33, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 34, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 35, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 36, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 37, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 38, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 39, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/xi/cxia7wa35pxqe3ipk2hj534hgtishlkgzalqitciukvpggljjgvb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_527, 2), kwargs = {})
triton_poi_fused_mul_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (90))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 90, tl.int32)
    tmp1 = tl.full([1], 91, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 92, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 93, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 94, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 95, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 96, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 97, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 98, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 99, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/vg/cvgvkhcyqlmwzbuqzzdzwwxy2h7tbfdzmwlmopd7e6bo7akspmpr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_737, 2), kwargs = {})
triton_poi_fused_mul_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (20))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 20, tl.int32)
    tmp1 = tl.full([1], 21, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 22, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 23, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 24, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 25, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 26, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 27, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 28, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 29, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/jq/cjq6dfh4f4wanftgwcsbz6zmqqp6vdgbdtp7jf25br3yl2wcipbu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_189 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_767, 2), kwargs = {})
triton_poi_fused_mul_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (10))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 10, tl.int32)
    tmp1 = tl.full([1], 11, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 12, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 13, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 14, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 15, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 16, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 17, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 18, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 19, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/ej/cejffxgqcst4zrexekywjfhkhipgxskje3ecdfd44wrabdmipld3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_797, 2), kwargs = {})
triton_poi_fused_mul_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp19 = tl.load(in_ptr0 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 2, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 3, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tl.full([1], 4, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp9 = tl.full([1], 5, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tl.full([1], 6, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp13 = tl.full([1], 7, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp15 = tl.full([1], 8, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp17 = tl.full([1], 9, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp21, tmp20)
    tmp23 = tl.where(tmp16, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp21, tmp23)
    tmp25 = tl.where(tmp12, tmp21, tmp24)
    tmp26 = tl.where(tmp10, tmp21, tmp25)
    tmp27 = tl.where(tmp8, tmp21, tmp26)
    tmp28 = tl.where(tmp6, tmp21, tmp27)
    tmp29 = tl.where(tmp4, tmp21, tmp28)
    tmp30 = tl.where(tmp2, tmp21, tmp29)
    tmp31 = 2.0
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/a4/ca44iqxhlgpim54bulenqq5ndvtpo3nvpbgqy52yrlwc6wrd3rmv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_500, 2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_100, 0, 99), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_503, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_101, 0, 98), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_1, %select_scatter_default_3), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_506, 2), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_102, 0, 97), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %select_scatter_default_5), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_509, 2), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_103, 0, 96), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %select_scatter_default_7), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_512, 2), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_104, 0, 95), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %select_scatter_default_9), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_515, 2), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_105, 0, 94), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %select_scatter_default_11), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_518, 2), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_106, 0, 93), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %select_scatter_default_13), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_521, 2), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_107, 0, 92), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %select_scatter_default_15), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_524, 2), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_108, 0, 91), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %select_scatter_default_17), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_109, 0, 90), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %select_scatter_default_19), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_530, 2), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_110, 0, 89), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %select_scatter_default_21), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_533, 2), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_111, 0, 88), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %select_scatter_default_23), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_536, 2), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_112, 0, 87), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %select_scatter_default_25), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_539, 2), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_113, 0, 86), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %select_scatter_default_27), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_542, 2), kwargs = {})
#   %select_scatter_default_29 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_114, 0, 85), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %select_scatter_default_29), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_545, 2), kwargs = {})
#   %select_scatter_default_31 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_115, 0, 84), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %select_scatter_default_31), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_548, 2), kwargs = {})
#   %select_scatter_default_33 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_116, 0, 83), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %select_scatter_default_33), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_551, 2), kwargs = {})
#   %select_scatter_default_35 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_117, 0, 82), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %select_scatter_default_35), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_554, 2), kwargs = {})
#   %select_scatter_default_37 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_118, 0, 81), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %select_scatter_default_37), kwargs = {})
#   %select_scatter_default_39 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_119, 0, 80), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %select_scatter_default_39), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_560, 2), kwargs = {})
#   %select_scatter_default_41 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_120, 0, 79), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %select_scatter_default_41), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_563, 2), kwargs = {})
#   %select_scatter_default_43 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_121, 0, 78), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %select_scatter_default_43), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_566, 2), kwargs = {})
#   %select_scatter_default_45 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_122, 0, 77), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %select_scatter_default_45), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_569, 2), kwargs = {})
#   %select_scatter_default_47 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_123, 0, 76), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %select_scatter_default_47), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_572, 2), kwargs = {})
#   %select_scatter_default_49 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_124, 0, 75), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %select_scatter_default_49), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_575, 2), kwargs = {})
#   %select_scatter_default_51 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_125, 0, 74), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %select_scatter_default_51), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_578, 2), kwargs = {})
#   %select_scatter_default_53 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_126, 0, 73), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %select_scatter_default_53), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_581, 2), kwargs = {})
#   %select_scatter_default_55 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_127, 0, 72), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %select_scatter_default_55), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_584, 2), kwargs = {})
#   %select_scatter_default_57 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_128, 0, 71), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %select_scatter_default_57), kwargs = {})
#   %select_scatter_default_59 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_129, 0, 70), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %select_scatter_default_59), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_590, 2), kwargs = {})
#   %select_scatter_default_61 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_130, 0, 69), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %select_scatter_default_61), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_593, 2), kwargs = {})
#   %select_scatter_default_63 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_131, 0, 68), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %select_scatter_default_63), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_596, 2), kwargs = {})
#   %select_scatter_default_65 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_132, 0, 67), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %select_scatter_default_65), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_599, 2), kwargs = {})
#   %select_scatter_default_67 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_133, 0, 66), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %select_scatter_default_67), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_602, 2), kwargs = {})
#   %select_scatter_default_69 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_134, 0, 65), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %select_scatter_default_69), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_605, 2), kwargs = {})
#   %select_scatter_default_71 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_135, 0, 64), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %select_scatter_default_71), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_608, 2), kwargs = {})
#   %select_scatter_default_73 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_136, 0, 63), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %select_scatter_default_73), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_611, 2), kwargs = {})
#   %select_scatter_default_75 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_137, 0, 62), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %select_scatter_default_75), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_614, 2), kwargs = {})
#   %select_scatter_default_77 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_138, 0, 61), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %select_scatter_default_77), kwargs = {})
#   %select_scatter_default_79 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_139, 0, 60), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %select_scatter_default_79), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_620, 2), kwargs = {})
#   %select_scatter_default_81 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_140, 0, 59), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %select_scatter_default_81), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_623, 2), kwargs = {})
#   %select_scatter_default_83 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_141, 0, 58), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %select_scatter_default_83), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_626, 2), kwargs = {})
#   %select_scatter_default_85 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_142, 0, 57), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %select_scatter_default_85), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_629, 2), kwargs = {})
#   %select_scatter_default_87 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_143, 0, 56), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %select_scatter_default_87), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_632, 2), kwargs = {})
#   %select_scatter_default_89 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_144, 0, 55), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %select_scatter_default_89), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_635, 2), kwargs = {})
#   %select_scatter_default_91 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_145, 0, 54), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %select_scatter_default_91), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_638, 2), kwargs = {})
#   %select_scatter_default_93 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_146, 0, 53), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %select_scatter_default_93), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_641, 2), kwargs = {})
#   %select_scatter_default_95 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_147, 0, 52), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %select_scatter_default_95), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_644, 2), kwargs = {})
#   %select_scatter_default_97 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_148, 0, 51), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %select_scatter_default_97), kwargs = {})
#   %select_scatter_default_99 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_149, 0, 50), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, %select_scatter_default_99), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_650, 2), kwargs = {})
#   %select_scatter_default_101 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_150, 0, 49), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %select_scatter_default_101), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_653, 2), kwargs = {})
#   %select_scatter_default_103 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_151, 0, 48), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %select_scatter_default_103), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_656, 2), kwargs = {})
#   %select_scatter_default_105 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_152, 0, 47), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_50, %select_scatter_default_105), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_659, 2), kwargs = {})
#   %select_scatter_default_107 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_153, 0, 46), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, %select_scatter_default_107), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_662, 2), kwargs = {})
#   %select_scatter_default_109 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_154, 0, 45), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_52, %select_scatter_default_109), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_665, 2), kwargs = {})
#   %select_scatter_default_111 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_155, 0, 44), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_53, %select_scatter_default_111), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_668, 2), kwargs = {})
#   %select_scatter_default_113 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_156, 0, 43), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %select_scatter_default_113), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_671, 2), kwargs = {})
#   %select_scatter_default_115 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_157, 0, 42), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %select_scatter_default_115), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_674, 2), kwargs = {})
#   %select_scatter_default_117 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_158, 0, 41), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %select_scatter_default_117), kwargs = {})
#   %select_scatter_default_119 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_159, 0, 40), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_57, %select_scatter_default_119), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_680, 2), kwargs = {})
#   %select_scatter_default_121 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_160, 0, 39), kwargs = {})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %select_scatter_default_121), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_683, 2), kwargs = {})
#   %select_scatter_default_123 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_161, 0, 38), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %select_scatter_default_123), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_686, 2), kwargs = {})
#   %select_scatter_default_125 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_162, 0, 37), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %select_scatter_default_125), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_689, 2), kwargs = {})
#   %select_scatter_default_127 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_163, 0, 36), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_61, %select_scatter_default_127), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_692, 2), kwargs = {})
#   %select_scatter_default_129 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_164, 0, 35), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_62, %select_scatter_default_129), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_695, 2), kwargs = {})
#   %select_scatter_default_131 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_165, 0, 34), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %select_scatter_default_131), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_698, 2), kwargs = {})
#   %select_scatter_default_133 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_166, 0, 33), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, %select_scatter_default_133), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_701, 2), kwargs = {})
#   %select_scatter_default_135 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_167, 0, 32), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %select_scatter_default_135), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_704, 2), kwargs = {})
#   %select_scatter_default_137 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_168, 0, 31), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %select_scatter_default_137), kwargs = {})
#   %select_scatter_default_139 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_169, 0, 30), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %select_scatter_default_139), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_710, 2), kwargs = {})
#   %select_scatter_default_141 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_170, 0, 29), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %select_scatter_default_141), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_713, 2), kwargs = {})
#   %select_scatter_default_143 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_171, 0, 28), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %select_scatter_default_143), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_716, 2), kwargs = {})
#   %select_scatter_default_145 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_172, 0, 27), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %select_scatter_default_145), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_719, 2), kwargs = {})
#   %select_scatter_default_147 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_173, 0, 26), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_71, %select_scatter_default_147), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_722, 2), kwargs = {})
#   %select_scatter_default_149 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_174, 0, 25), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_72, %select_scatter_default_149), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_725, 2), kwargs = {})
#   %select_scatter_default_151 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_175, 0, 24), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_73, %select_scatter_default_151), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_728, 2), kwargs = {})
#   %select_scatter_default_153 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_176, 0, 23), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %select_scatter_default_153), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_731, 2), kwargs = {})
#   %select_scatter_default_155 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_177, 0, 22), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_75, %select_scatter_default_155), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_734, 2), kwargs = {})
#   %select_scatter_default_157 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_178, 0, 21), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %select_scatter_default_157), kwargs = {})
#   %select_scatter_default_159 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_179, 0, 20), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_77, %select_scatter_default_159), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_740, 2), kwargs = {})
#   %select_scatter_default_161 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_180, 0, 19), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %select_scatter_default_161), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_743, 2), kwargs = {})
#   %select_scatter_default_163 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_181, 0, 18), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, %select_scatter_default_163), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_746, 2), kwargs = {})
#   %select_scatter_default_165 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_182, 0, 17), kwargs = {})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %select_scatter_default_165), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_749, 2), kwargs = {})
#   %select_scatter_default_167 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_183, 0, 16), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_81, %select_scatter_default_167), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_752, 2), kwargs = {})
#   %select_scatter_default_169 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_184, 0, 15), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %select_scatter_default_169), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_755, 2), kwargs = {})
#   %select_scatter_default_171 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_185, 0, 14), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_83, %select_scatter_default_171), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_758, 2), kwargs = {})
#   %select_scatter_default_173 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_186, 0, 13), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_84, %select_scatter_default_173), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_761, 2), kwargs = {})
#   %select_scatter_default_175 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_187, 0, 12), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_85, %select_scatter_default_175), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_764, 2), kwargs = {})
#   %select_scatter_default_177 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_188, 0, 11), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %select_scatter_default_177), kwargs = {})
#   %select_scatter_default_179 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_189, 0, 10), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_87, %select_scatter_default_179), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_770, 2), kwargs = {})
#   %select_scatter_default_181 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_190, 0, 9), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %select_scatter_default_181), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_773, 2), kwargs = {})
#   %select_scatter_default_183 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_191, 0, 8), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %select_scatter_default_183), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_776, 2), kwargs = {})
#   %select_scatter_default_185 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_192, 0, 7), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %select_scatter_default_185), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_779, 2), kwargs = {})
#   %select_scatter_default_187 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_193, 0, 6), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %select_scatter_default_187), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_782, 2), kwargs = {})
#   %select_scatter_default_189 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_194, 0, 5), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %select_scatter_default_189), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_785, 2), kwargs = {})
#   %select_scatter_default_191 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_195, 0, 4), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %select_scatter_default_191), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_788, 2), kwargs = {})
#   %select_scatter_default_193 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_196, 0, 3), kwargs = {})
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %select_scatter_default_193), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_791, 2), kwargs = {})
#   %select_scatter_default_195 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_197, 0, 2), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %select_scatter_default_195), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_794, 2), kwargs = {})
#   %select_scatter_default_197 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_198, 0, 1), kwargs = {})
#   %add_97 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %select_scatter_default_197), kwargs = {})
#   %select_scatter_default_198 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul_199, 0, 0), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_97, %select_scatter_default_198), kwargs = {})
triton_poi_fused_add_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr1'], 'no_x_dim': False, 'num_load': 101, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (99))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (x0), xmask)
    tmp12 = tl.load(in_ptr0 + (98))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp23 = tl.load(in_ptr0 + (97))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp35 = tl.load(in_ptr0 + (96))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp49 = tl.load(in_ptr0 + (95))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp65 = tl.load(in_ptr0 + (94))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp83 = tl.load(in_ptr0 + (93))
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK])
    tmp103 = tl.load(in_ptr0 + (92))
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK])
    tmp125 = tl.load(in_ptr0 + (91))
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK])
    tmp140 = tl.load(in_ptr2 + (0))
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK])
    tmp146 = tl.load(in_ptr3 + (89))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp154 = tl.load(in_ptr3 + (88))
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK])
    tmp164 = tl.load(in_ptr3 + (87))
    tmp165 = tl.broadcast_to(tmp164, [XBLOCK])
    tmp176 = tl.load(in_ptr3 + (86))
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK])
    tmp190 = tl.load(in_ptr3 + (85))
    tmp191 = tl.broadcast_to(tmp190, [XBLOCK])
    tmp206 = tl.load(in_ptr3 + (84))
    tmp207 = tl.broadcast_to(tmp206, [XBLOCK])
    tmp224 = tl.load(in_ptr3 + (83))
    tmp225 = tl.broadcast_to(tmp224, [XBLOCK])
    tmp244 = tl.load(in_ptr3 + (82))
    tmp245 = tl.broadcast_to(tmp244, [XBLOCK])
    tmp266 = tl.load(in_ptr3 + (81))
    tmp267 = tl.broadcast_to(tmp266, [XBLOCK])
    tmp281 = tl.load(in_ptr4 + (0))
    tmp282 = tl.broadcast_to(tmp281, [XBLOCK])
    tmp287 = tl.load(in_ptr5 + (79))
    tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
    tmp295 = tl.load(in_ptr5 + (78))
    tmp296 = tl.broadcast_to(tmp295, [XBLOCK])
    tmp305 = tl.load(in_ptr5 + (77))
    tmp306 = tl.broadcast_to(tmp305, [XBLOCK])
    tmp317 = tl.load(in_ptr5 + (76))
    tmp318 = tl.broadcast_to(tmp317, [XBLOCK])
    tmp331 = tl.load(in_ptr5 + (75))
    tmp332 = tl.broadcast_to(tmp331, [XBLOCK])
    tmp347 = tl.load(in_ptr5 + (74))
    tmp348 = tl.broadcast_to(tmp347, [XBLOCK])
    tmp365 = tl.load(in_ptr5 + (73))
    tmp366 = tl.broadcast_to(tmp365, [XBLOCK])
    tmp385 = tl.load(in_ptr5 + (72))
    tmp386 = tl.broadcast_to(tmp385, [XBLOCK])
    tmp407 = tl.load(in_ptr5 + (71))
    tmp408 = tl.broadcast_to(tmp407, [XBLOCK])
    tmp422 = tl.load(in_ptr6 + (0))
    tmp423 = tl.broadcast_to(tmp422, [XBLOCK])
    tmp428 = tl.load(in_ptr7 + (69))
    tmp429 = tl.broadcast_to(tmp428, [XBLOCK])
    tmp436 = tl.load(in_ptr7 + (68))
    tmp437 = tl.broadcast_to(tmp436, [XBLOCK])
    tmp446 = tl.load(in_ptr7 + (67))
    tmp447 = tl.broadcast_to(tmp446, [XBLOCK])
    tmp458 = tl.load(in_ptr7 + (66))
    tmp459 = tl.broadcast_to(tmp458, [XBLOCK])
    tmp472 = tl.load(in_ptr7 + (65))
    tmp473 = tl.broadcast_to(tmp472, [XBLOCK])
    tmp488 = tl.load(in_ptr7 + (64))
    tmp489 = tl.broadcast_to(tmp488, [XBLOCK])
    tmp506 = tl.load(in_ptr7 + (63))
    tmp507 = tl.broadcast_to(tmp506, [XBLOCK])
    tmp526 = tl.load(in_ptr7 + (62))
    tmp527 = tl.broadcast_to(tmp526, [XBLOCK])
    tmp548 = tl.load(in_ptr7 + (61))
    tmp549 = tl.broadcast_to(tmp548, [XBLOCK])
    tmp563 = tl.load(in_ptr8 + (0))
    tmp564 = tl.broadcast_to(tmp563, [XBLOCK])
    tmp569 = tl.load(in_ptr9 + (59))
    tmp570 = tl.broadcast_to(tmp569, [XBLOCK])
    tmp577 = tl.load(in_ptr9 + (58))
    tmp578 = tl.broadcast_to(tmp577, [XBLOCK])
    tmp587 = tl.load(in_ptr9 + (57))
    tmp588 = tl.broadcast_to(tmp587, [XBLOCK])
    tmp599 = tl.load(in_ptr9 + (56))
    tmp600 = tl.broadcast_to(tmp599, [XBLOCK])
    tmp613 = tl.load(in_ptr9 + (55))
    tmp614 = tl.broadcast_to(tmp613, [XBLOCK])
    tmp629 = tl.load(in_ptr9 + (54))
    tmp630 = tl.broadcast_to(tmp629, [XBLOCK])
    tmp647 = tl.load(in_ptr9 + (53))
    tmp648 = tl.broadcast_to(tmp647, [XBLOCK])
    tmp667 = tl.load(in_ptr9 + (52))
    tmp668 = tl.broadcast_to(tmp667, [XBLOCK])
    tmp689 = tl.load(in_ptr9 + (51))
    tmp690 = tl.broadcast_to(tmp689, [XBLOCK])
    tmp704 = tl.load(in_ptr10 + (0))
    tmp705 = tl.broadcast_to(tmp704, [XBLOCK])
    tmp710 = tl.load(in_ptr11 + (49))
    tmp711 = tl.broadcast_to(tmp710, [XBLOCK])
    tmp718 = tl.load(in_ptr11 + (48))
    tmp719 = tl.broadcast_to(tmp718, [XBLOCK])
    tmp728 = tl.load(in_ptr11 + (47))
    tmp729 = tl.broadcast_to(tmp728, [XBLOCK])
    tmp740 = tl.load(in_ptr11 + (46))
    tmp741 = tl.broadcast_to(tmp740, [XBLOCK])
    tmp754 = tl.load(in_ptr11 + (45))
    tmp755 = tl.broadcast_to(tmp754, [XBLOCK])
    tmp770 = tl.load(in_ptr11 + (44))
    tmp771 = tl.broadcast_to(tmp770, [XBLOCK])
    tmp788 = tl.load(in_ptr11 + (43))
    tmp789 = tl.broadcast_to(tmp788, [XBLOCK])
    tmp808 = tl.load(in_ptr11 + (42))
    tmp809 = tl.broadcast_to(tmp808, [XBLOCK])
    tmp830 = tl.load(in_ptr11 + (41))
    tmp831 = tl.broadcast_to(tmp830, [XBLOCK])
    tmp845 = tl.load(in_ptr12 + (0))
    tmp846 = tl.broadcast_to(tmp845, [XBLOCK])
    tmp851 = tl.load(in_ptr13 + (39))
    tmp852 = tl.broadcast_to(tmp851, [XBLOCK])
    tmp859 = tl.load(in_ptr13 + (38))
    tmp860 = tl.broadcast_to(tmp859, [XBLOCK])
    tmp869 = tl.load(in_ptr13 + (37))
    tmp870 = tl.broadcast_to(tmp869, [XBLOCK])
    tmp881 = tl.load(in_ptr13 + (36))
    tmp882 = tl.broadcast_to(tmp881, [XBLOCK])
    tmp895 = tl.load(in_ptr13 + (35))
    tmp896 = tl.broadcast_to(tmp895, [XBLOCK])
    tmp911 = tl.load(in_ptr13 + (34))
    tmp912 = tl.broadcast_to(tmp911, [XBLOCK])
    tmp929 = tl.load(in_ptr13 + (33))
    tmp930 = tl.broadcast_to(tmp929, [XBLOCK])
    tmp949 = tl.load(in_ptr13 + (32))
    tmp950 = tl.broadcast_to(tmp949, [XBLOCK])
    tmp971 = tl.load(in_ptr13 + (31))
    tmp972 = tl.broadcast_to(tmp971, [XBLOCK])
    tmp986 = tl.load(in_ptr14 + (0))
    tmp987 = tl.broadcast_to(tmp986, [XBLOCK])
    tmp992 = tl.load(in_ptr15 + (29))
    tmp993 = tl.broadcast_to(tmp992, [XBLOCK])
    tmp1000 = tl.load(in_ptr15 + (28))
    tmp1001 = tl.broadcast_to(tmp1000, [XBLOCK])
    tmp1010 = tl.load(in_ptr15 + (27))
    tmp1011 = tl.broadcast_to(tmp1010, [XBLOCK])
    tmp1022 = tl.load(in_ptr15 + (26))
    tmp1023 = tl.broadcast_to(tmp1022, [XBLOCK])
    tmp1036 = tl.load(in_ptr15 + (25))
    tmp1037 = tl.broadcast_to(tmp1036, [XBLOCK])
    tmp1052 = tl.load(in_ptr15 + (24))
    tmp1053 = tl.broadcast_to(tmp1052, [XBLOCK])
    tmp1070 = tl.load(in_ptr15 + (23))
    tmp1071 = tl.broadcast_to(tmp1070, [XBLOCK])
    tmp1090 = tl.load(in_ptr15 + (22))
    tmp1091 = tl.broadcast_to(tmp1090, [XBLOCK])
    tmp1112 = tl.load(in_ptr15 + (21))
    tmp1113 = tl.broadcast_to(tmp1112, [XBLOCK])
    tmp1127 = tl.load(in_ptr16 + (0))
    tmp1128 = tl.broadcast_to(tmp1127, [XBLOCK])
    tmp1133 = tl.load(in_ptr17 + (19))
    tmp1134 = tl.broadcast_to(tmp1133, [XBLOCK])
    tmp1141 = tl.load(in_ptr17 + (18))
    tmp1142 = tl.broadcast_to(tmp1141, [XBLOCK])
    tmp1151 = tl.load(in_ptr17 + (17))
    tmp1152 = tl.broadcast_to(tmp1151, [XBLOCK])
    tmp1163 = tl.load(in_ptr17 + (16))
    tmp1164 = tl.broadcast_to(tmp1163, [XBLOCK])
    tmp1177 = tl.load(in_ptr17 + (15))
    tmp1178 = tl.broadcast_to(tmp1177, [XBLOCK])
    tmp1193 = tl.load(in_ptr17 + (14))
    tmp1194 = tl.broadcast_to(tmp1193, [XBLOCK])
    tmp1211 = tl.load(in_ptr17 + (13))
    tmp1212 = tl.broadcast_to(tmp1211, [XBLOCK])
    tmp1231 = tl.load(in_ptr17 + (12))
    tmp1232 = tl.broadcast_to(tmp1231, [XBLOCK])
    tmp1253 = tl.load(in_ptr17 + (11))
    tmp1254 = tl.broadcast_to(tmp1253, [XBLOCK])
    tmp1268 = tl.load(in_ptr18 + (0))
    tmp1269 = tl.broadcast_to(tmp1268, [XBLOCK])
    tmp1274 = tl.load(in_ptr19 + (9))
    tmp1275 = tl.broadcast_to(tmp1274, [XBLOCK])
    tmp1282 = tl.load(in_ptr19 + (8))
    tmp1283 = tl.broadcast_to(tmp1282, [XBLOCK])
    tmp1292 = tl.load(in_ptr19 + (7))
    tmp1293 = tl.broadcast_to(tmp1292, [XBLOCK])
    tmp1304 = tl.load(in_ptr19 + (6))
    tmp1305 = tl.broadcast_to(tmp1304, [XBLOCK])
    tmp1318 = tl.load(in_ptr19 + (5))
    tmp1319 = tl.broadcast_to(tmp1318, [XBLOCK])
    tmp1334 = tl.load(in_ptr19 + (4))
    tmp1335 = tl.broadcast_to(tmp1334, [XBLOCK])
    tmp1352 = tl.load(in_ptr19 + (3))
    tmp1353 = tl.broadcast_to(tmp1352, [XBLOCK])
    tmp1372 = tl.load(in_ptr19 + (2))
    tmp1373 = tl.broadcast_to(tmp1372, [XBLOCK])
    tmp1394 = tl.load(in_ptr19 + (1))
    tmp1395 = tl.broadcast_to(tmp1394, [XBLOCK])
    tmp1409 = tl.load(in_ptr20 + (0))
    tmp1410 = tl.broadcast_to(tmp1409, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 99, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tmp9 = tl.full([1], 98, tl.int32)
    tmp10 = tmp0 == tmp9
    tmp11 = tmp9 == tmp1
    tmp14 = 0.0
    tmp15 = tl.where(tmp11, tmp14, tmp13)
    tmp16 = tmp15 * tmp5
    tmp17 = tl.where(tmp10, tmp16, tmp7)
    tmp18 = tmp8 + tmp17
    tmp19 = tl.full([1], 97, tl.int32)
    tmp20 = tmp0 == tmp19
    tmp21 = tmp19 == tmp9
    tmp22 = tmp19 == tmp1
    tmp25 = tl.where(tmp22, tmp14, tmp24)
    tmp26 = tl.where(tmp21, tmp14, tmp25)
    tmp27 = tmp26 * tmp5
    tmp28 = tl.where(tmp20, tmp27, tmp7)
    tmp29 = tmp18 + tmp28
    tmp30 = tl.full([1], 96, tl.int32)
    tmp31 = tmp0 == tmp30
    tmp32 = tmp30 == tmp19
    tmp33 = tmp30 == tmp9
    tmp34 = tmp30 == tmp1
    tmp37 = tl.where(tmp34, tmp14, tmp36)
    tmp38 = tl.where(tmp33, tmp14, tmp37)
    tmp39 = tl.where(tmp32, tmp14, tmp38)
    tmp40 = tmp39 * tmp5
    tmp41 = tl.where(tmp31, tmp40, tmp7)
    tmp42 = tmp29 + tmp41
    tmp43 = tl.full([1], 95, tl.int32)
    tmp44 = tmp0 == tmp43
    tmp45 = tmp43 == tmp30
    tmp46 = tmp43 == tmp19
    tmp47 = tmp43 == tmp9
    tmp48 = tmp43 == tmp1
    tmp51 = tl.where(tmp48, tmp14, tmp50)
    tmp52 = tl.where(tmp47, tmp14, tmp51)
    tmp53 = tl.where(tmp46, tmp14, tmp52)
    tmp54 = tl.where(tmp45, tmp14, tmp53)
    tmp55 = tmp54 * tmp5
    tmp56 = tl.where(tmp44, tmp55, tmp7)
    tmp57 = tmp42 + tmp56
    tmp58 = tl.full([1], 94, tl.int32)
    tmp59 = tmp0 == tmp58
    tmp60 = tmp58 == tmp43
    tmp61 = tmp58 == tmp30
    tmp62 = tmp58 == tmp19
    tmp63 = tmp58 == tmp9
    tmp64 = tmp58 == tmp1
    tmp67 = tl.where(tmp64, tmp14, tmp66)
    tmp68 = tl.where(tmp63, tmp14, tmp67)
    tmp69 = tl.where(tmp62, tmp14, tmp68)
    tmp70 = tl.where(tmp61, tmp14, tmp69)
    tmp71 = tl.where(tmp60, tmp14, tmp70)
    tmp72 = tmp71 * tmp5
    tmp73 = tl.where(tmp59, tmp72, tmp7)
    tmp74 = tmp57 + tmp73
    tmp75 = tl.full([1], 93, tl.int32)
    tmp76 = tmp0 == tmp75
    tmp77 = tmp75 == tmp58
    tmp78 = tmp75 == tmp43
    tmp79 = tmp75 == tmp30
    tmp80 = tmp75 == tmp19
    tmp81 = tmp75 == tmp9
    tmp82 = tmp75 == tmp1
    tmp85 = tl.where(tmp82, tmp14, tmp84)
    tmp86 = tl.where(tmp81, tmp14, tmp85)
    tmp87 = tl.where(tmp80, tmp14, tmp86)
    tmp88 = tl.where(tmp79, tmp14, tmp87)
    tmp89 = tl.where(tmp78, tmp14, tmp88)
    tmp90 = tl.where(tmp77, tmp14, tmp89)
    tmp91 = tmp90 * tmp5
    tmp92 = tl.where(tmp76, tmp91, tmp7)
    tmp93 = tmp74 + tmp92
    tmp94 = tl.full([1], 92, tl.int32)
    tmp95 = tmp0 == tmp94
    tmp96 = tmp94 == tmp75
    tmp97 = tmp94 == tmp58
    tmp98 = tmp94 == tmp43
    tmp99 = tmp94 == tmp30
    tmp100 = tmp94 == tmp19
    tmp101 = tmp94 == tmp9
    tmp102 = tmp94 == tmp1
    tmp105 = tl.where(tmp102, tmp14, tmp104)
    tmp106 = tl.where(tmp101, tmp14, tmp105)
    tmp107 = tl.where(tmp100, tmp14, tmp106)
    tmp108 = tl.where(tmp99, tmp14, tmp107)
    tmp109 = tl.where(tmp98, tmp14, tmp108)
    tmp110 = tl.where(tmp97, tmp14, tmp109)
    tmp111 = tl.where(tmp96, tmp14, tmp110)
    tmp112 = tmp111 * tmp5
    tmp113 = tl.where(tmp95, tmp112, tmp7)
    tmp114 = tmp93 + tmp113
    tmp115 = tl.full([1], 91, tl.int32)
    tmp116 = tmp0 == tmp115
    tmp117 = tmp115 == tmp94
    tmp118 = tmp115 == tmp75
    tmp119 = tmp115 == tmp58
    tmp120 = tmp115 == tmp43
    tmp121 = tmp115 == tmp30
    tmp122 = tmp115 == tmp19
    tmp123 = tmp115 == tmp9
    tmp124 = tmp115 == tmp1
    tmp127 = tl.where(tmp124, tmp14, tmp126)
    tmp128 = tl.where(tmp123, tmp14, tmp127)
    tmp129 = tl.where(tmp122, tmp14, tmp128)
    tmp130 = tl.where(tmp121, tmp14, tmp129)
    tmp131 = tl.where(tmp120, tmp14, tmp130)
    tmp132 = tl.where(tmp119, tmp14, tmp131)
    tmp133 = tl.where(tmp118, tmp14, tmp132)
    tmp134 = tl.where(tmp117, tmp14, tmp133)
    tmp135 = tmp134 * tmp5
    tmp136 = tl.where(tmp116, tmp135, tmp7)
    tmp137 = tmp114 + tmp136
    tmp138 = tl.full([1], 90, tl.int32)
    tmp139 = tmp0 == tmp138
    tmp142 = tl.where(tmp139, tmp141, tmp7)
    tmp143 = tmp137 + tmp142
    tmp144 = tl.full([1], 89, tl.int32)
    tmp145 = tmp0 == tmp144
    tmp148 = tmp147 * tmp5
    tmp149 = tl.where(tmp145, tmp148, tmp7)
    tmp150 = tmp143 + tmp149
    tmp151 = tl.full([1], 88, tl.int32)
    tmp152 = tmp0 == tmp151
    tmp153 = tmp151 == tmp144
    tmp156 = tl.where(tmp153, tmp14, tmp155)
    tmp157 = tmp156 * tmp5
    tmp158 = tl.where(tmp152, tmp157, tmp7)
    tmp159 = tmp150 + tmp158
    tmp160 = tl.full([1], 87, tl.int32)
    tmp161 = tmp0 == tmp160
    tmp162 = tmp160 == tmp151
    tmp163 = tmp160 == tmp144
    tmp166 = tl.where(tmp163, tmp14, tmp165)
    tmp167 = tl.where(tmp162, tmp14, tmp166)
    tmp168 = tmp167 * tmp5
    tmp169 = tl.where(tmp161, tmp168, tmp7)
    tmp170 = tmp159 + tmp169
    tmp171 = tl.full([1], 86, tl.int32)
    tmp172 = tmp0 == tmp171
    tmp173 = tmp171 == tmp160
    tmp174 = tmp171 == tmp151
    tmp175 = tmp171 == tmp144
    tmp178 = tl.where(tmp175, tmp14, tmp177)
    tmp179 = tl.where(tmp174, tmp14, tmp178)
    tmp180 = tl.where(tmp173, tmp14, tmp179)
    tmp181 = tmp180 * tmp5
    tmp182 = tl.where(tmp172, tmp181, tmp7)
    tmp183 = tmp170 + tmp182
    tmp184 = tl.full([1], 85, tl.int32)
    tmp185 = tmp0 == tmp184
    tmp186 = tmp184 == tmp171
    tmp187 = tmp184 == tmp160
    tmp188 = tmp184 == tmp151
    tmp189 = tmp184 == tmp144
    tmp192 = tl.where(tmp189, tmp14, tmp191)
    tmp193 = tl.where(tmp188, tmp14, tmp192)
    tmp194 = tl.where(tmp187, tmp14, tmp193)
    tmp195 = tl.where(tmp186, tmp14, tmp194)
    tmp196 = tmp195 * tmp5
    tmp197 = tl.where(tmp185, tmp196, tmp7)
    tmp198 = tmp183 + tmp197
    tmp199 = tl.full([1], 84, tl.int32)
    tmp200 = tmp0 == tmp199
    tmp201 = tmp199 == tmp184
    tmp202 = tmp199 == tmp171
    tmp203 = tmp199 == tmp160
    tmp204 = tmp199 == tmp151
    tmp205 = tmp199 == tmp144
    tmp208 = tl.where(tmp205, tmp14, tmp207)
    tmp209 = tl.where(tmp204, tmp14, tmp208)
    tmp210 = tl.where(tmp203, tmp14, tmp209)
    tmp211 = tl.where(tmp202, tmp14, tmp210)
    tmp212 = tl.where(tmp201, tmp14, tmp211)
    tmp213 = tmp212 * tmp5
    tmp214 = tl.where(tmp200, tmp213, tmp7)
    tmp215 = tmp198 + tmp214
    tmp216 = tl.full([1], 83, tl.int32)
    tmp217 = tmp0 == tmp216
    tmp218 = tmp216 == tmp199
    tmp219 = tmp216 == tmp184
    tmp220 = tmp216 == tmp171
    tmp221 = tmp216 == tmp160
    tmp222 = tmp216 == tmp151
    tmp223 = tmp216 == tmp144
    tmp226 = tl.where(tmp223, tmp14, tmp225)
    tmp227 = tl.where(tmp222, tmp14, tmp226)
    tmp228 = tl.where(tmp221, tmp14, tmp227)
    tmp229 = tl.where(tmp220, tmp14, tmp228)
    tmp230 = tl.where(tmp219, tmp14, tmp229)
    tmp231 = tl.where(tmp218, tmp14, tmp230)
    tmp232 = tmp231 * tmp5
    tmp233 = tl.where(tmp217, tmp232, tmp7)
    tmp234 = tmp215 + tmp233
    tmp235 = tl.full([1], 82, tl.int32)
    tmp236 = tmp0 == tmp235
    tmp237 = tmp235 == tmp216
    tmp238 = tmp235 == tmp199
    tmp239 = tmp235 == tmp184
    tmp240 = tmp235 == tmp171
    tmp241 = tmp235 == tmp160
    tmp242 = tmp235 == tmp151
    tmp243 = tmp235 == tmp144
    tmp246 = tl.where(tmp243, tmp14, tmp245)
    tmp247 = tl.where(tmp242, tmp14, tmp246)
    tmp248 = tl.where(tmp241, tmp14, tmp247)
    tmp249 = tl.where(tmp240, tmp14, tmp248)
    tmp250 = tl.where(tmp239, tmp14, tmp249)
    tmp251 = tl.where(tmp238, tmp14, tmp250)
    tmp252 = tl.where(tmp237, tmp14, tmp251)
    tmp253 = tmp252 * tmp5
    tmp254 = tl.where(tmp236, tmp253, tmp7)
    tmp255 = tmp234 + tmp254
    tmp256 = tl.full([1], 81, tl.int32)
    tmp257 = tmp0 == tmp256
    tmp258 = tmp256 == tmp235
    tmp259 = tmp256 == tmp216
    tmp260 = tmp256 == tmp199
    tmp261 = tmp256 == tmp184
    tmp262 = tmp256 == tmp171
    tmp263 = tmp256 == tmp160
    tmp264 = tmp256 == tmp151
    tmp265 = tmp256 == tmp144
    tmp268 = tl.where(tmp265, tmp14, tmp267)
    tmp269 = tl.where(tmp264, tmp14, tmp268)
    tmp270 = tl.where(tmp263, tmp14, tmp269)
    tmp271 = tl.where(tmp262, tmp14, tmp270)
    tmp272 = tl.where(tmp261, tmp14, tmp271)
    tmp273 = tl.where(tmp260, tmp14, tmp272)
    tmp274 = tl.where(tmp259, tmp14, tmp273)
    tmp275 = tl.where(tmp258, tmp14, tmp274)
    tmp276 = tmp275 * tmp5
    tmp277 = tl.where(tmp257, tmp276, tmp7)
    tmp278 = tmp255 + tmp277
    tmp279 = tl.full([1], 80, tl.int32)
    tmp280 = tmp0 == tmp279
    tmp283 = tl.where(tmp280, tmp282, tmp7)
    tmp284 = tmp278 + tmp283
    tmp285 = tl.full([1], 79, tl.int32)
    tmp286 = tmp0 == tmp285
    tmp289 = tmp288 * tmp5
    tmp290 = tl.where(tmp286, tmp289, tmp7)
    tmp291 = tmp284 + tmp290
    tmp292 = tl.full([1], 78, tl.int32)
    tmp293 = tmp0 == tmp292
    tmp294 = tmp292 == tmp285
    tmp297 = tl.where(tmp294, tmp14, tmp296)
    tmp298 = tmp297 * tmp5
    tmp299 = tl.where(tmp293, tmp298, tmp7)
    tmp300 = tmp291 + tmp299
    tmp301 = tl.full([1], 77, tl.int32)
    tmp302 = tmp0 == tmp301
    tmp303 = tmp301 == tmp292
    tmp304 = tmp301 == tmp285
    tmp307 = tl.where(tmp304, tmp14, tmp306)
    tmp308 = tl.where(tmp303, tmp14, tmp307)
    tmp309 = tmp308 * tmp5
    tmp310 = tl.where(tmp302, tmp309, tmp7)
    tmp311 = tmp300 + tmp310
    tmp312 = tl.full([1], 76, tl.int32)
    tmp313 = tmp0 == tmp312
    tmp314 = tmp312 == tmp301
    tmp315 = tmp312 == tmp292
    tmp316 = tmp312 == tmp285
    tmp319 = tl.where(tmp316, tmp14, tmp318)
    tmp320 = tl.where(tmp315, tmp14, tmp319)
    tmp321 = tl.where(tmp314, tmp14, tmp320)
    tmp322 = tmp321 * tmp5
    tmp323 = tl.where(tmp313, tmp322, tmp7)
    tmp324 = tmp311 + tmp323
    tmp325 = tl.full([1], 75, tl.int32)
    tmp326 = tmp0 == tmp325
    tmp327 = tmp325 == tmp312
    tmp328 = tmp325 == tmp301
    tmp329 = tmp325 == tmp292
    tmp330 = tmp325 == tmp285
    tmp333 = tl.where(tmp330, tmp14, tmp332)
    tmp334 = tl.where(tmp329, tmp14, tmp333)
    tmp335 = tl.where(tmp328, tmp14, tmp334)
    tmp336 = tl.where(tmp327, tmp14, tmp335)
    tmp337 = tmp336 * tmp5
    tmp338 = tl.where(tmp326, tmp337, tmp7)
    tmp339 = tmp324 + tmp338
    tmp340 = tl.full([1], 74, tl.int32)
    tmp341 = tmp0 == tmp340
    tmp342 = tmp340 == tmp325
    tmp343 = tmp340 == tmp312
    tmp344 = tmp340 == tmp301
    tmp345 = tmp340 == tmp292
    tmp346 = tmp340 == tmp285
    tmp349 = tl.where(tmp346, tmp14, tmp348)
    tmp350 = tl.where(tmp345, tmp14, tmp349)
    tmp351 = tl.where(tmp344, tmp14, tmp350)
    tmp352 = tl.where(tmp343, tmp14, tmp351)
    tmp353 = tl.where(tmp342, tmp14, tmp352)
    tmp354 = tmp353 * tmp5
    tmp355 = tl.where(tmp341, tmp354, tmp7)
    tmp356 = tmp339 + tmp355
    tmp357 = tl.full([1], 73, tl.int32)
    tmp358 = tmp0 == tmp357
    tmp359 = tmp357 == tmp340
    tmp360 = tmp357 == tmp325
    tmp361 = tmp357 == tmp312
    tmp362 = tmp357 == tmp301
    tmp363 = tmp357 == tmp292
    tmp364 = tmp357 == tmp285
    tmp367 = tl.where(tmp364, tmp14, tmp366)
    tmp368 = tl.where(tmp363, tmp14, tmp367)
    tmp369 = tl.where(tmp362, tmp14, tmp368)
    tmp370 = tl.where(tmp361, tmp14, tmp369)
    tmp371 = tl.where(tmp360, tmp14, tmp370)
    tmp372 = tl.where(tmp359, tmp14, tmp371)
    tmp373 = tmp372 * tmp5
    tmp374 = tl.where(tmp358, tmp373, tmp7)
    tmp375 = tmp356 + tmp374
    tmp376 = tl.full([1], 72, tl.int32)
    tmp377 = tmp0 == tmp376
    tmp378 = tmp376 == tmp357
    tmp379 = tmp376 == tmp340
    tmp380 = tmp376 == tmp325
    tmp381 = tmp376 == tmp312
    tmp382 = tmp376 == tmp301
    tmp383 = tmp376 == tmp292
    tmp384 = tmp376 == tmp285
    tmp387 = tl.where(tmp384, tmp14, tmp386)
    tmp388 = tl.where(tmp383, tmp14, tmp387)
    tmp389 = tl.where(tmp382, tmp14, tmp388)
    tmp390 = tl.where(tmp381, tmp14, tmp389)
    tmp391 = tl.where(tmp380, tmp14, tmp390)
    tmp392 = tl.where(tmp379, tmp14, tmp391)
    tmp393 = tl.where(tmp378, tmp14, tmp392)
    tmp394 = tmp393 * tmp5
    tmp395 = tl.where(tmp377, tmp394, tmp7)
    tmp396 = tmp375 + tmp395
    tmp397 = tl.full([1], 71, tl.int32)
    tmp398 = tmp0 == tmp397
    tmp399 = tmp397 == tmp376
    tmp400 = tmp397 == tmp357
    tmp401 = tmp397 == tmp340
    tmp402 = tmp397 == tmp325
    tmp403 = tmp397 == tmp312
    tmp404 = tmp397 == tmp301
    tmp405 = tmp397 == tmp292
    tmp406 = tmp397 == tmp285
    tmp409 = tl.where(tmp406, tmp14, tmp408)
    tmp410 = tl.where(tmp405, tmp14, tmp409)
    tmp411 = tl.where(tmp404, tmp14, tmp410)
    tmp412 = tl.where(tmp403, tmp14, tmp411)
    tmp413 = tl.where(tmp402, tmp14, tmp412)
    tmp414 = tl.where(tmp401, tmp14, tmp413)
    tmp415 = tl.where(tmp400, tmp14, tmp414)
    tmp416 = tl.where(tmp399, tmp14, tmp415)
    tmp417 = tmp416 * tmp5
    tmp418 = tl.where(tmp398, tmp417, tmp7)
    tmp419 = tmp396 + tmp418
    tmp420 = tl.full([1], 70, tl.int32)
    tmp421 = tmp0 == tmp420
    tmp424 = tl.where(tmp421, tmp423, tmp7)
    tmp425 = tmp419 + tmp424
    tmp426 = tl.full([1], 69, tl.int32)
    tmp427 = tmp0 == tmp426
    tmp430 = tmp429 * tmp5
    tmp431 = tl.where(tmp427, tmp430, tmp7)
    tmp432 = tmp425 + tmp431
    tmp433 = tl.full([1], 68, tl.int32)
    tmp434 = tmp0 == tmp433
    tmp435 = tmp433 == tmp426
    tmp438 = tl.where(tmp435, tmp14, tmp437)
    tmp439 = tmp438 * tmp5
    tmp440 = tl.where(tmp434, tmp439, tmp7)
    tmp441 = tmp432 + tmp440
    tmp442 = tl.full([1], 67, tl.int32)
    tmp443 = tmp0 == tmp442
    tmp444 = tmp442 == tmp433
    tmp445 = tmp442 == tmp426
    tmp448 = tl.where(tmp445, tmp14, tmp447)
    tmp449 = tl.where(tmp444, tmp14, tmp448)
    tmp450 = tmp449 * tmp5
    tmp451 = tl.where(tmp443, tmp450, tmp7)
    tmp452 = tmp441 + tmp451
    tmp453 = tl.full([1], 66, tl.int32)
    tmp454 = tmp0 == tmp453
    tmp455 = tmp453 == tmp442
    tmp456 = tmp453 == tmp433
    tmp457 = tmp453 == tmp426
    tmp460 = tl.where(tmp457, tmp14, tmp459)
    tmp461 = tl.where(tmp456, tmp14, tmp460)
    tmp462 = tl.where(tmp455, tmp14, tmp461)
    tmp463 = tmp462 * tmp5
    tmp464 = tl.where(tmp454, tmp463, tmp7)
    tmp465 = tmp452 + tmp464
    tmp466 = tl.full([1], 65, tl.int32)
    tmp467 = tmp0 == tmp466
    tmp468 = tmp466 == tmp453
    tmp469 = tmp466 == tmp442
    tmp470 = tmp466 == tmp433
    tmp471 = tmp466 == tmp426
    tmp474 = tl.where(tmp471, tmp14, tmp473)
    tmp475 = tl.where(tmp470, tmp14, tmp474)
    tmp476 = tl.where(tmp469, tmp14, tmp475)
    tmp477 = tl.where(tmp468, tmp14, tmp476)
    tmp478 = tmp477 * tmp5
    tmp479 = tl.where(tmp467, tmp478, tmp7)
    tmp480 = tmp465 + tmp479
    tmp481 = tl.full([1], 64, tl.int32)
    tmp482 = tmp0 == tmp481
    tmp483 = tmp481 == tmp466
    tmp484 = tmp481 == tmp453
    tmp485 = tmp481 == tmp442
    tmp486 = tmp481 == tmp433
    tmp487 = tmp481 == tmp426
    tmp490 = tl.where(tmp487, tmp14, tmp489)
    tmp491 = tl.where(tmp486, tmp14, tmp490)
    tmp492 = tl.where(tmp485, tmp14, tmp491)
    tmp493 = tl.where(tmp484, tmp14, tmp492)
    tmp494 = tl.where(tmp483, tmp14, tmp493)
    tmp495 = tmp494 * tmp5
    tmp496 = tl.where(tmp482, tmp495, tmp7)
    tmp497 = tmp480 + tmp496
    tmp498 = tl.full([1], 63, tl.int32)
    tmp499 = tmp0 == tmp498
    tmp500 = tmp498 == tmp481
    tmp501 = tmp498 == tmp466
    tmp502 = tmp498 == tmp453
    tmp503 = tmp498 == tmp442
    tmp504 = tmp498 == tmp433
    tmp505 = tmp498 == tmp426
    tmp508 = tl.where(tmp505, tmp14, tmp507)
    tmp509 = tl.where(tmp504, tmp14, tmp508)
    tmp510 = tl.where(tmp503, tmp14, tmp509)
    tmp511 = tl.where(tmp502, tmp14, tmp510)
    tmp512 = tl.where(tmp501, tmp14, tmp511)
    tmp513 = tl.where(tmp500, tmp14, tmp512)
    tmp514 = tmp513 * tmp5
    tmp515 = tl.where(tmp499, tmp514, tmp7)
    tmp516 = tmp497 + tmp515
    tmp517 = tl.full([1], 62, tl.int32)
    tmp518 = tmp0 == tmp517
    tmp519 = tmp517 == tmp498
    tmp520 = tmp517 == tmp481
    tmp521 = tmp517 == tmp466
    tmp522 = tmp517 == tmp453
    tmp523 = tmp517 == tmp442
    tmp524 = tmp517 == tmp433
    tmp525 = tmp517 == tmp426
    tmp528 = tl.where(tmp525, tmp14, tmp527)
    tmp529 = tl.where(tmp524, tmp14, tmp528)
    tmp530 = tl.where(tmp523, tmp14, tmp529)
    tmp531 = tl.where(tmp522, tmp14, tmp530)
    tmp532 = tl.where(tmp521, tmp14, tmp531)
    tmp533 = tl.where(tmp520, tmp14, tmp532)
    tmp534 = tl.where(tmp519, tmp14, tmp533)
    tmp535 = tmp534 * tmp5
    tmp536 = tl.where(tmp518, tmp535, tmp7)
    tmp537 = tmp516 + tmp536
    tmp538 = tl.full([1], 61, tl.int32)
    tmp539 = tmp0 == tmp538
    tmp540 = tmp538 == tmp517
    tmp541 = tmp538 == tmp498
    tmp542 = tmp538 == tmp481
    tmp543 = tmp538 == tmp466
    tmp544 = tmp538 == tmp453
    tmp545 = tmp538 == tmp442
    tmp546 = tmp538 == tmp433
    tmp547 = tmp538 == tmp426
    tmp550 = tl.where(tmp547, tmp14, tmp549)
    tmp551 = tl.where(tmp546, tmp14, tmp550)
    tmp552 = tl.where(tmp545, tmp14, tmp551)
    tmp553 = tl.where(tmp544, tmp14, tmp552)
    tmp554 = tl.where(tmp543, tmp14, tmp553)
    tmp555 = tl.where(tmp542, tmp14, tmp554)
    tmp556 = tl.where(tmp541, tmp14, tmp555)
    tmp557 = tl.where(tmp540, tmp14, tmp556)
    tmp558 = tmp557 * tmp5
    tmp559 = tl.where(tmp539, tmp558, tmp7)
    tmp560 = tmp537 + tmp559
    tmp561 = tl.full([1], 60, tl.int32)
    tmp562 = tmp0 == tmp561
    tmp565 = tl.where(tmp562, tmp564, tmp7)
    tmp566 = tmp560 + tmp565
    tmp567 = tl.full([1], 59, tl.int32)
    tmp568 = tmp0 == tmp567
    tmp571 = tmp570 * tmp5
    tmp572 = tl.where(tmp568, tmp571, tmp7)
    tmp573 = tmp566 + tmp572
    tmp574 = tl.full([1], 58, tl.int32)
    tmp575 = tmp0 == tmp574
    tmp576 = tmp574 == tmp567
    tmp579 = tl.where(tmp576, tmp14, tmp578)
    tmp580 = tmp579 * tmp5
    tmp581 = tl.where(tmp575, tmp580, tmp7)
    tmp582 = tmp573 + tmp581
    tmp583 = tl.full([1], 57, tl.int32)
    tmp584 = tmp0 == tmp583
    tmp585 = tmp583 == tmp574
    tmp586 = tmp583 == tmp567
    tmp589 = tl.where(tmp586, tmp14, tmp588)
    tmp590 = tl.where(tmp585, tmp14, tmp589)
    tmp591 = tmp590 * tmp5
    tmp592 = tl.where(tmp584, tmp591, tmp7)
    tmp593 = tmp582 + tmp592
    tmp594 = tl.full([1], 56, tl.int32)
    tmp595 = tmp0 == tmp594
    tmp596 = tmp594 == tmp583
    tmp597 = tmp594 == tmp574
    tmp598 = tmp594 == tmp567
    tmp601 = tl.where(tmp598, tmp14, tmp600)
    tmp602 = tl.where(tmp597, tmp14, tmp601)
    tmp603 = tl.where(tmp596, tmp14, tmp602)
    tmp604 = tmp603 * tmp5
    tmp605 = tl.where(tmp595, tmp604, tmp7)
    tmp606 = tmp593 + tmp605
    tmp607 = tl.full([1], 55, tl.int32)
    tmp608 = tmp0 == tmp607
    tmp609 = tmp607 == tmp594
    tmp610 = tmp607 == tmp583
    tmp611 = tmp607 == tmp574
    tmp612 = tmp607 == tmp567
    tmp615 = tl.where(tmp612, tmp14, tmp614)
    tmp616 = tl.where(tmp611, tmp14, tmp615)
    tmp617 = tl.where(tmp610, tmp14, tmp616)
    tmp618 = tl.where(tmp609, tmp14, tmp617)
    tmp619 = tmp618 * tmp5
    tmp620 = tl.where(tmp608, tmp619, tmp7)
    tmp621 = tmp606 + tmp620
    tmp622 = tl.full([1], 54, tl.int32)
    tmp623 = tmp0 == tmp622
    tmp624 = tmp622 == tmp607
    tmp625 = tmp622 == tmp594
    tmp626 = tmp622 == tmp583
    tmp627 = tmp622 == tmp574
    tmp628 = tmp622 == tmp567
    tmp631 = tl.where(tmp628, tmp14, tmp630)
    tmp632 = tl.where(tmp627, tmp14, tmp631)
    tmp633 = tl.where(tmp626, tmp14, tmp632)
    tmp634 = tl.where(tmp625, tmp14, tmp633)
    tmp635 = tl.where(tmp624, tmp14, tmp634)
    tmp636 = tmp635 * tmp5
    tmp637 = tl.where(tmp623, tmp636, tmp7)
    tmp638 = tmp621 + tmp637
    tmp639 = tl.full([1], 53, tl.int32)
    tmp640 = tmp0 == tmp639
    tmp641 = tmp639 == tmp622
    tmp642 = tmp639 == tmp607
    tmp643 = tmp639 == tmp594
    tmp644 = tmp639 == tmp583
    tmp645 = tmp639 == tmp574
    tmp646 = tmp639 == tmp567
    tmp649 = tl.where(tmp646, tmp14, tmp648)
    tmp650 = tl.where(tmp645, tmp14, tmp649)
    tmp651 = tl.where(tmp644, tmp14, tmp650)
    tmp652 = tl.where(tmp643, tmp14, tmp651)
    tmp653 = tl.where(tmp642, tmp14, tmp652)
    tmp654 = tl.where(tmp641, tmp14, tmp653)
    tmp655 = tmp654 * tmp5
    tmp656 = tl.where(tmp640, tmp655, tmp7)
    tmp657 = tmp638 + tmp656
    tmp658 = tl.full([1], 52, tl.int32)
    tmp659 = tmp0 == tmp658
    tmp660 = tmp658 == tmp639
    tmp661 = tmp658 == tmp622
    tmp662 = tmp658 == tmp607
    tmp663 = tmp658 == tmp594
    tmp664 = tmp658 == tmp583
    tmp665 = tmp658 == tmp574
    tmp666 = tmp658 == tmp567
    tmp669 = tl.where(tmp666, tmp14, tmp668)
    tmp670 = tl.where(tmp665, tmp14, tmp669)
    tmp671 = tl.where(tmp664, tmp14, tmp670)
    tmp672 = tl.where(tmp663, tmp14, tmp671)
    tmp673 = tl.where(tmp662, tmp14, tmp672)
    tmp674 = tl.where(tmp661, tmp14, tmp673)
    tmp675 = tl.where(tmp660, tmp14, tmp674)
    tmp676 = tmp675 * tmp5
    tmp677 = tl.where(tmp659, tmp676, tmp7)
    tmp678 = tmp657 + tmp677
    tmp679 = tl.full([1], 51, tl.int32)
    tmp680 = tmp0 == tmp679
    tmp681 = tmp679 == tmp658
    tmp682 = tmp679 == tmp639
    tmp683 = tmp679 == tmp622
    tmp684 = tmp679 == tmp607
    tmp685 = tmp679 == tmp594
    tmp686 = tmp679 == tmp583
    tmp687 = tmp679 == tmp574
    tmp688 = tmp679 == tmp567
    tmp691 = tl.where(tmp688, tmp14, tmp690)
    tmp692 = tl.where(tmp687, tmp14, tmp691)
    tmp693 = tl.where(tmp686, tmp14, tmp692)
    tmp694 = tl.where(tmp685, tmp14, tmp693)
    tmp695 = tl.where(tmp684, tmp14, tmp694)
    tmp696 = tl.where(tmp683, tmp14, tmp695)
    tmp697 = tl.where(tmp682, tmp14, tmp696)
    tmp698 = tl.where(tmp681, tmp14, tmp697)
    tmp699 = tmp698 * tmp5
    tmp700 = tl.where(tmp680, tmp699, tmp7)
    tmp701 = tmp678 + tmp700
    tmp702 = tl.full([1], 50, tl.int32)
    tmp703 = tmp0 == tmp702
    tmp706 = tl.where(tmp703, tmp705, tmp7)
    tmp707 = tmp701 + tmp706
    tmp708 = tl.full([1], 49, tl.int32)
    tmp709 = tmp0 == tmp708
    tmp712 = tmp711 * tmp5
    tmp713 = tl.where(tmp709, tmp712, tmp7)
    tmp714 = tmp707 + tmp713
    tmp715 = tl.full([1], 48, tl.int32)
    tmp716 = tmp0 == tmp715
    tmp717 = tmp715 == tmp708
    tmp720 = tl.where(tmp717, tmp14, tmp719)
    tmp721 = tmp720 * tmp5
    tmp722 = tl.where(tmp716, tmp721, tmp7)
    tmp723 = tmp714 + tmp722
    tmp724 = tl.full([1], 47, tl.int32)
    tmp725 = tmp0 == tmp724
    tmp726 = tmp724 == tmp715
    tmp727 = tmp724 == tmp708
    tmp730 = tl.where(tmp727, tmp14, tmp729)
    tmp731 = tl.where(tmp726, tmp14, tmp730)
    tmp732 = tmp731 * tmp5
    tmp733 = tl.where(tmp725, tmp732, tmp7)
    tmp734 = tmp723 + tmp733
    tmp735 = tl.full([1], 46, tl.int32)
    tmp736 = tmp0 == tmp735
    tmp737 = tmp735 == tmp724
    tmp738 = tmp735 == tmp715
    tmp739 = tmp735 == tmp708
    tmp742 = tl.where(tmp739, tmp14, tmp741)
    tmp743 = tl.where(tmp738, tmp14, tmp742)
    tmp744 = tl.where(tmp737, tmp14, tmp743)
    tmp745 = tmp744 * tmp5
    tmp746 = tl.where(tmp736, tmp745, tmp7)
    tmp747 = tmp734 + tmp746
    tmp748 = tl.full([1], 45, tl.int32)
    tmp749 = tmp0 == tmp748
    tmp750 = tmp748 == tmp735
    tmp751 = tmp748 == tmp724
    tmp752 = tmp748 == tmp715
    tmp753 = tmp748 == tmp708
    tmp756 = tl.where(tmp753, tmp14, tmp755)
    tmp757 = tl.where(tmp752, tmp14, tmp756)
    tmp758 = tl.where(tmp751, tmp14, tmp757)
    tmp759 = tl.where(tmp750, tmp14, tmp758)
    tmp760 = tmp759 * tmp5
    tmp761 = tl.where(tmp749, tmp760, tmp7)
    tmp762 = tmp747 + tmp761
    tmp763 = tl.full([1], 44, tl.int32)
    tmp764 = tmp0 == tmp763
    tmp765 = tmp763 == tmp748
    tmp766 = tmp763 == tmp735
    tmp767 = tmp763 == tmp724
    tmp768 = tmp763 == tmp715
    tmp769 = tmp763 == tmp708
    tmp772 = tl.where(tmp769, tmp14, tmp771)
    tmp773 = tl.where(tmp768, tmp14, tmp772)
    tmp774 = tl.where(tmp767, tmp14, tmp773)
    tmp775 = tl.where(tmp766, tmp14, tmp774)
    tmp776 = tl.where(tmp765, tmp14, tmp775)
    tmp777 = tmp776 * tmp5
    tmp778 = tl.where(tmp764, tmp777, tmp7)
    tmp779 = tmp762 + tmp778
    tmp780 = tl.full([1], 43, tl.int32)
    tmp781 = tmp0 == tmp780
    tmp782 = tmp780 == tmp763
    tmp783 = tmp780 == tmp748
    tmp784 = tmp780 == tmp735
    tmp785 = tmp780 == tmp724
    tmp786 = tmp780 == tmp715
    tmp787 = tmp780 == tmp708
    tmp790 = tl.where(tmp787, tmp14, tmp789)
    tmp791 = tl.where(tmp786, tmp14, tmp790)
    tmp792 = tl.where(tmp785, tmp14, tmp791)
    tmp793 = tl.where(tmp784, tmp14, tmp792)
    tmp794 = tl.where(tmp783, tmp14, tmp793)
    tmp795 = tl.where(tmp782, tmp14, tmp794)
    tmp796 = tmp795 * tmp5
    tmp797 = tl.where(tmp781, tmp796, tmp7)
    tmp798 = tmp779 + tmp797
    tmp799 = tl.full([1], 42, tl.int32)
    tmp800 = tmp0 == tmp799
    tmp801 = tmp799 == tmp780
    tmp802 = tmp799 == tmp763
    tmp803 = tmp799 == tmp748
    tmp804 = tmp799 == tmp735
    tmp805 = tmp799 == tmp724
    tmp806 = tmp799 == tmp715
    tmp807 = tmp799 == tmp708
    tmp810 = tl.where(tmp807, tmp14, tmp809)
    tmp811 = tl.where(tmp806, tmp14, tmp810)
    tmp812 = tl.where(tmp805, tmp14, tmp811)
    tmp813 = tl.where(tmp804, tmp14, tmp812)
    tmp814 = tl.where(tmp803, tmp14, tmp813)
    tmp815 = tl.where(tmp802, tmp14, tmp814)
    tmp816 = tl.where(tmp801, tmp14, tmp815)
    tmp817 = tmp816 * tmp5
    tmp818 = tl.where(tmp800, tmp817, tmp7)
    tmp819 = tmp798 + tmp818
    tmp820 = tl.full([1], 41, tl.int32)
    tmp821 = tmp0 == tmp820
    tmp822 = tmp820 == tmp799
    tmp823 = tmp820 == tmp780
    tmp824 = tmp820 == tmp763
    tmp825 = tmp820 == tmp748
    tmp826 = tmp820 == tmp735
    tmp827 = tmp820 == tmp724
    tmp828 = tmp820 == tmp715
    tmp829 = tmp820 == tmp708
    tmp832 = tl.where(tmp829, tmp14, tmp831)
    tmp833 = tl.where(tmp828, tmp14, tmp832)
    tmp834 = tl.where(tmp827, tmp14, tmp833)
    tmp835 = tl.where(tmp826, tmp14, tmp834)
    tmp836 = tl.where(tmp825, tmp14, tmp835)
    tmp837 = tl.where(tmp824, tmp14, tmp836)
    tmp838 = tl.where(tmp823, tmp14, tmp837)
    tmp839 = tl.where(tmp822, tmp14, tmp838)
    tmp840 = tmp839 * tmp5
    tmp841 = tl.where(tmp821, tmp840, tmp7)
    tmp842 = tmp819 + tmp841
    tmp843 = tl.full([1], 40, tl.int32)
    tmp844 = tmp0 == tmp843
    tmp847 = tl.where(tmp844, tmp846, tmp7)
    tmp848 = tmp842 + tmp847
    tmp849 = tl.full([1], 39, tl.int32)
    tmp850 = tmp0 == tmp849
    tmp853 = tmp852 * tmp5
    tmp854 = tl.where(tmp850, tmp853, tmp7)
    tmp855 = tmp848 + tmp854
    tmp856 = tl.full([1], 38, tl.int32)
    tmp857 = tmp0 == tmp856
    tmp858 = tmp856 == tmp849
    tmp861 = tl.where(tmp858, tmp14, tmp860)
    tmp862 = tmp861 * tmp5
    tmp863 = tl.where(tmp857, tmp862, tmp7)
    tmp864 = tmp855 + tmp863
    tmp865 = tl.full([1], 37, tl.int32)
    tmp866 = tmp0 == tmp865
    tmp867 = tmp865 == tmp856
    tmp868 = tmp865 == tmp849
    tmp871 = tl.where(tmp868, tmp14, tmp870)
    tmp872 = tl.where(tmp867, tmp14, tmp871)
    tmp873 = tmp872 * tmp5
    tmp874 = tl.where(tmp866, tmp873, tmp7)
    tmp875 = tmp864 + tmp874
    tmp876 = tl.full([1], 36, tl.int32)
    tmp877 = tmp0 == tmp876
    tmp878 = tmp876 == tmp865
    tmp879 = tmp876 == tmp856
    tmp880 = tmp876 == tmp849
    tmp883 = tl.where(tmp880, tmp14, tmp882)
    tmp884 = tl.where(tmp879, tmp14, tmp883)
    tmp885 = tl.where(tmp878, tmp14, tmp884)
    tmp886 = tmp885 * tmp5
    tmp887 = tl.where(tmp877, tmp886, tmp7)
    tmp888 = tmp875 + tmp887
    tmp889 = tl.full([1], 35, tl.int32)
    tmp890 = tmp0 == tmp889
    tmp891 = tmp889 == tmp876
    tmp892 = tmp889 == tmp865
    tmp893 = tmp889 == tmp856
    tmp894 = tmp889 == tmp849
    tmp897 = tl.where(tmp894, tmp14, tmp896)
    tmp898 = tl.where(tmp893, tmp14, tmp897)
    tmp899 = tl.where(tmp892, tmp14, tmp898)
    tmp900 = tl.where(tmp891, tmp14, tmp899)
    tmp901 = tmp900 * tmp5
    tmp902 = tl.where(tmp890, tmp901, tmp7)
    tmp903 = tmp888 + tmp902
    tmp904 = tl.full([1], 34, tl.int32)
    tmp905 = tmp0 == tmp904
    tmp906 = tmp904 == tmp889
    tmp907 = tmp904 == tmp876
    tmp908 = tmp904 == tmp865
    tmp909 = tmp904 == tmp856
    tmp910 = tmp904 == tmp849
    tmp913 = tl.where(tmp910, tmp14, tmp912)
    tmp914 = tl.where(tmp909, tmp14, tmp913)
    tmp915 = tl.where(tmp908, tmp14, tmp914)
    tmp916 = tl.where(tmp907, tmp14, tmp915)
    tmp917 = tl.where(tmp906, tmp14, tmp916)
    tmp918 = tmp917 * tmp5
    tmp919 = tl.where(tmp905, tmp918, tmp7)
    tmp920 = tmp903 + tmp919
    tmp921 = tl.full([1], 33, tl.int32)
    tmp922 = tmp0 == tmp921
    tmp923 = tmp921 == tmp904
    tmp924 = tmp921 == tmp889
    tmp925 = tmp921 == tmp876
    tmp926 = tmp921 == tmp865
    tmp927 = tmp921 == tmp856
    tmp928 = tmp921 == tmp849
    tmp931 = tl.where(tmp928, tmp14, tmp930)
    tmp932 = tl.where(tmp927, tmp14, tmp931)
    tmp933 = tl.where(tmp926, tmp14, tmp932)
    tmp934 = tl.where(tmp925, tmp14, tmp933)
    tmp935 = tl.where(tmp924, tmp14, tmp934)
    tmp936 = tl.where(tmp923, tmp14, tmp935)
    tmp937 = tmp936 * tmp5
    tmp938 = tl.where(tmp922, tmp937, tmp7)
    tmp939 = tmp920 + tmp938
    tmp940 = tl.full([1], 32, tl.int32)
    tmp941 = tmp0 == tmp940
    tmp942 = tmp940 == tmp921
    tmp943 = tmp940 == tmp904
    tmp944 = tmp940 == tmp889
    tmp945 = tmp940 == tmp876
    tmp946 = tmp940 == tmp865
    tmp947 = tmp940 == tmp856
    tmp948 = tmp940 == tmp849
    tmp951 = tl.where(tmp948, tmp14, tmp950)
    tmp952 = tl.where(tmp947, tmp14, tmp951)
    tmp953 = tl.where(tmp946, tmp14, tmp952)
    tmp954 = tl.where(tmp945, tmp14, tmp953)
    tmp955 = tl.where(tmp944, tmp14, tmp954)
    tmp956 = tl.where(tmp943, tmp14, tmp955)
    tmp957 = tl.where(tmp942, tmp14, tmp956)
    tmp958 = tmp957 * tmp5
    tmp959 = tl.where(tmp941, tmp958, tmp7)
    tmp960 = tmp939 + tmp959
    tmp961 = tl.full([1], 31, tl.int32)
    tmp962 = tmp0 == tmp961
    tmp963 = tmp961 == tmp940
    tmp964 = tmp961 == tmp921
    tmp965 = tmp961 == tmp904
    tmp966 = tmp961 == tmp889
    tmp967 = tmp961 == tmp876
    tmp968 = tmp961 == tmp865
    tmp969 = tmp961 == tmp856
    tmp970 = tmp961 == tmp849
    tmp973 = tl.where(tmp970, tmp14, tmp972)
    tmp974 = tl.where(tmp969, tmp14, tmp973)
    tmp975 = tl.where(tmp968, tmp14, tmp974)
    tmp976 = tl.where(tmp967, tmp14, tmp975)
    tmp977 = tl.where(tmp966, tmp14, tmp976)
    tmp978 = tl.where(tmp965, tmp14, tmp977)
    tmp979 = tl.where(tmp964, tmp14, tmp978)
    tmp980 = tl.where(tmp963, tmp14, tmp979)
    tmp981 = tmp980 * tmp5
    tmp982 = tl.where(tmp962, tmp981, tmp7)
    tmp983 = tmp960 + tmp982
    tmp984 = tl.full([1], 30, tl.int32)
    tmp985 = tmp0 == tmp984
    tmp988 = tl.where(tmp985, tmp987, tmp7)
    tmp989 = tmp983 + tmp988
    tmp990 = tl.full([1], 29, tl.int32)
    tmp991 = tmp0 == tmp990
    tmp994 = tmp993 * tmp5
    tmp995 = tl.where(tmp991, tmp994, tmp7)
    tmp996 = tmp989 + tmp995
    tmp997 = tl.full([1], 28, tl.int32)
    tmp998 = tmp0 == tmp997
    tmp999 = tmp997 == tmp990
    tmp1002 = tl.where(tmp999, tmp14, tmp1001)
    tmp1003 = tmp1002 * tmp5
    tmp1004 = tl.where(tmp998, tmp1003, tmp7)
    tmp1005 = tmp996 + tmp1004
    tmp1006 = tl.full([1], 27, tl.int32)
    tmp1007 = tmp0 == tmp1006
    tmp1008 = tmp1006 == tmp997
    tmp1009 = tmp1006 == tmp990
    tmp1012 = tl.where(tmp1009, tmp14, tmp1011)
    tmp1013 = tl.where(tmp1008, tmp14, tmp1012)
    tmp1014 = tmp1013 * tmp5
    tmp1015 = tl.where(tmp1007, tmp1014, tmp7)
    tmp1016 = tmp1005 + tmp1015
    tmp1017 = tl.full([1], 26, tl.int32)
    tmp1018 = tmp0 == tmp1017
    tmp1019 = tmp1017 == tmp1006
    tmp1020 = tmp1017 == tmp997
    tmp1021 = tmp1017 == tmp990
    tmp1024 = tl.where(tmp1021, tmp14, tmp1023)
    tmp1025 = tl.where(tmp1020, tmp14, tmp1024)
    tmp1026 = tl.where(tmp1019, tmp14, tmp1025)
    tmp1027 = tmp1026 * tmp5
    tmp1028 = tl.where(tmp1018, tmp1027, tmp7)
    tmp1029 = tmp1016 + tmp1028
    tmp1030 = tl.full([1], 25, tl.int32)
    tmp1031 = tmp0 == tmp1030
    tmp1032 = tmp1030 == tmp1017
    tmp1033 = tmp1030 == tmp1006
    tmp1034 = tmp1030 == tmp997
    tmp1035 = tmp1030 == tmp990
    tmp1038 = tl.where(tmp1035, tmp14, tmp1037)
    tmp1039 = tl.where(tmp1034, tmp14, tmp1038)
    tmp1040 = tl.where(tmp1033, tmp14, tmp1039)
    tmp1041 = tl.where(tmp1032, tmp14, tmp1040)
    tmp1042 = tmp1041 * tmp5
    tmp1043 = tl.where(tmp1031, tmp1042, tmp7)
    tmp1044 = tmp1029 + tmp1043
    tmp1045 = tl.full([1], 24, tl.int32)
    tmp1046 = tmp0 == tmp1045
    tmp1047 = tmp1045 == tmp1030
    tmp1048 = tmp1045 == tmp1017
    tmp1049 = tmp1045 == tmp1006
    tmp1050 = tmp1045 == tmp997
    tmp1051 = tmp1045 == tmp990
    tmp1054 = tl.where(tmp1051, tmp14, tmp1053)
    tmp1055 = tl.where(tmp1050, tmp14, tmp1054)
    tmp1056 = tl.where(tmp1049, tmp14, tmp1055)
    tmp1057 = tl.where(tmp1048, tmp14, tmp1056)
    tmp1058 = tl.where(tmp1047, tmp14, tmp1057)
    tmp1059 = tmp1058 * tmp5
    tmp1060 = tl.where(tmp1046, tmp1059, tmp7)
    tmp1061 = tmp1044 + tmp1060
    tmp1062 = tl.full([1], 23, tl.int32)
    tmp1063 = tmp0 == tmp1062
    tmp1064 = tmp1062 == tmp1045
    tmp1065 = tmp1062 == tmp1030
    tmp1066 = tmp1062 == tmp1017
    tmp1067 = tmp1062 == tmp1006
    tmp1068 = tmp1062 == tmp997
    tmp1069 = tmp1062 == tmp990
    tmp1072 = tl.where(tmp1069, tmp14, tmp1071)
    tmp1073 = tl.where(tmp1068, tmp14, tmp1072)
    tmp1074 = tl.where(tmp1067, tmp14, tmp1073)
    tmp1075 = tl.where(tmp1066, tmp14, tmp1074)
    tmp1076 = tl.where(tmp1065, tmp14, tmp1075)
    tmp1077 = tl.where(tmp1064, tmp14, tmp1076)
    tmp1078 = tmp1077 * tmp5
    tmp1079 = tl.where(tmp1063, tmp1078, tmp7)
    tmp1080 = tmp1061 + tmp1079
    tmp1081 = tl.full([1], 22, tl.int32)
    tmp1082 = tmp0 == tmp1081
    tmp1083 = tmp1081 == tmp1062
    tmp1084 = tmp1081 == tmp1045
    tmp1085 = tmp1081 == tmp1030
    tmp1086 = tmp1081 == tmp1017
    tmp1087 = tmp1081 == tmp1006
    tmp1088 = tmp1081 == tmp997
    tmp1089 = tmp1081 == tmp990
    tmp1092 = tl.where(tmp1089, tmp14, tmp1091)
    tmp1093 = tl.where(tmp1088, tmp14, tmp1092)
    tmp1094 = tl.where(tmp1087, tmp14, tmp1093)
    tmp1095 = tl.where(tmp1086, tmp14, tmp1094)
    tmp1096 = tl.where(tmp1085, tmp14, tmp1095)
    tmp1097 = tl.where(tmp1084, tmp14, tmp1096)
    tmp1098 = tl.where(tmp1083, tmp14, tmp1097)
    tmp1099 = tmp1098 * tmp5
    tmp1100 = tl.where(tmp1082, tmp1099, tmp7)
    tmp1101 = tmp1080 + tmp1100
    tmp1102 = tl.full([1], 21, tl.int32)
    tmp1103 = tmp0 == tmp1102
    tmp1104 = tmp1102 == tmp1081
    tmp1105 = tmp1102 == tmp1062
    tmp1106 = tmp1102 == tmp1045
    tmp1107 = tmp1102 == tmp1030
    tmp1108 = tmp1102 == tmp1017
    tmp1109 = tmp1102 == tmp1006
    tmp1110 = tmp1102 == tmp997
    tmp1111 = tmp1102 == tmp990
    tmp1114 = tl.where(tmp1111, tmp14, tmp1113)
    tmp1115 = tl.where(tmp1110, tmp14, tmp1114)
    tmp1116 = tl.where(tmp1109, tmp14, tmp1115)
    tmp1117 = tl.where(tmp1108, tmp14, tmp1116)
    tmp1118 = tl.where(tmp1107, tmp14, tmp1117)
    tmp1119 = tl.where(tmp1106, tmp14, tmp1118)
    tmp1120 = tl.where(tmp1105, tmp14, tmp1119)
    tmp1121 = tl.where(tmp1104, tmp14, tmp1120)
    tmp1122 = tmp1121 * tmp5
    tmp1123 = tl.where(tmp1103, tmp1122, tmp7)
    tmp1124 = tmp1101 + tmp1123
    tmp1125 = tl.full([1], 20, tl.int32)
    tmp1126 = tmp0 == tmp1125
    tmp1129 = tl.where(tmp1126, tmp1128, tmp7)
    tmp1130 = tmp1124 + tmp1129
    tmp1131 = tl.full([1], 19, tl.int32)
    tmp1132 = tmp0 == tmp1131
    tmp1135 = tmp1134 * tmp5
    tmp1136 = tl.where(tmp1132, tmp1135, tmp7)
    tmp1137 = tmp1130 + tmp1136
    tmp1138 = tl.full([1], 18, tl.int32)
    tmp1139 = tmp0 == tmp1138
    tmp1140 = tmp1138 == tmp1131
    tmp1143 = tl.where(tmp1140, tmp14, tmp1142)
    tmp1144 = tmp1143 * tmp5
    tmp1145 = tl.where(tmp1139, tmp1144, tmp7)
    tmp1146 = tmp1137 + tmp1145
    tmp1147 = tl.full([1], 17, tl.int32)
    tmp1148 = tmp0 == tmp1147
    tmp1149 = tmp1147 == tmp1138
    tmp1150 = tmp1147 == tmp1131
    tmp1153 = tl.where(tmp1150, tmp14, tmp1152)
    tmp1154 = tl.where(tmp1149, tmp14, tmp1153)
    tmp1155 = tmp1154 * tmp5
    tmp1156 = tl.where(tmp1148, tmp1155, tmp7)
    tmp1157 = tmp1146 + tmp1156
    tmp1158 = tl.full([1], 16, tl.int32)
    tmp1159 = tmp0 == tmp1158
    tmp1160 = tmp1158 == tmp1147
    tmp1161 = tmp1158 == tmp1138
    tmp1162 = tmp1158 == tmp1131
    tmp1165 = tl.where(tmp1162, tmp14, tmp1164)
    tmp1166 = tl.where(tmp1161, tmp14, tmp1165)
    tmp1167 = tl.where(tmp1160, tmp14, tmp1166)
    tmp1168 = tmp1167 * tmp5
    tmp1169 = tl.where(tmp1159, tmp1168, tmp7)
    tmp1170 = tmp1157 + tmp1169
    tmp1171 = tl.full([1], 15, tl.int32)
    tmp1172 = tmp0 == tmp1171
    tmp1173 = tmp1171 == tmp1158
    tmp1174 = tmp1171 == tmp1147
    tmp1175 = tmp1171 == tmp1138
    tmp1176 = tmp1171 == tmp1131
    tmp1179 = tl.where(tmp1176, tmp14, tmp1178)
    tmp1180 = tl.where(tmp1175, tmp14, tmp1179)
    tmp1181 = tl.where(tmp1174, tmp14, tmp1180)
    tmp1182 = tl.where(tmp1173, tmp14, tmp1181)
    tmp1183 = tmp1182 * tmp5
    tmp1184 = tl.where(tmp1172, tmp1183, tmp7)
    tmp1185 = tmp1170 + tmp1184
    tmp1186 = tl.full([1], 14, tl.int32)
    tmp1187 = tmp0 == tmp1186
    tmp1188 = tmp1186 == tmp1171
    tmp1189 = tmp1186 == tmp1158
    tmp1190 = tmp1186 == tmp1147
    tmp1191 = tmp1186 == tmp1138
    tmp1192 = tmp1186 == tmp1131
    tmp1195 = tl.where(tmp1192, tmp14, tmp1194)
    tmp1196 = tl.where(tmp1191, tmp14, tmp1195)
    tmp1197 = tl.where(tmp1190, tmp14, tmp1196)
    tmp1198 = tl.where(tmp1189, tmp14, tmp1197)
    tmp1199 = tl.where(tmp1188, tmp14, tmp1198)
    tmp1200 = tmp1199 * tmp5
    tmp1201 = tl.where(tmp1187, tmp1200, tmp7)
    tmp1202 = tmp1185 + tmp1201
    tmp1203 = tl.full([1], 13, tl.int32)
    tmp1204 = tmp0 == tmp1203
    tmp1205 = tmp1203 == tmp1186
    tmp1206 = tmp1203 == tmp1171
    tmp1207 = tmp1203 == tmp1158
    tmp1208 = tmp1203 == tmp1147
    tmp1209 = tmp1203 == tmp1138
    tmp1210 = tmp1203 == tmp1131
    tmp1213 = tl.where(tmp1210, tmp14, tmp1212)
    tmp1214 = tl.where(tmp1209, tmp14, tmp1213)
    tmp1215 = tl.where(tmp1208, tmp14, tmp1214)
    tmp1216 = tl.where(tmp1207, tmp14, tmp1215)
    tmp1217 = tl.where(tmp1206, tmp14, tmp1216)
    tmp1218 = tl.where(tmp1205, tmp14, tmp1217)
    tmp1219 = tmp1218 * tmp5
    tmp1220 = tl.where(tmp1204, tmp1219, tmp7)
    tmp1221 = tmp1202 + tmp1220
    tmp1222 = tl.full([1], 12, tl.int32)
    tmp1223 = tmp0 == tmp1222
    tmp1224 = tmp1222 == tmp1203
    tmp1225 = tmp1222 == tmp1186
    tmp1226 = tmp1222 == tmp1171
    tmp1227 = tmp1222 == tmp1158
    tmp1228 = tmp1222 == tmp1147
    tmp1229 = tmp1222 == tmp1138
    tmp1230 = tmp1222 == tmp1131
    tmp1233 = tl.where(tmp1230, tmp14, tmp1232)
    tmp1234 = tl.where(tmp1229, tmp14, tmp1233)
    tmp1235 = tl.where(tmp1228, tmp14, tmp1234)
    tmp1236 = tl.where(tmp1227, tmp14, tmp1235)
    tmp1237 = tl.where(tmp1226, tmp14, tmp1236)
    tmp1238 = tl.where(tmp1225, tmp14, tmp1237)
    tmp1239 = tl.where(tmp1224, tmp14, tmp1238)
    tmp1240 = tmp1239 * tmp5
    tmp1241 = tl.where(tmp1223, tmp1240, tmp7)
    tmp1242 = tmp1221 + tmp1241
    tmp1243 = tl.full([1], 11, tl.int32)
    tmp1244 = tmp0 == tmp1243
    tmp1245 = tmp1243 == tmp1222
    tmp1246 = tmp1243 == tmp1203
    tmp1247 = tmp1243 == tmp1186
    tmp1248 = tmp1243 == tmp1171
    tmp1249 = tmp1243 == tmp1158
    tmp1250 = tmp1243 == tmp1147
    tmp1251 = tmp1243 == tmp1138
    tmp1252 = tmp1243 == tmp1131
    tmp1255 = tl.where(tmp1252, tmp14, tmp1254)
    tmp1256 = tl.where(tmp1251, tmp14, tmp1255)
    tmp1257 = tl.where(tmp1250, tmp14, tmp1256)
    tmp1258 = tl.where(tmp1249, tmp14, tmp1257)
    tmp1259 = tl.where(tmp1248, tmp14, tmp1258)
    tmp1260 = tl.where(tmp1247, tmp14, tmp1259)
    tmp1261 = tl.where(tmp1246, tmp14, tmp1260)
    tmp1262 = tl.where(tmp1245, tmp14, tmp1261)
    tmp1263 = tmp1262 * tmp5
    tmp1264 = tl.where(tmp1244, tmp1263, tmp7)
    tmp1265 = tmp1242 + tmp1264
    tmp1266 = tl.full([1], 10, tl.int32)
    tmp1267 = tmp0 == tmp1266
    tmp1270 = tl.where(tmp1267, tmp1269, tmp7)
    tmp1271 = tmp1265 + tmp1270
    tmp1272 = tl.full([1], 9, tl.int32)
    tmp1273 = tmp0 == tmp1272
    tmp1276 = tmp1275 * tmp5
    tmp1277 = tl.where(tmp1273, tmp1276, tmp7)
    tmp1278 = tmp1271 + tmp1277
    tmp1279 = tl.full([1], 8, tl.int32)
    tmp1280 = tmp0 == tmp1279
    tmp1281 = tmp1279 == tmp1272
    tmp1284 = tl.where(tmp1281, tmp14, tmp1283)
    tmp1285 = tmp1284 * tmp5
    tmp1286 = tl.where(tmp1280, tmp1285, tmp7)
    tmp1287 = tmp1278 + tmp1286
    tmp1288 = tl.full([1], 7, tl.int32)
    tmp1289 = tmp0 == tmp1288
    tmp1290 = tmp1288 == tmp1279
    tmp1291 = tmp1288 == tmp1272
    tmp1294 = tl.where(tmp1291, tmp14, tmp1293)
    tmp1295 = tl.where(tmp1290, tmp14, tmp1294)
    tmp1296 = tmp1295 * tmp5
    tmp1297 = tl.where(tmp1289, tmp1296, tmp7)
    tmp1298 = tmp1287 + tmp1297
    tmp1299 = tl.full([1], 6, tl.int32)
    tmp1300 = tmp0 == tmp1299
    tmp1301 = tmp1299 == tmp1288
    tmp1302 = tmp1299 == tmp1279
    tmp1303 = tmp1299 == tmp1272
    tmp1306 = tl.where(tmp1303, tmp14, tmp1305)
    tmp1307 = tl.where(tmp1302, tmp14, tmp1306)
    tmp1308 = tl.where(tmp1301, tmp14, tmp1307)
    tmp1309 = tmp1308 * tmp5
    tmp1310 = tl.where(tmp1300, tmp1309, tmp7)
    tmp1311 = tmp1298 + tmp1310
    tmp1312 = tl.full([1], 5, tl.int32)
    tmp1313 = tmp0 == tmp1312
    tmp1314 = tmp1312 == tmp1299
    tmp1315 = tmp1312 == tmp1288
    tmp1316 = tmp1312 == tmp1279
    tmp1317 = tmp1312 == tmp1272
    tmp1320 = tl.where(tmp1317, tmp14, tmp1319)
    tmp1321 = tl.where(tmp1316, tmp14, tmp1320)
    tmp1322 = tl.where(tmp1315, tmp14, tmp1321)
    tmp1323 = tl.where(tmp1314, tmp14, tmp1322)
    tmp1324 = tmp1323 * tmp5
    tmp1325 = tl.where(tmp1313, tmp1324, tmp7)
    tmp1326 = tmp1311 + tmp1325
    tmp1327 = tl.full([1], 4, tl.int32)
    tmp1328 = tmp0 == tmp1327
    tmp1329 = tmp1327 == tmp1312
    tmp1330 = tmp1327 == tmp1299
    tmp1331 = tmp1327 == tmp1288
    tmp1332 = tmp1327 == tmp1279
    tmp1333 = tmp1327 == tmp1272
    tmp1336 = tl.where(tmp1333, tmp14, tmp1335)
    tmp1337 = tl.where(tmp1332, tmp14, tmp1336)
    tmp1338 = tl.where(tmp1331, tmp14, tmp1337)
    tmp1339 = tl.where(tmp1330, tmp14, tmp1338)
    tmp1340 = tl.where(tmp1329, tmp14, tmp1339)
    tmp1341 = tmp1340 * tmp5
    tmp1342 = tl.where(tmp1328, tmp1341, tmp7)
    tmp1343 = tmp1326 + tmp1342
    tmp1344 = tl.full([1], 3, tl.int32)
    tmp1345 = tmp0 == tmp1344
    tmp1346 = tmp1344 == tmp1327
    tmp1347 = tmp1344 == tmp1312
    tmp1348 = tmp1344 == tmp1299
    tmp1349 = tmp1344 == tmp1288
    tmp1350 = tmp1344 == tmp1279
    tmp1351 = tmp1344 == tmp1272
    tmp1354 = tl.where(tmp1351, tmp14, tmp1353)
    tmp1355 = tl.where(tmp1350, tmp14, tmp1354)
    tmp1356 = tl.where(tmp1349, tmp14, tmp1355)
    tmp1357 = tl.where(tmp1348, tmp14, tmp1356)
    tmp1358 = tl.where(tmp1347, tmp14, tmp1357)
    tmp1359 = tl.where(tmp1346, tmp14, tmp1358)
    tmp1360 = tmp1359 * tmp5
    tmp1361 = tl.where(tmp1345, tmp1360, tmp7)
    tmp1362 = tmp1343 + tmp1361
    tmp1363 = tl.full([1], 2, tl.int32)
    tmp1364 = tmp0 == tmp1363
    tmp1365 = tmp1363 == tmp1344
    tmp1366 = tmp1363 == tmp1327
    tmp1367 = tmp1363 == tmp1312
    tmp1368 = tmp1363 == tmp1299
    tmp1369 = tmp1363 == tmp1288
    tmp1370 = tmp1363 == tmp1279
    tmp1371 = tmp1363 == tmp1272
    tmp1374 = tl.where(tmp1371, tmp14, tmp1373)
    tmp1375 = tl.where(tmp1370, tmp14, tmp1374)
    tmp1376 = tl.where(tmp1369, tmp14, tmp1375)
    tmp1377 = tl.where(tmp1368, tmp14, tmp1376)
    tmp1378 = tl.where(tmp1367, tmp14, tmp1377)
    tmp1379 = tl.where(tmp1366, tmp14, tmp1378)
    tmp1380 = tl.where(tmp1365, tmp14, tmp1379)
    tmp1381 = tmp1380 * tmp5
    tmp1382 = tl.where(tmp1364, tmp1381, tmp7)
    tmp1383 = tmp1362 + tmp1382
    tmp1384 = tl.full([1], 1, tl.int32)
    tmp1385 = tmp0 == tmp1384
    tmp1386 = tmp1384 == tmp1363
    tmp1387 = tmp1384 == tmp1344
    tmp1388 = tmp1384 == tmp1327
    tmp1389 = tmp1384 == tmp1312
    tmp1390 = tmp1384 == tmp1299
    tmp1391 = tmp1384 == tmp1288
    tmp1392 = tmp1384 == tmp1279
    tmp1393 = tmp1384 == tmp1272
    tmp1396 = tl.where(tmp1393, tmp14, tmp1395)
    tmp1397 = tl.where(tmp1392, tmp14, tmp1396)
    tmp1398 = tl.where(tmp1391, tmp14, tmp1397)
    tmp1399 = tl.where(tmp1390, tmp14, tmp1398)
    tmp1400 = tl.where(tmp1389, tmp14, tmp1399)
    tmp1401 = tl.where(tmp1388, tmp14, tmp1400)
    tmp1402 = tl.where(tmp1387, tmp14, tmp1401)
    tmp1403 = tl.where(tmp1386, tmp14, tmp1402)
    tmp1404 = tmp1403 * tmp5
    tmp1405 = tl.where(tmp1385, tmp1404, tmp7)
    tmp1406 = tmp1383 + tmp1405
    tmp1407 = tl.full([1], 0, tl.int32)
    tmp1408 = tmp0 == tmp1407
    tmp1411 = tl.where(tmp1408, tmp1410, tmp7)
    tmp1412 = tmp1406 + tmp1411
    tl.store(in_out_ptr1 + (x0), tmp1412, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    full_default, tangents_1 = args
    args.clear()
    assert_size_stride(full_default, (100, ), (1, ))
    assert_size_stride(tangents_1, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf11 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf18 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf25 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf32 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf39 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf46 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf53 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf60 = empty_strided_cuda((100, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_0.run(tangents_1, buf4, buf11, buf18, buf25, buf32, buf39, buf46, buf53, buf60, 100, grid=grid(100), stream=stream0)
        buf12 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_1.run(buf4, buf12, 1, grid=grid(1), stream=stream0)
        buf19 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_2.run(buf11, buf19, 1, grid=grid(1), stream=stream0)
        buf26 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_3.run(buf18, buf26, 1, grid=grid(1), stream=stream0)
        buf33 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_4.run(buf25, buf33, 1, grid=grid(1), stream=stream0)
        buf40 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_5.run(buf32, buf40, 1, grid=grid(1), stream=stream0)
        buf47 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_6.run(buf39, buf47, 1, grid=grid(1), stream=stream0)
        buf5 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_7.run(tangents_1, buf5, 1, grid=grid(1), stream=stream0)
        buf54 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_8.run(buf46, buf54, 1, grid=grid(1), stream=stream0)
        buf61 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_9.run(buf53, buf61, 1, grid=grid(1), stream=stream0)
        buf67 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_10.run(buf60, buf67, 1, grid=grid(1), stream=stream0)
        buf10 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf13 = buf10; del buf10  # reuse
        buf14 = buf13; del buf13  # reuse
        buf15 = buf14; del buf14  # reuse
        buf16 = buf15; del buf15  # reuse
        buf20 = buf16; del buf16  # reuse
        buf21 = buf20; del buf20  # reuse
        buf22 = buf21; del buf21  # reuse
        buf23 = buf22; del buf22  # reuse
        buf27 = buf23; del buf23  # reuse
        buf28 = buf27; del buf27  # reuse
        buf29 = buf28; del buf28  # reuse
        buf30 = buf29; del buf29  # reuse
        buf34 = buf30; del buf30  # reuse
        buf35 = buf34; del buf34  # reuse
        buf36 = buf35; del buf35  # reuse
        buf37 = buf36; del buf36  # reuse
        buf41 = buf37; del buf37  # reuse
        buf42 = buf41; del buf41  # reuse
        buf43 = buf42; del buf42  # reuse
        buf44 = buf43; del buf43  # reuse
        buf48 = buf44; del buf44  # reuse
        buf49 = buf48; del buf48  # reuse
        buf50 = buf49; del buf49  # reuse
        buf51 = buf50; del buf50  # reuse
        buf55 = buf51; del buf51  # reuse
        buf56 = buf55; del buf55  # reuse
        buf57 = buf56; del buf56  # reuse
        buf58 = buf57; del buf57  # reuse
        buf62 = buf58; del buf58  # reuse
        buf63 = buf62; del buf62  # reuse
        buf64 = buf63; del buf63  # reuse
        buf65 = buf64; del buf64  # reuse
        buf68 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_11.run(buf68, tangents_1, full_default, buf5, buf4, buf12, buf11, buf19, buf18, buf26, buf25, buf33, buf32, buf40, buf39, buf47, buf46, buf54, buf53, buf61, buf60, buf67, 100, grid=grid(100), stream=stream0)
        del buf11
        del buf12
        del buf18
        del buf19
        del buf25
        del buf26
        del buf32
        del buf33
        del buf39
        del buf4
        del buf40
        del buf46
        del buf47
        del buf5
        del buf53
        del buf54
        del buf60
        del buf61
        del buf67
        del full_default
        del tangents_1
    return (buf68, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    full_default = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([full_default, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
