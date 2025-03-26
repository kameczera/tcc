# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_kamei/xa/cxaoaqkrnnb7xqjkol3mnvnoopdjwb2uyipbfrldqguakss3lmyo.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   out => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([100], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/ru/cruxtqyz6ys4imhrctf4vko6a66xbsn32typmrpue4ao7bavrwkc.py
# Topologically Sorted Source Nodes: [mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, mul_58, mul_59, mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, mul_88, mul_89, mul_90, mul_91, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, mul_98, mul_99], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_4 => mul_4
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_5 => mul_5
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   mul_59 => mul_59
#   mul_6 => mul_6
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_62
#   mul_63 => mul_63
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_7 => mul_7
#   mul_70 => mul_70
#   mul_71 => mul_71
#   mul_72 => mul_72
#   mul_73 => mul_73
#   mul_74 => mul_74
#   mul_75 => mul_75
#   mul_76 => mul_76
#   mul_77 => mul_77
#   mul_78 => mul_78
#   mul_79 => mul_79
#   mul_8 => mul_8
#   mul_80 => mul_80
#   mul_81 => mul_81
#   mul_82 => mul_82
#   mul_83 => mul_83
#   mul_84 => mul_84
#   mul_85 => mul_85
#   mul_86 => mul_86
#   mul_87 => mul_87
#   mul_88 => mul_88
#   mul_89 => mul_89
#   mul_9 => mul_9
#   mul_90 => mul_90
#   mul_91 => mul_91
#   mul_92 => mul_92
#   mul_93 => mul_93
#   mul_94 => mul_94
#   mul_95 => mul_95
#   mul_96 => mul_96
#   mul_97 => mul_97
#   mul_98 => mul_98
#   mul_99 => mul_99
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 2), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mul, 0, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, 2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %mul_1, 0, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, 2), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %mul_2, 0, 2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %mul_3, 0, 3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, 2), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %mul_4, 0, 4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_24, 2), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %mul_5, 0, 5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_29, 2), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_5, %mul_6, 0, 6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_34, 2), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_6, %mul_7, 0, 7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_39, 2), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_7, %mul_8, 0, 8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_44, 2), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %mul_9, 0, 9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_49, 2), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_9, %mul_10, 0, 10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_54, 2), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_10, %mul_11, 0, 11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_59, 2), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_11, %mul_12, 0, 12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_64, 2), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %mul_13, 0, 13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_69, 2), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_13, %mul_14, 0, 14), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_74, 2), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_14, %mul_15, 0, 15), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_79, 2), kwargs = {})
#   %select_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_15, %mul_16, 0, 16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_84, 2), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_16, %mul_17, 0, 17), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_89, 2), kwargs = {})
#   %select_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_17, %mul_18, 0, 18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_94, 2), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_18, %mul_19, 0, 19), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_99, 2), kwargs = {})
#   %select_scatter_default_20 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_19, %mul_20, 0, 20), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_104, 2), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_20, %mul_21, 0, 21), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_109, 2), kwargs = {})
#   %select_scatter_default_22 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_21, %mul_22, 0, 22), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_114, 2), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_22, %mul_23, 0, 23), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_119, 2), kwargs = {})
#   %select_scatter_default_24 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_23, %mul_24, 0, 24), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_124, 2), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_24, %mul_25, 0, 25), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_129, 2), kwargs = {})
#   %select_scatter_default_26 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_25, %mul_26, 0, 26), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_134, 2), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_26, %mul_27, 0, 27), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_139, 2), kwargs = {})
#   %select_scatter_default_28 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_27, %mul_28, 0, 28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_144, 2), kwargs = {})
#   %select_scatter_default_29 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_28, %mul_29, 0, 29), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_149, 2), kwargs = {})
#   %select_scatter_default_30 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_29, %mul_30, 0, 30), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_154, 2), kwargs = {})
#   %select_scatter_default_31 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_30, %mul_31, 0, 31), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_159, 2), kwargs = {})
#   %select_scatter_default_32 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_31, %mul_32, 0, 32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_164, 2), kwargs = {})
#   %select_scatter_default_33 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_32, %mul_33, 0, 33), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_169, 2), kwargs = {})
#   %select_scatter_default_34 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_33, %mul_34, 0, 34), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_174, 2), kwargs = {})
#   %select_scatter_default_35 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_34, %mul_35, 0, 35), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_179, 2), kwargs = {})
#   %select_scatter_default_36 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_35, %mul_36, 0, 36), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_184, 2), kwargs = {})
#   %select_scatter_default_37 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_36, %mul_37, 0, 37), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_189, 2), kwargs = {})
#   %select_scatter_default_38 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_37, %mul_38, 0, 38), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_194, 2), kwargs = {})
#   %select_scatter_default_39 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_38, %mul_39, 0, 39), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_199, 2), kwargs = {})
#   %select_scatter_default_40 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_39, %mul_40, 0, 40), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_204, 2), kwargs = {})
#   %select_scatter_default_41 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_40, %mul_41, 0, 41), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_209, 2), kwargs = {})
#   %select_scatter_default_42 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_41, %mul_42, 0, 42), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_214, 2), kwargs = {})
#   %select_scatter_default_43 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_42, %mul_43, 0, 43), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_219, 2), kwargs = {})
#   %select_scatter_default_44 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_43, %mul_44, 0, 44), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_224, 2), kwargs = {})
#   %select_scatter_default_45 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_44, %mul_45, 0, 45), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_229, 2), kwargs = {})
#   %select_scatter_default_46 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_45, %mul_46, 0, 46), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_234, 2), kwargs = {})
#   %select_scatter_default_47 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_46, %mul_47, 0, 47), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_239, 2), kwargs = {})
#   %select_scatter_default_48 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_47, %mul_48, 0, 48), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_244, 2), kwargs = {})
#   %select_scatter_default_49 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_48, %mul_49, 0, 49), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_249, 2), kwargs = {})
#   %select_scatter_default_50 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_49, %mul_50, 0, 50), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_254, 2), kwargs = {})
#   %select_scatter_default_51 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_50, %mul_51, 0, 51), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_259, 2), kwargs = {})
#   %select_scatter_default_52 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_51, %mul_52, 0, 52), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_264, 2), kwargs = {})
#   %select_scatter_default_53 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_52, %mul_53, 0, 53), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_269, 2), kwargs = {})
#   %select_scatter_default_54 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_53, %mul_54, 0, 54), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_274, 2), kwargs = {})
#   %select_scatter_default_55 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_54, %mul_55, 0, 55), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_279, 2), kwargs = {})
#   %select_scatter_default_56 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_55, %mul_56, 0, 56), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_284, 2), kwargs = {})
#   %select_scatter_default_57 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_56, %mul_57, 0, 57), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_289, 2), kwargs = {})
#   %select_scatter_default_58 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_57, %mul_58, 0, 58), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_294, 2), kwargs = {})
#   %select_scatter_default_59 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_58, %mul_59, 0, 59), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_299, 2), kwargs = {})
#   %select_scatter_default_60 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_59, %mul_60, 0, 60), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_304, 2), kwargs = {})
#   %select_scatter_default_61 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_60, %mul_61, 0, 61), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_309, 2), kwargs = {})
#   %select_scatter_default_62 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_61, %mul_62, 0, 62), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_314, 2), kwargs = {})
#   %select_scatter_default_63 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_62, %mul_63, 0, 63), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_319, 2), kwargs = {})
#   %select_scatter_default_64 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_63, %mul_64, 0, 64), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_324, 2), kwargs = {})
#   %select_scatter_default_65 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_64, %mul_65, 0, 65), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_329, 2), kwargs = {})
#   %select_scatter_default_66 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_65, %mul_66, 0, 66), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_334, 2), kwargs = {})
#   %select_scatter_default_67 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_66, %mul_67, 0, 67), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_339, 2), kwargs = {})
#   %select_scatter_default_68 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_67, %mul_68, 0, 68), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_344, 2), kwargs = {})
#   %select_scatter_default_69 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_68, %mul_69, 0, 69), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_349, 2), kwargs = {})
#   %select_scatter_default_70 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_69, %mul_70, 0, 70), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_354, 2), kwargs = {})
#   %select_scatter_default_71 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_70, %mul_71, 0, 71), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_359, 2), kwargs = {})
#   %select_scatter_default_72 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_71, %mul_72, 0, 72), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_364, 2), kwargs = {})
#   %select_scatter_default_73 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_72, %mul_73, 0, 73), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_369, 2), kwargs = {})
#   %select_scatter_default_74 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_73, %mul_74, 0, 74), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_374, 2), kwargs = {})
#   %select_scatter_default_75 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_74, %mul_75, 0, 75), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_379, 2), kwargs = {})
#   %select_scatter_default_76 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_75, %mul_76, 0, 76), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_384, 2), kwargs = {})
#   %select_scatter_default_77 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_76, %mul_77, 0, 77), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_389, 2), kwargs = {})
#   %select_scatter_default_78 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_77, %mul_78, 0, 78), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_394, 2), kwargs = {})
#   %select_scatter_default_79 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_78, %mul_79, 0, 79), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_399, 2), kwargs = {})
#   %select_scatter_default_80 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_79, %mul_80, 0, 80), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_404, 2), kwargs = {})
#   %select_scatter_default_81 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_80, %mul_81, 0, 81), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_409, 2), kwargs = {})
#   %select_scatter_default_82 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_81, %mul_82, 0, 82), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_414, 2), kwargs = {})
#   %select_scatter_default_83 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_82, %mul_83, 0, 83), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_419, 2), kwargs = {})
#   %select_scatter_default_84 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_83, %mul_84, 0, 84), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_424, 2), kwargs = {})
#   %select_scatter_default_85 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_84, %mul_85, 0, 85), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_429, 2), kwargs = {})
#   %select_scatter_default_86 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_85, %mul_86, 0, 86), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_434, 2), kwargs = {})
#   %select_scatter_default_87 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_86, %mul_87, 0, 87), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_439, 2), kwargs = {})
#   %select_scatter_default_88 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_87, %mul_88, 0, 88), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_444, 2), kwargs = {})
#   %select_scatter_default_89 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_88, %mul_89, 0, 89), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_449, 2), kwargs = {})
#   %select_scatter_default_90 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_89, %mul_90, 0, 90), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_454, 2), kwargs = {})
#   %select_scatter_default_91 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_90, %mul_91, 0, 91), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_459, 2), kwargs = {})
#   %select_scatter_default_92 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_91, %mul_92, 0, 92), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_464, 2), kwargs = {})
#   %select_scatter_default_93 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_92, %mul_93, 0, 93), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_469, 2), kwargs = {})
#   %select_scatter_default_94 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_93, %mul_94, 0, 94), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_474, 2), kwargs = {})
#   %select_scatter_default_95 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_94, %mul_95, 0, 95), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_479, 2), kwargs = {})
#   %select_scatter_default_96 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_95, %mul_96, 0, 96), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_484, 2), kwargs = {})
#   %select_scatter_default_97 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_96, %mul_97, 0, 97), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_489, 2), kwargs = {})
#   %select_scatter_default_98 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_97, %mul_98, 0, 98), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_494, 2), kwargs = {})
#   %select_scatter_default_99 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_98, %mul_99, 0, 99), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 100, 'num_reduction': 0, 'backend_hash': '712B1D69F892A891D8FFA5075DCAB47CFF4E132D88BFC66744701CEAE226F127', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp3 = tl.load(in_ptr0 + (5))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (4))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (3))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp19 = tl.load(in_ptr0 + (2))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp24 = tl.load(in_ptr0 + (1))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (11))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (10))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp51 = tl.load(in_ptr0 + (9))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp56 = tl.load(in_ptr0 + (8))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp61 = tl.load(in_ptr0 + (7))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp66 = tl.load(in_ptr0 + (6))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp77 = tl.load(in_ptr0 + (17))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp82 = tl.load(in_ptr0 + (16))
    tmp83 = tl.broadcast_to(tmp82, [XBLOCK])
    tmp87 = tl.load(in_ptr0 + (15))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp92 = tl.load(in_ptr0 + (14))
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK])
    tmp97 = tl.load(in_ptr0 + (13))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp102 = tl.load(in_ptr0 + (12))
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK])
    tmp113 = tl.load(in_ptr0 + (23))
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK])
    tmp118 = tl.load(in_ptr0 + (22))
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK])
    tmp123 = tl.load(in_ptr0 + (21))
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK])
    tmp128 = tl.load(in_ptr0 + (20))
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK])
    tmp133 = tl.load(in_ptr0 + (19))
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK])
    tmp138 = tl.load(in_ptr0 + (18))
    tmp139 = tl.broadcast_to(tmp138, [XBLOCK])
    tmp149 = tl.load(in_ptr0 + (29))
    tmp150 = tl.broadcast_to(tmp149, [XBLOCK])
    tmp154 = tl.load(in_ptr0 + (28))
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK])
    tmp159 = tl.load(in_ptr0 + (27))
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK])
    tmp164 = tl.load(in_ptr0 + (26))
    tmp165 = tl.broadcast_to(tmp164, [XBLOCK])
    tmp169 = tl.load(in_ptr0 + (25))
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK])
    tmp174 = tl.load(in_ptr0 + (24))
    tmp175 = tl.broadcast_to(tmp174, [XBLOCK])
    tmp185 = tl.load(in_ptr0 + (35))
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK])
    tmp190 = tl.load(in_ptr0 + (34))
    tmp191 = tl.broadcast_to(tmp190, [XBLOCK])
    tmp195 = tl.load(in_ptr0 + (33))
    tmp196 = tl.broadcast_to(tmp195, [XBLOCK])
    tmp200 = tl.load(in_ptr0 + (32))
    tmp201 = tl.broadcast_to(tmp200, [XBLOCK])
    tmp205 = tl.load(in_ptr0 + (31))
    tmp206 = tl.broadcast_to(tmp205, [XBLOCK])
    tmp210 = tl.load(in_ptr0 + (30))
    tmp211 = tl.broadcast_to(tmp210, [XBLOCK])
    tmp221 = tl.load(in_ptr0 + (41))
    tmp222 = tl.broadcast_to(tmp221, [XBLOCK])
    tmp226 = tl.load(in_ptr0 + (40))
    tmp227 = tl.broadcast_to(tmp226, [XBLOCK])
    tmp231 = tl.load(in_ptr0 + (39))
    tmp232 = tl.broadcast_to(tmp231, [XBLOCK])
    tmp236 = tl.load(in_ptr0 + (38))
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK])
    tmp241 = tl.load(in_ptr0 + (37))
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK])
    tmp246 = tl.load(in_ptr0 + (36))
    tmp247 = tl.broadcast_to(tmp246, [XBLOCK])
    tmp257 = tl.load(in_ptr0 + (47))
    tmp258 = tl.broadcast_to(tmp257, [XBLOCK])
    tmp262 = tl.load(in_ptr0 + (46))
    tmp263 = tl.broadcast_to(tmp262, [XBLOCK])
    tmp267 = tl.load(in_ptr0 + (45))
    tmp268 = tl.broadcast_to(tmp267, [XBLOCK])
    tmp272 = tl.load(in_ptr0 + (44))
    tmp273 = tl.broadcast_to(tmp272, [XBLOCK])
    tmp277 = tl.load(in_ptr0 + (43))
    tmp278 = tl.broadcast_to(tmp277, [XBLOCK])
    tmp282 = tl.load(in_ptr0 + (42))
    tmp283 = tl.broadcast_to(tmp282, [XBLOCK])
    tmp293 = tl.load(in_ptr0 + (53))
    tmp294 = tl.broadcast_to(tmp293, [XBLOCK])
    tmp298 = tl.load(in_ptr0 + (52))
    tmp299 = tl.broadcast_to(tmp298, [XBLOCK])
    tmp303 = tl.load(in_ptr0 + (51))
    tmp304 = tl.broadcast_to(tmp303, [XBLOCK])
    tmp308 = tl.load(in_ptr0 + (50))
    tmp309 = tl.broadcast_to(tmp308, [XBLOCK])
    tmp313 = tl.load(in_ptr0 + (49))
    tmp314 = tl.broadcast_to(tmp313, [XBLOCK])
    tmp318 = tl.load(in_ptr0 + (48))
    tmp319 = tl.broadcast_to(tmp318, [XBLOCK])
    tmp329 = tl.load(in_ptr0 + (59))
    tmp330 = tl.broadcast_to(tmp329, [XBLOCK])
    tmp334 = tl.load(in_ptr0 + (58))
    tmp335 = tl.broadcast_to(tmp334, [XBLOCK])
    tmp339 = tl.load(in_ptr0 + (57))
    tmp340 = tl.broadcast_to(tmp339, [XBLOCK])
    tmp344 = tl.load(in_ptr0 + (56))
    tmp345 = tl.broadcast_to(tmp344, [XBLOCK])
    tmp349 = tl.load(in_ptr0 + (55))
    tmp350 = tl.broadcast_to(tmp349, [XBLOCK])
    tmp354 = tl.load(in_ptr0 + (54))
    tmp355 = tl.broadcast_to(tmp354, [XBLOCK])
    tmp365 = tl.load(in_ptr0 + (65))
    tmp366 = tl.broadcast_to(tmp365, [XBLOCK])
    tmp370 = tl.load(in_ptr0 + (64))
    tmp371 = tl.broadcast_to(tmp370, [XBLOCK])
    tmp375 = tl.load(in_ptr0 + (63))
    tmp376 = tl.broadcast_to(tmp375, [XBLOCK])
    tmp380 = tl.load(in_ptr0 + (62))
    tmp381 = tl.broadcast_to(tmp380, [XBLOCK])
    tmp385 = tl.load(in_ptr0 + (61))
    tmp386 = tl.broadcast_to(tmp385, [XBLOCK])
    tmp390 = tl.load(in_ptr0 + (60))
    tmp391 = tl.broadcast_to(tmp390, [XBLOCK])
    tmp401 = tl.load(in_ptr0 + (71))
    tmp402 = tl.broadcast_to(tmp401, [XBLOCK])
    tmp406 = tl.load(in_ptr0 + (70))
    tmp407 = tl.broadcast_to(tmp406, [XBLOCK])
    tmp411 = tl.load(in_ptr0 + (69))
    tmp412 = tl.broadcast_to(tmp411, [XBLOCK])
    tmp416 = tl.load(in_ptr0 + (68))
    tmp417 = tl.broadcast_to(tmp416, [XBLOCK])
    tmp421 = tl.load(in_ptr0 + (67))
    tmp422 = tl.broadcast_to(tmp421, [XBLOCK])
    tmp426 = tl.load(in_ptr0 + (66))
    tmp427 = tl.broadcast_to(tmp426, [XBLOCK])
    tmp437 = tl.load(in_ptr0 + (77))
    tmp438 = tl.broadcast_to(tmp437, [XBLOCK])
    tmp442 = tl.load(in_ptr0 + (76))
    tmp443 = tl.broadcast_to(tmp442, [XBLOCK])
    tmp447 = tl.load(in_ptr0 + (75))
    tmp448 = tl.broadcast_to(tmp447, [XBLOCK])
    tmp452 = tl.load(in_ptr0 + (74))
    tmp453 = tl.broadcast_to(tmp452, [XBLOCK])
    tmp457 = tl.load(in_ptr0 + (73))
    tmp458 = tl.broadcast_to(tmp457, [XBLOCK])
    tmp462 = tl.load(in_ptr0 + (72))
    tmp463 = tl.broadcast_to(tmp462, [XBLOCK])
    tmp473 = tl.load(in_ptr0 + (83))
    tmp474 = tl.broadcast_to(tmp473, [XBLOCK])
    tmp478 = tl.load(in_ptr0 + (82))
    tmp479 = tl.broadcast_to(tmp478, [XBLOCK])
    tmp483 = tl.load(in_ptr0 + (81))
    tmp484 = tl.broadcast_to(tmp483, [XBLOCK])
    tmp488 = tl.load(in_ptr0 + (80))
    tmp489 = tl.broadcast_to(tmp488, [XBLOCK])
    tmp493 = tl.load(in_ptr0 + (79))
    tmp494 = tl.broadcast_to(tmp493, [XBLOCK])
    tmp498 = tl.load(in_ptr0 + (78))
    tmp499 = tl.broadcast_to(tmp498, [XBLOCK])
    tmp509 = tl.load(in_ptr0 + (89))
    tmp510 = tl.broadcast_to(tmp509, [XBLOCK])
    tmp514 = tl.load(in_ptr0 + (88))
    tmp515 = tl.broadcast_to(tmp514, [XBLOCK])
    tmp519 = tl.load(in_ptr0 + (87))
    tmp520 = tl.broadcast_to(tmp519, [XBLOCK])
    tmp524 = tl.load(in_ptr0 + (86))
    tmp525 = tl.broadcast_to(tmp524, [XBLOCK])
    tmp529 = tl.load(in_ptr0 + (85))
    tmp530 = tl.broadcast_to(tmp529, [XBLOCK])
    tmp534 = tl.load(in_ptr0 + (84))
    tmp535 = tl.broadcast_to(tmp534, [XBLOCK])
    tmp545 = tl.load(in_ptr0 + (95))
    tmp546 = tl.broadcast_to(tmp545, [XBLOCK])
    tmp550 = tl.load(in_ptr0 + (94))
    tmp551 = tl.broadcast_to(tmp550, [XBLOCK])
    tmp555 = tl.load(in_ptr0 + (93))
    tmp556 = tl.broadcast_to(tmp555, [XBLOCK])
    tmp560 = tl.load(in_ptr0 + (92))
    tmp561 = tl.broadcast_to(tmp560, [XBLOCK])
    tmp565 = tl.load(in_ptr0 + (91))
    tmp566 = tl.broadcast_to(tmp565, [XBLOCK])
    tmp570 = tl.load(in_ptr0 + (90))
    tmp571 = tl.broadcast_to(tmp570, [XBLOCK])
    tmp581 = tl.load(in_ptr0 + (99))
    tmp582 = tl.broadcast_to(tmp581, [XBLOCK])
    tmp586 = tl.load(in_ptr0 + (98))
    tmp587 = tl.broadcast_to(tmp586, [XBLOCK])
    tmp591 = tl.load(in_ptr0 + (97))
    tmp592 = tl.broadcast_to(tmp591, [XBLOCK])
    tmp596 = tl.load(in_ptr0 + (96))
    tmp597 = tl.broadcast_to(tmp596, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = 2.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 4, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp11 = tmp10 * tmp5
    tmp12 = tl.full([1], 3, tl.int32)
    tmp13 = tmp0 == tmp12
    tmp16 = tmp15 * tmp5
    tmp17 = tl.full([1], 2, tl.int32)
    tmp18 = tmp0 == tmp17
    tmp21 = tmp20 * tmp5
    tmp22 = tl.full([1], 1, tl.int32)
    tmp23 = tmp0 == tmp22
    tmp26 = tmp25 * tmp5
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = tmp0 == tmp27
    tmp31 = tmp30 * tmp5
    tmp32 = 0.0
    tmp33 = tl.where(tmp28, tmp31, tmp32)
    tmp34 = tl.where(tmp23, tmp26, tmp33)
    tmp35 = tl.where(tmp18, tmp21, tmp34)
    tmp36 = tl.where(tmp13, tmp16, tmp35)
    tmp37 = tl.where(tmp8, tmp11, tmp36)
    tmp38 = tl.where(tmp2, tmp6, tmp37)
    tmp39 = tl.full([1], 11, tl.int32)
    tmp40 = tmp0 == tmp39
    tmp43 = tmp42 * tmp5
    tmp44 = tl.full([1], 10, tl.int32)
    tmp45 = tmp0 == tmp44
    tmp48 = tmp47 * tmp5
    tmp49 = tl.full([1], 9, tl.int32)
    tmp50 = tmp0 == tmp49
    tmp53 = tmp52 * tmp5
    tmp54 = tl.full([1], 8, tl.int32)
    tmp55 = tmp0 == tmp54
    tmp58 = tmp57 * tmp5
    tmp59 = tl.full([1], 7, tl.int32)
    tmp60 = tmp0 == tmp59
    tmp63 = tmp62 * tmp5
    tmp64 = tl.full([1], 6, tl.int32)
    tmp65 = tmp0 == tmp64
    tmp68 = tmp67 * tmp5
    tmp69 = tl.where(tmp65, tmp68, tmp38)
    tmp70 = tl.where(tmp60, tmp63, tmp69)
    tmp71 = tl.where(tmp55, tmp58, tmp70)
    tmp72 = tl.where(tmp50, tmp53, tmp71)
    tmp73 = tl.where(tmp45, tmp48, tmp72)
    tmp74 = tl.where(tmp40, tmp43, tmp73)
    tmp75 = tl.full([1], 17, tl.int32)
    tmp76 = tmp0 == tmp75
    tmp79 = tmp78 * tmp5
    tmp80 = tl.full([1], 16, tl.int32)
    tmp81 = tmp0 == tmp80
    tmp84 = tmp83 * tmp5
    tmp85 = tl.full([1], 15, tl.int32)
    tmp86 = tmp0 == tmp85
    tmp89 = tmp88 * tmp5
    tmp90 = tl.full([1], 14, tl.int32)
    tmp91 = tmp0 == tmp90
    tmp94 = tmp93 * tmp5
    tmp95 = tl.full([1], 13, tl.int32)
    tmp96 = tmp0 == tmp95
    tmp99 = tmp98 * tmp5
    tmp100 = tl.full([1], 12, tl.int32)
    tmp101 = tmp0 == tmp100
    tmp104 = tmp103 * tmp5
    tmp105 = tl.where(tmp101, tmp104, tmp74)
    tmp106 = tl.where(tmp96, tmp99, tmp105)
    tmp107 = tl.where(tmp91, tmp94, tmp106)
    tmp108 = tl.where(tmp86, tmp89, tmp107)
    tmp109 = tl.where(tmp81, tmp84, tmp108)
    tmp110 = tl.where(tmp76, tmp79, tmp109)
    tmp111 = tl.full([1], 23, tl.int32)
    tmp112 = tmp0 == tmp111
    tmp115 = tmp114 * tmp5
    tmp116 = tl.full([1], 22, tl.int32)
    tmp117 = tmp0 == tmp116
    tmp120 = tmp119 * tmp5
    tmp121 = tl.full([1], 21, tl.int32)
    tmp122 = tmp0 == tmp121
    tmp125 = tmp124 * tmp5
    tmp126 = tl.full([1], 20, tl.int32)
    tmp127 = tmp0 == tmp126
    tmp130 = tmp129 * tmp5
    tmp131 = tl.full([1], 19, tl.int32)
    tmp132 = tmp0 == tmp131
    tmp135 = tmp134 * tmp5
    tmp136 = tl.full([1], 18, tl.int32)
    tmp137 = tmp0 == tmp136
    tmp140 = tmp139 * tmp5
    tmp141 = tl.where(tmp137, tmp140, tmp110)
    tmp142 = tl.where(tmp132, tmp135, tmp141)
    tmp143 = tl.where(tmp127, tmp130, tmp142)
    tmp144 = tl.where(tmp122, tmp125, tmp143)
    tmp145 = tl.where(tmp117, tmp120, tmp144)
    tmp146 = tl.where(tmp112, tmp115, tmp145)
    tmp147 = tl.full([1], 29, tl.int32)
    tmp148 = tmp0 == tmp147
    tmp151 = tmp150 * tmp5
    tmp152 = tl.full([1], 28, tl.int32)
    tmp153 = tmp0 == tmp152
    tmp156 = tmp155 * tmp5
    tmp157 = tl.full([1], 27, tl.int32)
    tmp158 = tmp0 == tmp157
    tmp161 = tmp160 * tmp5
    tmp162 = tl.full([1], 26, tl.int32)
    tmp163 = tmp0 == tmp162
    tmp166 = tmp165 * tmp5
    tmp167 = tl.full([1], 25, tl.int32)
    tmp168 = tmp0 == tmp167
    tmp171 = tmp170 * tmp5
    tmp172 = tl.full([1], 24, tl.int32)
    tmp173 = tmp0 == tmp172
    tmp176 = tmp175 * tmp5
    tmp177 = tl.where(tmp173, tmp176, tmp146)
    tmp178 = tl.where(tmp168, tmp171, tmp177)
    tmp179 = tl.where(tmp163, tmp166, tmp178)
    tmp180 = tl.where(tmp158, tmp161, tmp179)
    tmp181 = tl.where(tmp153, tmp156, tmp180)
    tmp182 = tl.where(tmp148, tmp151, tmp181)
    tmp183 = tl.full([1], 35, tl.int32)
    tmp184 = tmp0 == tmp183
    tmp187 = tmp186 * tmp5
    tmp188 = tl.full([1], 34, tl.int32)
    tmp189 = tmp0 == tmp188
    tmp192 = tmp191 * tmp5
    tmp193 = tl.full([1], 33, tl.int32)
    tmp194 = tmp0 == tmp193
    tmp197 = tmp196 * tmp5
    tmp198 = tl.full([1], 32, tl.int32)
    tmp199 = tmp0 == tmp198
    tmp202 = tmp201 * tmp5
    tmp203 = tl.full([1], 31, tl.int32)
    tmp204 = tmp0 == tmp203
    tmp207 = tmp206 * tmp5
    tmp208 = tl.full([1], 30, tl.int32)
    tmp209 = tmp0 == tmp208
    tmp212 = tmp211 * tmp5
    tmp213 = tl.where(tmp209, tmp212, tmp182)
    tmp214 = tl.where(tmp204, tmp207, tmp213)
    tmp215 = tl.where(tmp199, tmp202, tmp214)
    tmp216 = tl.where(tmp194, tmp197, tmp215)
    tmp217 = tl.where(tmp189, tmp192, tmp216)
    tmp218 = tl.where(tmp184, tmp187, tmp217)
    tmp219 = tl.full([1], 41, tl.int32)
    tmp220 = tmp0 == tmp219
    tmp223 = tmp222 * tmp5
    tmp224 = tl.full([1], 40, tl.int32)
    tmp225 = tmp0 == tmp224
    tmp228 = tmp227 * tmp5
    tmp229 = tl.full([1], 39, tl.int32)
    tmp230 = tmp0 == tmp229
    tmp233 = tmp232 * tmp5
    tmp234 = tl.full([1], 38, tl.int32)
    tmp235 = tmp0 == tmp234
    tmp238 = tmp237 * tmp5
    tmp239 = tl.full([1], 37, tl.int32)
    tmp240 = tmp0 == tmp239
    tmp243 = tmp242 * tmp5
    tmp244 = tl.full([1], 36, tl.int32)
    tmp245 = tmp0 == tmp244
    tmp248 = tmp247 * tmp5
    tmp249 = tl.where(tmp245, tmp248, tmp218)
    tmp250 = tl.where(tmp240, tmp243, tmp249)
    tmp251 = tl.where(tmp235, tmp238, tmp250)
    tmp252 = tl.where(tmp230, tmp233, tmp251)
    tmp253 = tl.where(tmp225, tmp228, tmp252)
    tmp254 = tl.where(tmp220, tmp223, tmp253)
    tmp255 = tl.full([1], 47, tl.int32)
    tmp256 = tmp0 == tmp255
    tmp259 = tmp258 * tmp5
    tmp260 = tl.full([1], 46, tl.int32)
    tmp261 = tmp0 == tmp260
    tmp264 = tmp263 * tmp5
    tmp265 = tl.full([1], 45, tl.int32)
    tmp266 = tmp0 == tmp265
    tmp269 = tmp268 * tmp5
    tmp270 = tl.full([1], 44, tl.int32)
    tmp271 = tmp0 == tmp270
    tmp274 = tmp273 * tmp5
    tmp275 = tl.full([1], 43, tl.int32)
    tmp276 = tmp0 == tmp275
    tmp279 = tmp278 * tmp5
    tmp280 = tl.full([1], 42, tl.int32)
    tmp281 = tmp0 == tmp280
    tmp284 = tmp283 * tmp5
    tmp285 = tl.where(tmp281, tmp284, tmp254)
    tmp286 = tl.where(tmp276, tmp279, tmp285)
    tmp287 = tl.where(tmp271, tmp274, tmp286)
    tmp288 = tl.where(tmp266, tmp269, tmp287)
    tmp289 = tl.where(tmp261, tmp264, tmp288)
    tmp290 = tl.where(tmp256, tmp259, tmp289)
    tmp291 = tl.full([1], 53, tl.int32)
    tmp292 = tmp0 == tmp291
    tmp295 = tmp294 * tmp5
    tmp296 = tl.full([1], 52, tl.int32)
    tmp297 = tmp0 == tmp296
    tmp300 = tmp299 * tmp5
    tmp301 = tl.full([1], 51, tl.int32)
    tmp302 = tmp0 == tmp301
    tmp305 = tmp304 * tmp5
    tmp306 = tl.full([1], 50, tl.int32)
    tmp307 = tmp0 == tmp306
    tmp310 = tmp309 * tmp5
    tmp311 = tl.full([1], 49, tl.int32)
    tmp312 = tmp0 == tmp311
    tmp315 = tmp314 * tmp5
    tmp316 = tl.full([1], 48, tl.int32)
    tmp317 = tmp0 == tmp316
    tmp320 = tmp319 * tmp5
    tmp321 = tl.where(tmp317, tmp320, tmp290)
    tmp322 = tl.where(tmp312, tmp315, tmp321)
    tmp323 = tl.where(tmp307, tmp310, tmp322)
    tmp324 = tl.where(tmp302, tmp305, tmp323)
    tmp325 = tl.where(tmp297, tmp300, tmp324)
    tmp326 = tl.where(tmp292, tmp295, tmp325)
    tmp327 = tl.full([1], 59, tl.int32)
    tmp328 = tmp0 == tmp327
    tmp331 = tmp330 * tmp5
    tmp332 = tl.full([1], 58, tl.int32)
    tmp333 = tmp0 == tmp332
    tmp336 = tmp335 * tmp5
    tmp337 = tl.full([1], 57, tl.int32)
    tmp338 = tmp0 == tmp337
    tmp341 = tmp340 * tmp5
    tmp342 = tl.full([1], 56, tl.int32)
    tmp343 = tmp0 == tmp342
    tmp346 = tmp345 * tmp5
    tmp347 = tl.full([1], 55, tl.int32)
    tmp348 = tmp0 == tmp347
    tmp351 = tmp350 * tmp5
    tmp352 = tl.full([1], 54, tl.int32)
    tmp353 = tmp0 == tmp352
    tmp356 = tmp355 * tmp5
    tmp357 = tl.where(tmp353, tmp356, tmp326)
    tmp358 = tl.where(tmp348, tmp351, tmp357)
    tmp359 = tl.where(tmp343, tmp346, tmp358)
    tmp360 = tl.where(tmp338, tmp341, tmp359)
    tmp361 = tl.where(tmp333, tmp336, tmp360)
    tmp362 = tl.where(tmp328, tmp331, tmp361)
    tmp363 = tl.full([1], 65, tl.int32)
    tmp364 = tmp0 == tmp363
    tmp367 = tmp366 * tmp5
    tmp368 = tl.full([1], 64, tl.int32)
    tmp369 = tmp0 == tmp368
    tmp372 = tmp371 * tmp5
    tmp373 = tl.full([1], 63, tl.int32)
    tmp374 = tmp0 == tmp373
    tmp377 = tmp376 * tmp5
    tmp378 = tl.full([1], 62, tl.int32)
    tmp379 = tmp0 == tmp378
    tmp382 = tmp381 * tmp5
    tmp383 = tl.full([1], 61, tl.int32)
    tmp384 = tmp0 == tmp383
    tmp387 = tmp386 * tmp5
    tmp388 = tl.full([1], 60, tl.int32)
    tmp389 = tmp0 == tmp388
    tmp392 = tmp391 * tmp5
    tmp393 = tl.where(tmp389, tmp392, tmp362)
    tmp394 = tl.where(tmp384, tmp387, tmp393)
    tmp395 = tl.where(tmp379, tmp382, tmp394)
    tmp396 = tl.where(tmp374, tmp377, tmp395)
    tmp397 = tl.where(tmp369, tmp372, tmp396)
    tmp398 = tl.where(tmp364, tmp367, tmp397)
    tmp399 = tl.full([1], 71, tl.int32)
    tmp400 = tmp0 == tmp399
    tmp403 = tmp402 * tmp5
    tmp404 = tl.full([1], 70, tl.int32)
    tmp405 = tmp0 == tmp404
    tmp408 = tmp407 * tmp5
    tmp409 = tl.full([1], 69, tl.int32)
    tmp410 = tmp0 == tmp409
    tmp413 = tmp412 * tmp5
    tmp414 = tl.full([1], 68, tl.int32)
    tmp415 = tmp0 == tmp414
    tmp418 = tmp417 * tmp5
    tmp419 = tl.full([1], 67, tl.int32)
    tmp420 = tmp0 == tmp419
    tmp423 = tmp422 * tmp5
    tmp424 = tl.full([1], 66, tl.int32)
    tmp425 = tmp0 == tmp424
    tmp428 = tmp427 * tmp5
    tmp429 = tl.where(tmp425, tmp428, tmp398)
    tmp430 = tl.where(tmp420, tmp423, tmp429)
    tmp431 = tl.where(tmp415, tmp418, tmp430)
    tmp432 = tl.where(tmp410, tmp413, tmp431)
    tmp433 = tl.where(tmp405, tmp408, tmp432)
    tmp434 = tl.where(tmp400, tmp403, tmp433)
    tmp435 = tl.full([1], 77, tl.int32)
    tmp436 = tmp0 == tmp435
    tmp439 = tmp438 * tmp5
    tmp440 = tl.full([1], 76, tl.int32)
    tmp441 = tmp0 == tmp440
    tmp444 = tmp443 * tmp5
    tmp445 = tl.full([1], 75, tl.int32)
    tmp446 = tmp0 == tmp445
    tmp449 = tmp448 * tmp5
    tmp450 = tl.full([1], 74, tl.int32)
    tmp451 = tmp0 == tmp450
    tmp454 = tmp453 * tmp5
    tmp455 = tl.full([1], 73, tl.int32)
    tmp456 = tmp0 == tmp455
    tmp459 = tmp458 * tmp5
    tmp460 = tl.full([1], 72, tl.int32)
    tmp461 = tmp0 == tmp460
    tmp464 = tmp463 * tmp5
    tmp465 = tl.where(tmp461, tmp464, tmp434)
    tmp466 = tl.where(tmp456, tmp459, tmp465)
    tmp467 = tl.where(tmp451, tmp454, tmp466)
    tmp468 = tl.where(tmp446, tmp449, tmp467)
    tmp469 = tl.where(tmp441, tmp444, tmp468)
    tmp470 = tl.where(tmp436, tmp439, tmp469)
    tmp471 = tl.full([1], 83, tl.int32)
    tmp472 = tmp0 == tmp471
    tmp475 = tmp474 * tmp5
    tmp476 = tl.full([1], 82, tl.int32)
    tmp477 = tmp0 == tmp476
    tmp480 = tmp479 * tmp5
    tmp481 = tl.full([1], 81, tl.int32)
    tmp482 = tmp0 == tmp481
    tmp485 = tmp484 * tmp5
    tmp486 = tl.full([1], 80, tl.int32)
    tmp487 = tmp0 == tmp486
    tmp490 = tmp489 * tmp5
    tmp491 = tl.full([1], 79, tl.int32)
    tmp492 = tmp0 == tmp491
    tmp495 = tmp494 * tmp5
    tmp496 = tl.full([1], 78, tl.int32)
    tmp497 = tmp0 == tmp496
    tmp500 = tmp499 * tmp5
    tmp501 = tl.where(tmp497, tmp500, tmp470)
    tmp502 = tl.where(tmp492, tmp495, tmp501)
    tmp503 = tl.where(tmp487, tmp490, tmp502)
    tmp504 = tl.where(tmp482, tmp485, tmp503)
    tmp505 = tl.where(tmp477, tmp480, tmp504)
    tmp506 = tl.where(tmp472, tmp475, tmp505)
    tmp507 = tl.full([1], 89, tl.int32)
    tmp508 = tmp0 == tmp507
    tmp511 = tmp510 * tmp5
    tmp512 = tl.full([1], 88, tl.int32)
    tmp513 = tmp0 == tmp512
    tmp516 = tmp515 * tmp5
    tmp517 = tl.full([1], 87, tl.int32)
    tmp518 = tmp0 == tmp517
    tmp521 = tmp520 * tmp5
    tmp522 = tl.full([1], 86, tl.int32)
    tmp523 = tmp0 == tmp522
    tmp526 = tmp525 * tmp5
    tmp527 = tl.full([1], 85, tl.int32)
    tmp528 = tmp0 == tmp527
    tmp531 = tmp530 * tmp5
    tmp532 = tl.full([1], 84, tl.int32)
    tmp533 = tmp0 == tmp532
    tmp536 = tmp535 * tmp5
    tmp537 = tl.where(tmp533, tmp536, tmp506)
    tmp538 = tl.where(tmp528, tmp531, tmp537)
    tmp539 = tl.where(tmp523, tmp526, tmp538)
    tmp540 = tl.where(tmp518, tmp521, tmp539)
    tmp541 = tl.where(tmp513, tmp516, tmp540)
    tmp542 = tl.where(tmp508, tmp511, tmp541)
    tmp543 = tl.full([1], 95, tl.int32)
    tmp544 = tmp0 == tmp543
    tmp547 = tmp546 * tmp5
    tmp548 = tl.full([1], 94, tl.int32)
    tmp549 = tmp0 == tmp548
    tmp552 = tmp551 * tmp5
    tmp553 = tl.full([1], 93, tl.int32)
    tmp554 = tmp0 == tmp553
    tmp557 = tmp556 * tmp5
    tmp558 = tl.full([1], 92, tl.int32)
    tmp559 = tmp0 == tmp558
    tmp562 = tmp561 * tmp5
    tmp563 = tl.full([1], 91, tl.int32)
    tmp564 = tmp0 == tmp563
    tmp567 = tmp566 * tmp5
    tmp568 = tl.full([1], 90, tl.int32)
    tmp569 = tmp0 == tmp568
    tmp572 = tmp571 * tmp5
    tmp573 = tl.where(tmp569, tmp572, tmp542)
    tmp574 = tl.where(tmp564, tmp567, tmp573)
    tmp575 = tl.where(tmp559, tmp562, tmp574)
    tmp576 = tl.where(tmp554, tmp557, tmp575)
    tmp577 = tl.where(tmp549, tmp552, tmp576)
    tmp578 = tl.where(tmp544, tmp547, tmp577)
    tmp579 = tl.full([1], 99, tl.int32)
    tmp580 = tmp0 == tmp579
    tmp583 = tmp582 * tmp5
    tmp584 = tl.full([1], 98, tl.int32)
    tmp585 = tmp0 == tmp584
    tmp588 = tmp587 * tmp5
    tmp589 = tl.full([1], 97, tl.int32)
    tmp590 = tmp0 == tmp589
    tmp593 = tmp592 * tmp5
    tmp594 = tl.full([1], 96, tl.int32)
    tmp595 = tmp0 == tmp594
    tmp598 = tmp597 * tmp5
    tmp599 = tl.where(tmp595, tmp598, tmp578)
    tmp600 = tl.where(tmp590, tmp593, tmp599)
    tmp601 = tl.where(tmp585, tmp588, tmp600)
    tmp602 = tl.where(tmp580, tmp583, tmp601)
    tl.store(in_out_ptr0 + (x0), tmp602, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, = args
    args.clear()
    assert_size_stride(primals_1, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((100, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_0.run(buf0, 100, grid=grid(100), stream=stream0)
        buf1 = empty_strided_cuda((100, ), (1, ), torch.float32)
        buf2 = buf1; del buf1  # reuse
        buf3 = buf2; del buf2  # reuse
        buf4 = buf3; del buf3  # reuse
        buf5 = buf4; del buf4  # reuse
        buf6 = buf5; del buf5  # reuse
        buf7 = buf6; del buf6  # reuse
        buf8 = buf7; del buf7  # reuse
        buf9 = buf8; del buf8  # reuse
        buf10 = buf9; del buf9  # reuse
        buf11 = buf10; del buf10  # reuse
        buf12 = buf11; del buf11  # reuse
        buf13 = buf12; del buf12  # reuse
        buf14 = buf13; del buf13  # reuse
        buf15 = buf14; del buf14  # reuse
        buf16 = buf15; del buf15  # reuse
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, mul_58, mul_59, mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, mul_88, mul_89, mul_90, mul_91, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, mul_98, mul_99], Original ATen: [aten.mul]
        triton_poi_fused_mul_1.run(buf17, primals_1, 100, grid=grid(100), stream=stream0)
        del primals_1
    return (buf17, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
