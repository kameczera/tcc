
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True




isolate_fails_code_str = None



# torch version: 2.5.1+cu121
# torch cuda version: 12.1
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Sep_12_02:18:05_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.77 
# Build cuda_12.6.r12.6/compiler.34841621_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3050 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, full_default, tangents_1):
        select_500 = torch.ops.aten.select.int(tangents_1, 0, 99)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_100 = torch.ops.aten.select_scatter.default(tangents_1, full_default_1, 0, 99);  tangents_1 = None
        mul_100 = torch.ops.aten.mul.Tensor(select_500, 2);  select_500 = None
        select_scatter_101 = torch.ops.aten.select_scatter.default(full_default, mul_100, 0, 99);  mul_100 = None
        select_503 = torch.ops.aten.select.int(select_scatter_100, 0, 98)
        select_scatter_102 = torch.ops.aten.select_scatter.default(select_scatter_100, full_default_1, 0, 98);  select_scatter_100 = None
        mul_101 = torch.ops.aten.mul.Tensor(select_503, 2);  select_503 = None
        select_scatter_103 = torch.ops.aten.select_scatter.default(full_default, mul_101, 0, 98);  mul_101 = None
        add = torch.ops.aten.add.Tensor(select_scatter_101, select_scatter_103);  select_scatter_101 = select_scatter_103 = None
        select_506 = torch.ops.aten.select.int(select_scatter_102, 0, 97)
        select_scatter_104 = torch.ops.aten.select_scatter.default(select_scatter_102, full_default_1, 0, 97);  select_scatter_102 = None
        mul_102 = torch.ops.aten.mul.Tensor(select_506, 2);  select_506 = None
        select_scatter_105 = torch.ops.aten.select_scatter.default(full_default, mul_102, 0, 97);  mul_102 = None
        add_1 = torch.ops.aten.add.Tensor(add, select_scatter_105);  add = select_scatter_105 = None
        select_509 = torch.ops.aten.select.int(select_scatter_104, 0, 96)
        select_scatter_106 = torch.ops.aten.select_scatter.default(select_scatter_104, full_default_1, 0, 96);  select_scatter_104 = None
        mul_103 = torch.ops.aten.mul.Tensor(select_509, 2);  select_509 = None
        select_scatter_107 = torch.ops.aten.select_scatter.default(full_default, mul_103, 0, 96);  mul_103 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, select_scatter_107);  add_1 = select_scatter_107 = None
        select_512 = torch.ops.aten.select.int(select_scatter_106, 0, 95)
        select_scatter_108 = torch.ops.aten.select_scatter.default(select_scatter_106, full_default_1, 0, 95);  select_scatter_106 = None
        mul_104 = torch.ops.aten.mul.Tensor(select_512, 2);  select_512 = None
        select_scatter_109 = torch.ops.aten.select_scatter.default(full_default, mul_104, 0, 95);  mul_104 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, select_scatter_109);  add_2 = select_scatter_109 = None
        select_515 = torch.ops.aten.select.int(select_scatter_108, 0, 94)
        select_scatter_110 = torch.ops.aten.select_scatter.default(select_scatter_108, full_default_1, 0, 94);  select_scatter_108 = None
        mul_105 = torch.ops.aten.mul.Tensor(select_515, 2);  select_515 = None
        select_scatter_111 = torch.ops.aten.select_scatter.default(full_default, mul_105, 0, 94);  mul_105 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, select_scatter_111);  add_3 = select_scatter_111 = None
        select_518 = torch.ops.aten.select.int(select_scatter_110, 0, 93)
        select_scatter_112 = torch.ops.aten.select_scatter.default(select_scatter_110, full_default_1, 0, 93);  select_scatter_110 = None
        mul_106 = torch.ops.aten.mul.Tensor(select_518, 2);  select_518 = None
        select_scatter_113 = torch.ops.aten.select_scatter.default(full_default, mul_106, 0, 93);  mul_106 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, select_scatter_113);  add_4 = select_scatter_113 = None
        select_521 = torch.ops.aten.select.int(select_scatter_112, 0, 92)
        select_scatter_114 = torch.ops.aten.select_scatter.default(select_scatter_112, full_default_1, 0, 92);  select_scatter_112 = None
        mul_107 = torch.ops.aten.mul.Tensor(select_521, 2);  select_521 = None
        select_scatter_115 = torch.ops.aten.select_scatter.default(full_default, mul_107, 0, 92);  mul_107 = None
        add_6 = torch.ops.aten.add.Tensor(add_5, select_scatter_115);  add_5 = select_scatter_115 = None
        select_524 = torch.ops.aten.select.int(select_scatter_114, 0, 91)
        select_scatter_116 = torch.ops.aten.select_scatter.default(select_scatter_114, full_default_1, 0, 91);  select_scatter_114 = None
        mul_108 = torch.ops.aten.mul.Tensor(select_524, 2);  select_524 = None
        select_scatter_117 = torch.ops.aten.select_scatter.default(full_default, mul_108, 0, 91);  mul_108 = None
        add_7 = torch.ops.aten.add.Tensor(add_6, select_scatter_117);  add_6 = select_scatter_117 = None
        select_527 = torch.ops.aten.select.int(select_scatter_116, 0, 90)
        select_scatter_118 = torch.ops.aten.select_scatter.default(select_scatter_116, full_default_1, 0, 90);  select_scatter_116 = None
        mul_109 = torch.ops.aten.mul.Tensor(select_527, 2);  select_527 = None
        select_scatter_119 = torch.ops.aten.select_scatter.default(full_default, mul_109, 0, 90);  mul_109 = None
        add_8 = torch.ops.aten.add.Tensor(add_7, select_scatter_119);  add_7 = select_scatter_119 = None
        select_530 = torch.ops.aten.select.int(select_scatter_118, 0, 89)
        select_scatter_120 = torch.ops.aten.select_scatter.default(select_scatter_118, full_default_1, 0, 89);  select_scatter_118 = None
        mul_110 = torch.ops.aten.mul.Tensor(select_530, 2);  select_530 = None
        select_scatter_121 = torch.ops.aten.select_scatter.default(full_default, mul_110, 0, 89);  mul_110 = None
        add_9 = torch.ops.aten.add.Tensor(add_8, select_scatter_121);  add_8 = select_scatter_121 = None
        select_533 = torch.ops.aten.select.int(select_scatter_120, 0, 88)
        select_scatter_122 = torch.ops.aten.select_scatter.default(select_scatter_120, full_default_1, 0, 88);  select_scatter_120 = None
        mul_111 = torch.ops.aten.mul.Tensor(select_533, 2);  select_533 = None
        select_scatter_123 = torch.ops.aten.select_scatter.default(full_default, mul_111, 0, 88);  mul_111 = None
        add_10 = torch.ops.aten.add.Tensor(add_9, select_scatter_123);  add_9 = select_scatter_123 = None
        select_536 = torch.ops.aten.select.int(select_scatter_122, 0, 87)
        select_scatter_124 = torch.ops.aten.select_scatter.default(select_scatter_122, full_default_1, 0, 87);  select_scatter_122 = None
        mul_112 = torch.ops.aten.mul.Tensor(select_536, 2);  select_536 = None
        select_scatter_125 = torch.ops.aten.select_scatter.default(full_default, mul_112, 0, 87);  mul_112 = None
        add_11 = torch.ops.aten.add.Tensor(add_10, select_scatter_125);  add_10 = select_scatter_125 = None
        select_539 = torch.ops.aten.select.int(select_scatter_124, 0, 86)
        select_scatter_126 = torch.ops.aten.select_scatter.default(select_scatter_124, full_default_1, 0, 86);  select_scatter_124 = None
        mul_113 = torch.ops.aten.mul.Tensor(select_539, 2);  select_539 = None
        select_scatter_127 = torch.ops.aten.select_scatter.default(full_default, mul_113, 0, 86);  mul_113 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, select_scatter_127);  add_11 = select_scatter_127 = None
        select_542 = torch.ops.aten.select.int(select_scatter_126, 0, 85)
        select_scatter_128 = torch.ops.aten.select_scatter.default(select_scatter_126, full_default_1, 0, 85);  select_scatter_126 = None
        mul_114 = torch.ops.aten.mul.Tensor(select_542, 2);  select_542 = None
        select_scatter_129 = torch.ops.aten.select_scatter.default(full_default, mul_114, 0, 85);  mul_114 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, select_scatter_129);  add_12 = select_scatter_129 = None
        select_545 = torch.ops.aten.select.int(select_scatter_128, 0, 84)
        select_scatter_130 = torch.ops.aten.select_scatter.default(select_scatter_128, full_default_1, 0, 84);  select_scatter_128 = None
        mul_115 = torch.ops.aten.mul.Tensor(select_545, 2);  select_545 = None
        select_scatter_131 = torch.ops.aten.select_scatter.default(full_default, mul_115, 0, 84);  mul_115 = None
        add_14 = torch.ops.aten.add.Tensor(add_13, select_scatter_131);  add_13 = select_scatter_131 = None
        select_548 = torch.ops.aten.select.int(select_scatter_130, 0, 83)
        select_scatter_132 = torch.ops.aten.select_scatter.default(select_scatter_130, full_default_1, 0, 83);  select_scatter_130 = None
        mul_116 = torch.ops.aten.mul.Tensor(select_548, 2);  select_548 = None
        select_scatter_133 = torch.ops.aten.select_scatter.default(full_default, mul_116, 0, 83);  mul_116 = None
        add_15 = torch.ops.aten.add.Tensor(add_14, select_scatter_133);  add_14 = select_scatter_133 = None
        select_551 = torch.ops.aten.select.int(select_scatter_132, 0, 82)
        select_scatter_134 = torch.ops.aten.select_scatter.default(select_scatter_132, full_default_1, 0, 82);  select_scatter_132 = None
        mul_117 = torch.ops.aten.mul.Tensor(select_551, 2);  select_551 = None
        select_scatter_135 = torch.ops.aten.select_scatter.default(full_default, mul_117, 0, 82);  mul_117 = None
        add_16 = torch.ops.aten.add.Tensor(add_15, select_scatter_135);  add_15 = select_scatter_135 = None
        select_554 = torch.ops.aten.select.int(select_scatter_134, 0, 81)
        select_scatter_136 = torch.ops.aten.select_scatter.default(select_scatter_134, full_default_1, 0, 81);  select_scatter_134 = None
        mul_118 = torch.ops.aten.mul.Tensor(select_554, 2);  select_554 = None
        select_scatter_137 = torch.ops.aten.select_scatter.default(full_default, mul_118, 0, 81);  mul_118 = None
        add_17 = torch.ops.aten.add.Tensor(add_16, select_scatter_137);  add_16 = select_scatter_137 = None
        select_557 = torch.ops.aten.select.int(select_scatter_136, 0, 80)
        select_scatter_138 = torch.ops.aten.select_scatter.default(select_scatter_136, full_default_1, 0, 80);  select_scatter_136 = None
        mul_119 = torch.ops.aten.mul.Tensor(select_557, 2);  select_557 = None
        select_scatter_139 = torch.ops.aten.select_scatter.default(full_default, mul_119, 0, 80);  mul_119 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, select_scatter_139);  add_17 = select_scatter_139 = None
        select_560 = torch.ops.aten.select.int(select_scatter_138, 0, 79)
        select_scatter_140 = torch.ops.aten.select_scatter.default(select_scatter_138, full_default_1, 0, 79);  select_scatter_138 = None
        mul_120 = torch.ops.aten.mul.Tensor(select_560, 2);  select_560 = None
        select_scatter_141 = torch.ops.aten.select_scatter.default(full_default, mul_120, 0, 79);  mul_120 = None
        add_19 = torch.ops.aten.add.Tensor(add_18, select_scatter_141);  add_18 = select_scatter_141 = None
        select_563 = torch.ops.aten.select.int(select_scatter_140, 0, 78)
        select_scatter_142 = torch.ops.aten.select_scatter.default(select_scatter_140, full_default_1, 0, 78);  select_scatter_140 = None
        mul_121 = torch.ops.aten.mul.Tensor(select_563, 2);  select_563 = None
        select_scatter_143 = torch.ops.aten.select_scatter.default(full_default, mul_121, 0, 78);  mul_121 = None
        add_20 = torch.ops.aten.add.Tensor(add_19, select_scatter_143);  add_19 = select_scatter_143 = None
        select_566 = torch.ops.aten.select.int(select_scatter_142, 0, 77)
        select_scatter_144 = torch.ops.aten.select_scatter.default(select_scatter_142, full_default_1, 0, 77);  select_scatter_142 = None
        mul_122 = torch.ops.aten.mul.Tensor(select_566, 2);  select_566 = None
        select_scatter_145 = torch.ops.aten.select_scatter.default(full_default, mul_122, 0, 77);  mul_122 = None
        add_21 = torch.ops.aten.add.Tensor(add_20, select_scatter_145);  add_20 = select_scatter_145 = None
        select_569 = torch.ops.aten.select.int(select_scatter_144, 0, 76)
        select_scatter_146 = torch.ops.aten.select_scatter.default(select_scatter_144, full_default_1, 0, 76);  select_scatter_144 = None
        mul_123 = torch.ops.aten.mul.Tensor(select_569, 2);  select_569 = None
        select_scatter_147 = torch.ops.aten.select_scatter.default(full_default, mul_123, 0, 76);  mul_123 = None
        add_22 = torch.ops.aten.add.Tensor(add_21, select_scatter_147);  add_21 = select_scatter_147 = None
        select_572 = torch.ops.aten.select.int(select_scatter_146, 0, 75)
        select_scatter_148 = torch.ops.aten.select_scatter.default(select_scatter_146, full_default_1, 0, 75);  select_scatter_146 = None
        mul_124 = torch.ops.aten.mul.Tensor(select_572, 2);  select_572 = None
        select_scatter_149 = torch.ops.aten.select_scatter.default(full_default, mul_124, 0, 75);  mul_124 = None
        add_23 = torch.ops.aten.add.Tensor(add_22, select_scatter_149);  add_22 = select_scatter_149 = None
        select_575 = torch.ops.aten.select.int(select_scatter_148, 0, 74)
        select_scatter_150 = torch.ops.aten.select_scatter.default(select_scatter_148, full_default_1, 0, 74);  select_scatter_148 = None
        mul_125 = torch.ops.aten.mul.Tensor(select_575, 2);  select_575 = None
        select_scatter_151 = torch.ops.aten.select_scatter.default(full_default, mul_125, 0, 74);  mul_125 = None
        add_24 = torch.ops.aten.add.Tensor(add_23, select_scatter_151);  add_23 = select_scatter_151 = None
        select_578 = torch.ops.aten.select.int(select_scatter_150, 0, 73)
        select_scatter_152 = torch.ops.aten.select_scatter.default(select_scatter_150, full_default_1, 0, 73);  select_scatter_150 = None
        mul_126 = torch.ops.aten.mul.Tensor(select_578, 2);  select_578 = None
        select_scatter_153 = torch.ops.aten.select_scatter.default(full_default, mul_126, 0, 73);  mul_126 = None
        add_25 = torch.ops.aten.add.Tensor(add_24, select_scatter_153);  add_24 = select_scatter_153 = None
        select_581 = torch.ops.aten.select.int(select_scatter_152, 0, 72)
        select_scatter_154 = torch.ops.aten.select_scatter.default(select_scatter_152, full_default_1, 0, 72);  select_scatter_152 = None
        mul_127 = torch.ops.aten.mul.Tensor(select_581, 2);  select_581 = None
        select_scatter_155 = torch.ops.aten.select_scatter.default(full_default, mul_127, 0, 72);  mul_127 = None
        add_26 = torch.ops.aten.add.Tensor(add_25, select_scatter_155);  add_25 = select_scatter_155 = None
        select_584 = torch.ops.aten.select.int(select_scatter_154, 0, 71)
        select_scatter_156 = torch.ops.aten.select_scatter.default(select_scatter_154, full_default_1, 0, 71);  select_scatter_154 = None
        mul_128 = torch.ops.aten.mul.Tensor(select_584, 2);  select_584 = None
        select_scatter_157 = torch.ops.aten.select_scatter.default(full_default, mul_128, 0, 71);  mul_128 = None
        add_27 = torch.ops.aten.add.Tensor(add_26, select_scatter_157);  add_26 = select_scatter_157 = None
        select_587 = torch.ops.aten.select.int(select_scatter_156, 0, 70)
        select_scatter_158 = torch.ops.aten.select_scatter.default(select_scatter_156, full_default_1, 0, 70);  select_scatter_156 = None
        mul_129 = torch.ops.aten.mul.Tensor(select_587, 2);  select_587 = None
        select_scatter_159 = torch.ops.aten.select_scatter.default(full_default, mul_129, 0, 70);  mul_129 = None
        add_28 = torch.ops.aten.add.Tensor(add_27, select_scatter_159);  add_27 = select_scatter_159 = None
        select_590 = torch.ops.aten.select.int(select_scatter_158, 0, 69)
        select_scatter_160 = torch.ops.aten.select_scatter.default(select_scatter_158, full_default_1, 0, 69);  select_scatter_158 = None
        mul_130 = torch.ops.aten.mul.Tensor(select_590, 2);  select_590 = None
        select_scatter_161 = torch.ops.aten.select_scatter.default(full_default, mul_130, 0, 69);  mul_130 = None
        add_29 = torch.ops.aten.add.Tensor(add_28, select_scatter_161);  add_28 = select_scatter_161 = None
        select_593 = torch.ops.aten.select.int(select_scatter_160, 0, 68)
        select_scatter_162 = torch.ops.aten.select_scatter.default(select_scatter_160, full_default_1, 0, 68);  select_scatter_160 = None
        mul_131 = torch.ops.aten.mul.Tensor(select_593, 2);  select_593 = None
        select_scatter_163 = torch.ops.aten.select_scatter.default(full_default, mul_131, 0, 68);  mul_131 = None
        add_30 = torch.ops.aten.add.Tensor(add_29, select_scatter_163);  add_29 = select_scatter_163 = None
        select_596 = torch.ops.aten.select.int(select_scatter_162, 0, 67)
        select_scatter_164 = torch.ops.aten.select_scatter.default(select_scatter_162, full_default_1, 0, 67);  select_scatter_162 = None
        mul_132 = torch.ops.aten.mul.Tensor(select_596, 2);  select_596 = None
        select_scatter_165 = torch.ops.aten.select_scatter.default(full_default, mul_132, 0, 67);  mul_132 = None
        add_31 = torch.ops.aten.add.Tensor(add_30, select_scatter_165);  add_30 = select_scatter_165 = None
        select_599 = torch.ops.aten.select.int(select_scatter_164, 0, 66)
        select_scatter_166 = torch.ops.aten.select_scatter.default(select_scatter_164, full_default_1, 0, 66);  select_scatter_164 = None
        mul_133 = torch.ops.aten.mul.Tensor(select_599, 2);  select_599 = None
        select_scatter_167 = torch.ops.aten.select_scatter.default(full_default, mul_133, 0, 66);  mul_133 = None
        add_32 = torch.ops.aten.add.Tensor(add_31, select_scatter_167);  add_31 = select_scatter_167 = None
        select_602 = torch.ops.aten.select.int(select_scatter_166, 0, 65)
        select_scatter_168 = torch.ops.aten.select_scatter.default(select_scatter_166, full_default_1, 0, 65);  select_scatter_166 = None
        mul_134 = torch.ops.aten.mul.Tensor(select_602, 2);  select_602 = None
        select_scatter_169 = torch.ops.aten.select_scatter.default(full_default, mul_134, 0, 65);  mul_134 = None
        add_33 = torch.ops.aten.add.Tensor(add_32, select_scatter_169);  add_32 = select_scatter_169 = None
        select_605 = torch.ops.aten.select.int(select_scatter_168, 0, 64)
        select_scatter_170 = torch.ops.aten.select_scatter.default(select_scatter_168, full_default_1, 0, 64);  select_scatter_168 = None
        mul_135 = torch.ops.aten.mul.Tensor(select_605, 2);  select_605 = None
        select_scatter_171 = torch.ops.aten.select_scatter.default(full_default, mul_135, 0, 64);  mul_135 = None
        add_34 = torch.ops.aten.add.Tensor(add_33, select_scatter_171);  add_33 = select_scatter_171 = None
        select_608 = torch.ops.aten.select.int(select_scatter_170, 0, 63)
        select_scatter_172 = torch.ops.aten.select_scatter.default(select_scatter_170, full_default_1, 0, 63);  select_scatter_170 = None
        mul_136 = torch.ops.aten.mul.Tensor(select_608, 2);  select_608 = None
        select_scatter_173 = torch.ops.aten.select_scatter.default(full_default, mul_136, 0, 63);  mul_136 = None
        add_35 = torch.ops.aten.add.Tensor(add_34, select_scatter_173);  add_34 = select_scatter_173 = None
        select_611 = torch.ops.aten.select.int(select_scatter_172, 0, 62)
        select_scatter_174 = torch.ops.aten.select_scatter.default(select_scatter_172, full_default_1, 0, 62);  select_scatter_172 = None
        mul_137 = torch.ops.aten.mul.Tensor(select_611, 2);  select_611 = None
        select_scatter_175 = torch.ops.aten.select_scatter.default(full_default, mul_137, 0, 62);  mul_137 = None
        add_36 = torch.ops.aten.add.Tensor(add_35, select_scatter_175);  add_35 = select_scatter_175 = None
        select_614 = torch.ops.aten.select.int(select_scatter_174, 0, 61)
        select_scatter_176 = torch.ops.aten.select_scatter.default(select_scatter_174, full_default_1, 0, 61);  select_scatter_174 = None
        mul_138 = torch.ops.aten.mul.Tensor(select_614, 2);  select_614 = None
        select_scatter_177 = torch.ops.aten.select_scatter.default(full_default, mul_138, 0, 61);  mul_138 = None
        add_37 = torch.ops.aten.add.Tensor(add_36, select_scatter_177);  add_36 = select_scatter_177 = None
        select_617 = torch.ops.aten.select.int(select_scatter_176, 0, 60)
        select_scatter_178 = torch.ops.aten.select_scatter.default(select_scatter_176, full_default_1, 0, 60);  select_scatter_176 = None
        mul_139 = torch.ops.aten.mul.Tensor(select_617, 2);  select_617 = None
        select_scatter_179 = torch.ops.aten.select_scatter.default(full_default, mul_139, 0, 60);  mul_139 = None
        add_38 = torch.ops.aten.add.Tensor(add_37, select_scatter_179);  add_37 = select_scatter_179 = None
        select_620 = torch.ops.aten.select.int(select_scatter_178, 0, 59)
        select_scatter_180 = torch.ops.aten.select_scatter.default(select_scatter_178, full_default_1, 0, 59);  select_scatter_178 = None
        mul_140 = torch.ops.aten.mul.Tensor(select_620, 2);  select_620 = None
        select_scatter_181 = torch.ops.aten.select_scatter.default(full_default, mul_140, 0, 59);  mul_140 = None
        add_39 = torch.ops.aten.add.Tensor(add_38, select_scatter_181);  add_38 = select_scatter_181 = None
        select_623 = torch.ops.aten.select.int(select_scatter_180, 0, 58)
        select_scatter_182 = torch.ops.aten.select_scatter.default(select_scatter_180, full_default_1, 0, 58);  select_scatter_180 = None
        mul_141 = torch.ops.aten.mul.Tensor(select_623, 2);  select_623 = None
        select_scatter_183 = torch.ops.aten.select_scatter.default(full_default, mul_141, 0, 58);  mul_141 = None
        add_40 = torch.ops.aten.add.Tensor(add_39, select_scatter_183);  add_39 = select_scatter_183 = None
        select_626 = torch.ops.aten.select.int(select_scatter_182, 0, 57)
        select_scatter_184 = torch.ops.aten.select_scatter.default(select_scatter_182, full_default_1, 0, 57);  select_scatter_182 = None
        mul_142 = torch.ops.aten.mul.Tensor(select_626, 2);  select_626 = None
        select_scatter_185 = torch.ops.aten.select_scatter.default(full_default, mul_142, 0, 57);  mul_142 = None
        add_41 = torch.ops.aten.add.Tensor(add_40, select_scatter_185);  add_40 = select_scatter_185 = None
        select_629 = torch.ops.aten.select.int(select_scatter_184, 0, 56)
        select_scatter_186 = torch.ops.aten.select_scatter.default(select_scatter_184, full_default_1, 0, 56);  select_scatter_184 = None
        mul_143 = torch.ops.aten.mul.Tensor(select_629, 2);  select_629 = None
        select_scatter_187 = torch.ops.aten.select_scatter.default(full_default, mul_143, 0, 56);  mul_143 = None
        add_42 = torch.ops.aten.add.Tensor(add_41, select_scatter_187);  add_41 = select_scatter_187 = None
        select_632 = torch.ops.aten.select.int(select_scatter_186, 0, 55)
        select_scatter_188 = torch.ops.aten.select_scatter.default(select_scatter_186, full_default_1, 0, 55);  select_scatter_186 = None
        mul_144 = torch.ops.aten.mul.Tensor(select_632, 2);  select_632 = None
        select_scatter_189 = torch.ops.aten.select_scatter.default(full_default, mul_144, 0, 55);  mul_144 = None
        add_43 = torch.ops.aten.add.Tensor(add_42, select_scatter_189);  add_42 = select_scatter_189 = None
        select_635 = torch.ops.aten.select.int(select_scatter_188, 0, 54)
        select_scatter_190 = torch.ops.aten.select_scatter.default(select_scatter_188, full_default_1, 0, 54);  select_scatter_188 = None
        mul_145 = torch.ops.aten.mul.Tensor(select_635, 2);  select_635 = None
        select_scatter_191 = torch.ops.aten.select_scatter.default(full_default, mul_145, 0, 54);  mul_145 = None
        add_44 = torch.ops.aten.add.Tensor(add_43, select_scatter_191);  add_43 = select_scatter_191 = None
        select_638 = torch.ops.aten.select.int(select_scatter_190, 0, 53)
        select_scatter_192 = torch.ops.aten.select_scatter.default(select_scatter_190, full_default_1, 0, 53);  select_scatter_190 = None
        mul_146 = torch.ops.aten.mul.Tensor(select_638, 2);  select_638 = None
        select_scatter_193 = torch.ops.aten.select_scatter.default(full_default, mul_146, 0, 53);  mul_146 = None
        add_45 = torch.ops.aten.add.Tensor(add_44, select_scatter_193);  add_44 = select_scatter_193 = None
        select_641 = torch.ops.aten.select.int(select_scatter_192, 0, 52)
        select_scatter_194 = torch.ops.aten.select_scatter.default(select_scatter_192, full_default_1, 0, 52);  select_scatter_192 = None
        mul_147 = torch.ops.aten.mul.Tensor(select_641, 2);  select_641 = None
        select_scatter_195 = torch.ops.aten.select_scatter.default(full_default, mul_147, 0, 52);  mul_147 = None
        add_46 = torch.ops.aten.add.Tensor(add_45, select_scatter_195);  add_45 = select_scatter_195 = None
        select_644 = torch.ops.aten.select.int(select_scatter_194, 0, 51)
        select_scatter_196 = torch.ops.aten.select_scatter.default(select_scatter_194, full_default_1, 0, 51);  select_scatter_194 = None
        mul_148 = torch.ops.aten.mul.Tensor(select_644, 2);  select_644 = None
        select_scatter_197 = torch.ops.aten.select_scatter.default(full_default, mul_148, 0, 51);  mul_148 = None
        add_47 = torch.ops.aten.add.Tensor(add_46, select_scatter_197);  add_46 = select_scatter_197 = None
        select_647 = torch.ops.aten.select.int(select_scatter_196, 0, 50)
        select_scatter_198 = torch.ops.aten.select_scatter.default(select_scatter_196, full_default_1, 0, 50);  select_scatter_196 = None
        mul_149 = torch.ops.aten.mul.Tensor(select_647, 2);  select_647 = None
        select_scatter_199 = torch.ops.aten.select_scatter.default(full_default, mul_149, 0, 50);  mul_149 = None
        add_48 = torch.ops.aten.add.Tensor(add_47, select_scatter_199);  add_47 = select_scatter_199 = None
        select_650 = torch.ops.aten.select.int(select_scatter_198, 0, 49)
        select_scatter_200 = torch.ops.aten.select_scatter.default(select_scatter_198, full_default_1, 0, 49);  select_scatter_198 = None
        mul_150 = torch.ops.aten.mul.Tensor(select_650, 2);  select_650 = None
        select_scatter_201 = torch.ops.aten.select_scatter.default(full_default, mul_150, 0, 49);  mul_150 = None
        add_49 = torch.ops.aten.add.Tensor(add_48, select_scatter_201);  add_48 = select_scatter_201 = None
        select_653 = torch.ops.aten.select.int(select_scatter_200, 0, 48)
        select_scatter_202 = torch.ops.aten.select_scatter.default(select_scatter_200, full_default_1, 0, 48);  select_scatter_200 = None
        mul_151 = torch.ops.aten.mul.Tensor(select_653, 2);  select_653 = None
        select_scatter_203 = torch.ops.aten.select_scatter.default(full_default, mul_151, 0, 48);  mul_151 = None
        add_50 = torch.ops.aten.add.Tensor(add_49, select_scatter_203);  add_49 = select_scatter_203 = None
        select_656 = torch.ops.aten.select.int(select_scatter_202, 0, 47)
        select_scatter_204 = torch.ops.aten.select_scatter.default(select_scatter_202, full_default_1, 0, 47);  select_scatter_202 = None
        mul_152 = torch.ops.aten.mul.Tensor(select_656, 2);  select_656 = None
        select_scatter_205 = torch.ops.aten.select_scatter.default(full_default, mul_152, 0, 47);  mul_152 = None
        add_51 = torch.ops.aten.add.Tensor(add_50, select_scatter_205);  add_50 = select_scatter_205 = None
        select_659 = torch.ops.aten.select.int(select_scatter_204, 0, 46)
        select_scatter_206 = torch.ops.aten.select_scatter.default(select_scatter_204, full_default_1, 0, 46);  select_scatter_204 = None
        mul_153 = torch.ops.aten.mul.Tensor(select_659, 2);  select_659 = None
        select_scatter_207 = torch.ops.aten.select_scatter.default(full_default, mul_153, 0, 46);  mul_153 = None
        add_52 = torch.ops.aten.add.Tensor(add_51, select_scatter_207);  add_51 = select_scatter_207 = None
        select_662 = torch.ops.aten.select.int(select_scatter_206, 0, 45)
        select_scatter_208 = torch.ops.aten.select_scatter.default(select_scatter_206, full_default_1, 0, 45);  select_scatter_206 = None
        mul_154 = torch.ops.aten.mul.Tensor(select_662, 2);  select_662 = None
        select_scatter_209 = torch.ops.aten.select_scatter.default(full_default, mul_154, 0, 45);  mul_154 = None
        add_53 = torch.ops.aten.add.Tensor(add_52, select_scatter_209);  add_52 = select_scatter_209 = None
        select_665 = torch.ops.aten.select.int(select_scatter_208, 0, 44)
        select_scatter_210 = torch.ops.aten.select_scatter.default(select_scatter_208, full_default_1, 0, 44);  select_scatter_208 = None
        mul_155 = torch.ops.aten.mul.Tensor(select_665, 2);  select_665 = None
        select_scatter_211 = torch.ops.aten.select_scatter.default(full_default, mul_155, 0, 44);  mul_155 = None
        add_54 = torch.ops.aten.add.Tensor(add_53, select_scatter_211);  add_53 = select_scatter_211 = None
        select_668 = torch.ops.aten.select.int(select_scatter_210, 0, 43)
        select_scatter_212 = torch.ops.aten.select_scatter.default(select_scatter_210, full_default_1, 0, 43);  select_scatter_210 = None
        mul_156 = torch.ops.aten.mul.Tensor(select_668, 2);  select_668 = None
        select_scatter_213 = torch.ops.aten.select_scatter.default(full_default, mul_156, 0, 43);  mul_156 = None
        add_55 = torch.ops.aten.add.Tensor(add_54, select_scatter_213);  add_54 = select_scatter_213 = None
        select_671 = torch.ops.aten.select.int(select_scatter_212, 0, 42)
        select_scatter_214 = torch.ops.aten.select_scatter.default(select_scatter_212, full_default_1, 0, 42);  select_scatter_212 = None
        mul_157 = torch.ops.aten.mul.Tensor(select_671, 2);  select_671 = None
        select_scatter_215 = torch.ops.aten.select_scatter.default(full_default, mul_157, 0, 42);  mul_157 = None
        add_56 = torch.ops.aten.add.Tensor(add_55, select_scatter_215);  add_55 = select_scatter_215 = None
        select_674 = torch.ops.aten.select.int(select_scatter_214, 0, 41)
        select_scatter_216 = torch.ops.aten.select_scatter.default(select_scatter_214, full_default_1, 0, 41);  select_scatter_214 = None
        mul_158 = torch.ops.aten.mul.Tensor(select_674, 2);  select_674 = None
        select_scatter_217 = torch.ops.aten.select_scatter.default(full_default, mul_158, 0, 41);  mul_158 = None
        add_57 = torch.ops.aten.add.Tensor(add_56, select_scatter_217);  add_56 = select_scatter_217 = None
        select_677 = torch.ops.aten.select.int(select_scatter_216, 0, 40)
        select_scatter_218 = torch.ops.aten.select_scatter.default(select_scatter_216, full_default_1, 0, 40);  select_scatter_216 = None
        mul_159 = torch.ops.aten.mul.Tensor(select_677, 2);  select_677 = None
        select_scatter_219 = torch.ops.aten.select_scatter.default(full_default, mul_159, 0, 40);  mul_159 = None
        add_58 = torch.ops.aten.add.Tensor(add_57, select_scatter_219);  add_57 = select_scatter_219 = None
        select_680 = torch.ops.aten.select.int(select_scatter_218, 0, 39)
        select_scatter_220 = torch.ops.aten.select_scatter.default(select_scatter_218, full_default_1, 0, 39);  select_scatter_218 = None
        mul_160 = torch.ops.aten.mul.Tensor(select_680, 2);  select_680 = None
        select_scatter_221 = torch.ops.aten.select_scatter.default(full_default, mul_160, 0, 39);  mul_160 = None
        add_59 = torch.ops.aten.add.Tensor(add_58, select_scatter_221);  add_58 = select_scatter_221 = None
        select_683 = torch.ops.aten.select.int(select_scatter_220, 0, 38)
        select_scatter_222 = torch.ops.aten.select_scatter.default(select_scatter_220, full_default_1, 0, 38);  select_scatter_220 = None
        mul_161 = torch.ops.aten.mul.Tensor(select_683, 2);  select_683 = None
        select_scatter_223 = torch.ops.aten.select_scatter.default(full_default, mul_161, 0, 38);  mul_161 = None
        add_60 = torch.ops.aten.add.Tensor(add_59, select_scatter_223);  add_59 = select_scatter_223 = None
        select_686 = torch.ops.aten.select.int(select_scatter_222, 0, 37)
        select_scatter_224 = torch.ops.aten.select_scatter.default(select_scatter_222, full_default_1, 0, 37);  select_scatter_222 = None
        mul_162 = torch.ops.aten.mul.Tensor(select_686, 2);  select_686 = None
        select_scatter_225 = torch.ops.aten.select_scatter.default(full_default, mul_162, 0, 37);  mul_162 = None
        add_61 = torch.ops.aten.add.Tensor(add_60, select_scatter_225);  add_60 = select_scatter_225 = None
        select_689 = torch.ops.aten.select.int(select_scatter_224, 0, 36)
        select_scatter_226 = torch.ops.aten.select_scatter.default(select_scatter_224, full_default_1, 0, 36);  select_scatter_224 = None
        mul_163 = torch.ops.aten.mul.Tensor(select_689, 2);  select_689 = None
        select_scatter_227 = torch.ops.aten.select_scatter.default(full_default, mul_163, 0, 36);  mul_163 = None
        add_62 = torch.ops.aten.add.Tensor(add_61, select_scatter_227);  add_61 = select_scatter_227 = None
        select_692 = torch.ops.aten.select.int(select_scatter_226, 0, 35)
        select_scatter_228 = torch.ops.aten.select_scatter.default(select_scatter_226, full_default_1, 0, 35);  select_scatter_226 = None
        mul_164 = torch.ops.aten.mul.Tensor(select_692, 2);  select_692 = None
        select_scatter_229 = torch.ops.aten.select_scatter.default(full_default, mul_164, 0, 35);  mul_164 = None
        add_63 = torch.ops.aten.add.Tensor(add_62, select_scatter_229);  add_62 = select_scatter_229 = None
        select_695 = torch.ops.aten.select.int(select_scatter_228, 0, 34)
        select_scatter_230 = torch.ops.aten.select_scatter.default(select_scatter_228, full_default_1, 0, 34);  select_scatter_228 = None
        mul_165 = torch.ops.aten.mul.Tensor(select_695, 2);  select_695 = None
        select_scatter_231 = torch.ops.aten.select_scatter.default(full_default, mul_165, 0, 34);  mul_165 = None
        add_64 = torch.ops.aten.add.Tensor(add_63, select_scatter_231);  add_63 = select_scatter_231 = None
        select_698 = torch.ops.aten.select.int(select_scatter_230, 0, 33)
        select_scatter_232 = torch.ops.aten.select_scatter.default(select_scatter_230, full_default_1, 0, 33);  select_scatter_230 = None
        mul_166 = torch.ops.aten.mul.Tensor(select_698, 2);  select_698 = None
        select_scatter_233 = torch.ops.aten.select_scatter.default(full_default, mul_166, 0, 33);  mul_166 = None
        add_65 = torch.ops.aten.add.Tensor(add_64, select_scatter_233);  add_64 = select_scatter_233 = None
        select_701 = torch.ops.aten.select.int(select_scatter_232, 0, 32)
        select_scatter_234 = torch.ops.aten.select_scatter.default(select_scatter_232, full_default_1, 0, 32);  select_scatter_232 = None
        mul_167 = torch.ops.aten.mul.Tensor(select_701, 2);  select_701 = None
        select_scatter_235 = torch.ops.aten.select_scatter.default(full_default, mul_167, 0, 32);  mul_167 = None
        add_66 = torch.ops.aten.add.Tensor(add_65, select_scatter_235);  add_65 = select_scatter_235 = None
        select_704 = torch.ops.aten.select.int(select_scatter_234, 0, 31)
        select_scatter_236 = torch.ops.aten.select_scatter.default(select_scatter_234, full_default_1, 0, 31);  select_scatter_234 = None
        mul_168 = torch.ops.aten.mul.Tensor(select_704, 2);  select_704 = None
        select_scatter_237 = torch.ops.aten.select_scatter.default(full_default, mul_168, 0, 31);  mul_168 = None
        add_67 = torch.ops.aten.add.Tensor(add_66, select_scatter_237);  add_66 = select_scatter_237 = None
        select_707 = torch.ops.aten.select.int(select_scatter_236, 0, 30)
        select_scatter_238 = torch.ops.aten.select_scatter.default(select_scatter_236, full_default_1, 0, 30);  select_scatter_236 = None
        mul_169 = torch.ops.aten.mul.Tensor(select_707, 2);  select_707 = None
        select_scatter_239 = torch.ops.aten.select_scatter.default(full_default, mul_169, 0, 30);  mul_169 = None
        add_68 = torch.ops.aten.add.Tensor(add_67, select_scatter_239);  add_67 = select_scatter_239 = None
        select_710 = torch.ops.aten.select.int(select_scatter_238, 0, 29)
        select_scatter_240 = torch.ops.aten.select_scatter.default(select_scatter_238, full_default_1, 0, 29);  select_scatter_238 = None
        mul_170 = torch.ops.aten.mul.Tensor(select_710, 2);  select_710 = None
        select_scatter_241 = torch.ops.aten.select_scatter.default(full_default, mul_170, 0, 29);  mul_170 = None
        add_69 = torch.ops.aten.add.Tensor(add_68, select_scatter_241);  add_68 = select_scatter_241 = None
        select_713 = torch.ops.aten.select.int(select_scatter_240, 0, 28)
        select_scatter_242 = torch.ops.aten.select_scatter.default(select_scatter_240, full_default_1, 0, 28);  select_scatter_240 = None
        mul_171 = torch.ops.aten.mul.Tensor(select_713, 2);  select_713 = None
        select_scatter_243 = torch.ops.aten.select_scatter.default(full_default, mul_171, 0, 28);  mul_171 = None
        add_70 = torch.ops.aten.add.Tensor(add_69, select_scatter_243);  add_69 = select_scatter_243 = None
        select_716 = torch.ops.aten.select.int(select_scatter_242, 0, 27)
        select_scatter_244 = torch.ops.aten.select_scatter.default(select_scatter_242, full_default_1, 0, 27);  select_scatter_242 = None
        mul_172 = torch.ops.aten.mul.Tensor(select_716, 2);  select_716 = None
        select_scatter_245 = torch.ops.aten.select_scatter.default(full_default, mul_172, 0, 27);  mul_172 = None
        add_71 = torch.ops.aten.add.Tensor(add_70, select_scatter_245);  add_70 = select_scatter_245 = None
        select_719 = torch.ops.aten.select.int(select_scatter_244, 0, 26)
        select_scatter_246 = torch.ops.aten.select_scatter.default(select_scatter_244, full_default_1, 0, 26);  select_scatter_244 = None
        mul_173 = torch.ops.aten.mul.Tensor(select_719, 2);  select_719 = None
        select_scatter_247 = torch.ops.aten.select_scatter.default(full_default, mul_173, 0, 26);  mul_173 = None
        add_72 = torch.ops.aten.add.Tensor(add_71, select_scatter_247);  add_71 = select_scatter_247 = None
        select_722 = torch.ops.aten.select.int(select_scatter_246, 0, 25)
        select_scatter_248 = torch.ops.aten.select_scatter.default(select_scatter_246, full_default_1, 0, 25);  select_scatter_246 = None
        mul_174 = torch.ops.aten.mul.Tensor(select_722, 2);  select_722 = None
        select_scatter_249 = torch.ops.aten.select_scatter.default(full_default, mul_174, 0, 25);  mul_174 = None
        add_73 = torch.ops.aten.add.Tensor(add_72, select_scatter_249);  add_72 = select_scatter_249 = None
        select_725 = torch.ops.aten.select.int(select_scatter_248, 0, 24)
        select_scatter_250 = torch.ops.aten.select_scatter.default(select_scatter_248, full_default_1, 0, 24);  select_scatter_248 = None
        mul_175 = torch.ops.aten.mul.Tensor(select_725, 2);  select_725 = None
        select_scatter_251 = torch.ops.aten.select_scatter.default(full_default, mul_175, 0, 24);  mul_175 = None
        add_74 = torch.ops.aten.add.Tensor(add_73, select_scatter_251);  add_73 = select_scatter_251 = None
        select_728 = torch.ops.aten.select.int(select_scatter_250, 0, 23)
        select_scatter_252 = torch.ops.aten.select_scatter.default(select_scatter_250, full_default_1, 0, 23);  select_scatter_250 = None
        mul_176 = torch.ops.aten.mul.Tensor(select_728, 2);  select_728 = None
        select_scatter_253 = torch.ops.aten.select_scatter.default(full_default, mul_176, 0, 23);  mul_176 = None
        add_75 = torch.ops.aten.add.Tensor(add_74, select_scatter_253);  add_74 = select_scatter_253 = None
        select_731 = torch.ops.aten.select.int(select_scatter_252, 0, 22)
        select_scatter_254 = torch.ops.aten.select_scatter.default(select_scatter_252, full_default_1, 0, 22);  select_scatter_252 = None
        mul_177 = torch.ops.aten.mul.Tensor(select_731, 2);  select_731 = None
        select_scatter_255 = torch.ops.aten.select_scatter.default(full_default, mul_177, 0, 22);  mul_177 = None
        add_76 = torch.ops.aten.add.Tensor(add_75, select_scatter_255);  add_75 = select_scatter_255 = None
        select_734 = torch.ops.aten.select.int(select_scatter_254, 0, 21)
        select_scatter_256 = torch.ops.aten.select_scatter.default(select_scatter_254, full_default_1, 0, 21);  select_scatter_254 = None
        mul_178 = torch.ops.aten.mul.Tensor(select_734, 2);  select_734 = None
        select_scatter_257 = torch.ops.aten.select_scatter.default(full_default, mul_178, 0, 21);  mul_178 = None
        add_77 = torch.ops.aten.add.Tensor(add_76, select_scatter_257);  add_76 = select_scatter_257 = None
        select_737 = torch.ops.aten.select.int(select_scatter_256, 0, 20)
        select_scatter_258 = torch.ops.aten.select_scatter.default(select_scatter_256, full_default_1, 0, 20);  select_scatter_256 = None
        mul_179 = torch.ops.aten.mul.Tensor(select_737, 2);  select_737 = None
        select_scatter_259 = torch.ops.aten.select_scatter.default(full_default, mul_179, 0, 20);  mul_179 = None
        add_78 = torch.ops.aten.add.Tensor(add_77, select_scatter_259);  add_77 = select_scatter_259 = None
        select_740 = torch.ops.aten.select.int(select_scatter_258, 0, 19)
        select_scatter_260 = torch.ops.aten.select_scatter.default(select_scatter_258, full_default_1, 0, 19);  select_scatter_258 = None
        mul_180 = torch.ops.aten.mul.Tensor(select_740, 2);  select_740 = None
        select_scatter_261 = torch.ops.aten.select_scatter.default(full_default, mul_180, 0, 19);  mul_180 = None
        add_79 = torch.ops.aten.add.Tensor(add_78, select_scatter_261);  add_78 = select_scatter_261 = None
        select_743 = torch.ops.aten.select.int(select_scatter_260, 0, 18)
        select_scatter_262 = torch.ops.aten.select_scatter.default(select_scatter_260, full_default_1, 0, 18);  select_scatter_260 = None
        mul_181 = torch.ops.aten.mul.Tensor(select_743, 2);  select_743 = None
        select_scatter_263 = torch.ops.aten.select_scatter.default(full_default, mul_181, 0, 18);  mul_181 = None
        add_80 = torch.ops.aten.add.Tensor(add_79, select_scatter_263);  add_79 = select_scatter_263 = None
        select_746 = torch.ops.aten.select.int(select_scatter_262, 0, 17)
        select_scatter_264 = torch.ops.aten.select_scatter.default(select_scatter_262, full_default_1, 0, 17);  select_scatter_262 = None
        mul_182 = torch.ops.aten.mul.Tensor(select_746, 2);  select_746 = None
        select_scatter_265 = torch.ops.aten.select_scatter.default(full_default, mul_182, 0, 17);  mul_182 = None
        add_81 = torch.ops.aten.add.Tensor(add_80, select_scatter_265);  add_80 = select_scatter_265 = None
        select_749 = torch.ops.aten.select.int(select_scatter_264, 0, 16)
        select_scatter_266 = torch.ops.aten.select_scatter.default(select_scatter_264, full_default_1, 0, 16);  select_scatter_264 = None
        mul_183 = torch.ops.aten.mul.Tensor(select_749, 2);  select_749 = None
        select_scatter_267 = torch.ops.aten.select_scatter.default(full_default, mul_183, 0, 16);  mul_183 = None
        add_82 = torch.ops.aten.add.Tensor(add_81, select_scatter_267);  add_81 = select_scatter_267 = None
        select_752 = torch.ops.aten.select.int(select_scatter_266, 0, 15)
        select_scatter_268 = torch.ops.aten.select_scatter.default(select_scatter_266, full_default_1, 0, 15);  select_scatter_266 = None
        mul_184 = torch.ops.aten.mul.Tensor(select_752, 2);  select_752 = None
        select_scatter_269 = torch.ops.aten.select_scatter.default(full_default, mul_184, 0, 15);  mul_184 = None
        add_83 = torch.ops.aten.add.Tensor(add_82, select_scatter_269);  add_82 = select_scatter_269 = None
        select_755 = torch.ops.aten.select.int(select_scatter_268, 0, 14)
        select_scatter_270 = torch.ops.aten.select_scatter.default(select_scatter_268, full_default_1, 0, 14);  select_scatter_268 = None
        mul_185 = torch.ops.aten.mul.Tensor(select_755, 2);  select_755 = None
        select_scatter_271 = torch.ops.aten.select_scatter.default(full_default, mul_185, 0, 14);  mul_185 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, select_scatter_271);  add_83 = select_scatter_271 = None
        select_758 = torch.ops.aten.select.int(select_scatter_270, 0, 13)
        select_scatter_272 = torch.ops.aten.select_scatter.default(select_scatter_270, full_default_1, 0, 13);  select_scatter_270 = None
        mul_186 = torch.ops.aten.mul.Tensor(select_758, 2);  select_758 = None
        select_scatter_273 = torch.ops.aten.select_scatter.default(full_default, mul_186, 0, 13);  mul_186 = None
        add_85 = torch.ops.aten.add.Tensor(add_84, select_scatter_273);  add_84 = select_scatter_273 = None
        select_761 = torch.ops.aten.select.int(select_scatter_272, 0, 12)
        select_scatter_274 = torch.ops.aten.select_scatter.default(select_scatter_272, full_default_1, 0, 12);  select_scatter_272 = None
        mul_187 = torch.ops.aten.mul.Tensor(select_761, 2);  select_761 = None
        select_scatter_275 = torch.ops.aten.select_scatter.default(full_default, mul_187, 0, 12);  mul_187 = None
        add_86 = torch.ops.aten.add.Tensor(add_85, select_scatter_275);  add_85 = select_scatter_275 = None
        select_764 = torch.ops.aten.select.int(select_scatter_274, 0, 11)
        select_scatter_276 = torch.ops.aten.select_scatter.default(select_scatter_274, full_default_1, 0, 11);  select_scatter_274 = None
        mul_188 = torch.ops.aten.mul.Tensor(select_764, 2);  select_764 = None
        select_scatter_277 = torch.ops.aten.select_scatter.default(full_default, mul_188, 0, 11);  mul_188 = None
        add_87 = torch.ops.aten.add.Tensor(add_86, select_scatter_277);  add_86 = select_scatter_277 = None
        select_767 = torch.ops.aten.select.int(select_scatter_276, 0, 10)
        select_scatter_278 = torch.ops.aten.select_scatter.default(select_scatter_276, full_default_1, 0, 10);  select_scatter_276 = None
        mul_189 = torch.ops.aten.mul.Tensor(select_767, 2);  select_767 = None
        select_scatter_279 = torch.ops.aten.select_scatter.default(full_default, mul_189, 0, 10);  mul_189 = None
        add_88 = torch.ops.aten.add.Tensor(add_87, select_scatter_279);  add_87 = select_scatter_279 = None
        select_770 = torch.ops.aten.select.int(select_scatter_278, 0, 9)
        select_scatter_280 = torch.ops.aten.select_scatter.default(select_scatter_278, full_default_1, 0, 9);  select_scatter_278 = None
        mul_190 = torch.ops.aten.mul.Tensor(select_770, 2);  select_770 = None
        select_scatter_281 = torch.ops.aten.select_scatter.default(full_default, mul_190, 0, 9);  mul_190 = None
        add_89 = torch.ops.aten.add.Tensor(add_88, select_scatter_281);  add_88 = select_scatter_281 = None
        select_773 = torch.ops.aten.select.int(select_scatter_280, 0, 8)
        select_scatter_282 = torch.ops.aten.select_scatter.default(select_scatter_280, full_default_1, 0, 8);  select_scatter_280 = None
        mul_191 = torch.ops.aten.mul.Tensor(select_773, 2);  select_773 = None
        select_scatter_283 = torch.ops.aten.select_scatter.default(full_default, mul_191, 0, 8);  mul_191 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, select_scatter_283);  add_89 = select_scatter_283 = None
        select_776 = torch.ops.aten.select.int(select_scatter_282, 0, 7)
        select_scatter_284 = torch.ops.aten.select_scatter.default(select_scatter_282, full_default_1, 0, 7);  select_scatter_282 = None
        mul_192 = torch.ops.aten.mul.Tensor(select_776, 2);  select_776 = None
        select_scatter_285 = torch.ops.aten.select_scatter.default(full_default, mul_192, 0, 7);  mul_192 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, select_scatter_285);  add_90 = select_scatter_285 = None
        select_779 = torch.ops.aten.select.int(select_scatter_284, 0, 6)
        select_scatter_286 = torch.ops.aten.select_scatter.default(select_scatter_284, full_default_1, 0, 6);  select_scatter_284 = None
        mul_193 = torch.ops.aten.mul.Tensor(select_779, 2);  select_779 = None
        select_scatter_287 = torch.ops.aten.select_scatter.default(full_default, mul_193, 0, 6);  mul_193 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, select_scatter_287);  add_91 = select_scatter_287 = None
        select_782 = torch.ops.aten.select.int(select_scatter_286, 0, 5)
        select_scatter_288 = torch.ops.aten.select_scatter.default(select_scatter_286, full_default_1, 0, 5);  select_scatter_286 = None
        mul_194 = torch.ops.aten.mul.Tensor(select_782, 2);  select_782 = None
        select_scatter_289 = torch.ops.aten.select_scatter.default(full_default, mul_194, 0, 5);  mul_194 = None
        add_93 = torch.ops.aten.add.Tensor(add_92, select_scatter_289);  add_92 = select_scatter_289 = None
        select_785 = torch.ops.aten.select.int(select_scatter_288, 0, 4)
        select_scatter_290 = torch.ops.aten.select_scatter.default(select_scatter_288, full_default_1, 0, 4);  select_scatter_288 = None
        mul_195 = torch.ops.aten.mul.Tensor(select_785, 2);  select_785 = None
        select_scatter_291 = torch.ops.aten.select_scatter.default(full_default, mul_195, 0, 4);  mul_195 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, select_scatter_291);  add_93 = select_scatter_291 = None
        select_788 = torch.ops.aten.select.int(select_scatter_290, 0, 3)
        select_scatter_292 = torch.ops.aten.select_scatter.default(select_scatter_290, full_default_1, 0, 3);  select_scatter_290 = None
        mul_196 = torch.ops.aten.mul.Tensor(select_788, 2);  select_788 = None
        select_scatter_293 = torch.ops.aten.select_scatter.default(full_default, mul_196, 0, 3);  mul_196 = None
        add_95 = torch.ops.aten.add.Tensor(add_94, select_scatter_293);  add_94 = select_scatter_293 = None
        select_791 = torch.ops.aten.select.int(select_scatter_292, 0, 2)
        select_scatter_294 = torch.ops.aten.select_scatter.default(select_scatter_292, full_default_1, 0, 2);  select_scatter_292 = None
        mul_197 = torch.ops.aten.mul.Tensor(select_791, 2);  select_791 = None
        select_scatter_295 = torch.ops.aten.select_scatter.default(full_default, mul_197, 0, 2);  mul_197 = None
        add_96 = torch.ops.aten.add.Tensor(add_95, select_scatter_295);  add_95 = select_scatter_295 = None
        select_794 = torch.ops.aten.select.int(select_scatter_294, 0, 1)
        select_scatter_296 = torch.ops.aten.select_scatter.default(select_scatter_294, full_default_1, 0, 1);  select_scatter_294 = full_default_1 = None
        mul_198 = torch.ops.aten.mul.Tensor(select_794, 2);  select_794 = None
        select_scatter_297 = torch.ops.aten.select_scatter.default(full_default, mul_198, 0, 1);  mul_198 = None
        add_97 = torch.ops.aten.add.Tensor(add_96, select_scatter_297);  add_96 = select_scatter_297 = None
        select_797 = torch.ops.aten.select.int(select_scatter_296, 0, 0);  select_scatter_296 = None
        mul_199 = torch.ops.aten.mul.Tensor(select_797, 2);  select_797 = None
        select_scatter_298 = torch.ops.aten.select_scatter.default(full_default, mul_199, 0, 0);  full_default = mul_199 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, select_scatter_298);  add_97 = select_scatter_298 = None
        return (add_98,)
        
def load_args(reader):
    buf0 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf0, (100,), is_leaf=True)  # full_default
    buf1 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf1, (100,), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)