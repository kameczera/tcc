class GraphModule(torch.nn.Module):
    def forward(self, full_default: "f32[100]", tangents_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_500: "f32[]" = torch.ops.aten.select.int(tangents_1, 0, 99)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_100: "f32[100]" = torch.ops.aten.select_scatter.default(tangents_1, full_default_1, 0, 99);  tangents_1 = None
        mul_100: "f32[]" = torch.ops.aten.mul.Tensor(select_500, 2);  select_500 = None
        select_scatter_101: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_100, 0, 99);  mul_100 = None
        select_503: "f32[]" = torch.ops.aten.select.int(select_scatter_100, 0, 98)
        select_scatter_102: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_100, full_default_1, 0, 98);  select_scatter_100 = None
        mul_101: "f32[]" = torch.ops.aten.mul.Tensor(select_503, 2);  select_503 = None
        select_scatter_103: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_101, 0, 98);  mul_101 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add: "f32[100]" = torch.ops.aten.add.Tensor(select_scatter_101, select_scatter_103);  select_scatter_101 = select_scatter_103 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_506: "f32[]" = torch.ops.aten.select.int(select_scatter_102, 0, 97)
        select_scatter_104: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_102, full_default_1, 0, 97);  select_scatter_102 = None
        mul_102: "f32[]" = torch.ops.aten.mul.Tensor(select_506, 2);  select_506 = None
        select_scatter_105: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_102, 0, 97);  mul_102 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_1: "f32[100]" = torch.ops.aten.add.Tensor(add, select_scatter_105);  add = select_scatter_105 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_509: "f32[]" = torch.ops.aten.select.int(select_scatter_104, 0, 96)
        select_scatter_106: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_104, full_default_1, 0, 96);  select_scatter_104 = None
        mul_103: "f32[]" = torch.ops.aten.mul.Tensor(select_509, 2);  select_509 = None
        select_scatter_107: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_103, 0, 96);  mul_103 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_2: "f32[100]" = torch.ops.aten.add.Tensor(add_1, select_scatter_107);  add_1 = select_scatter_107 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_512: "f32[]" = torch.ops.aten.select.int(select_scatter_106, 0, 95)
        select_scatter_108: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_106, full_default_1, 0, 95);  select_scatter_106 = None
        mul_104: "f32[]" = torch.ops.aten.mul.Tensor(select_512, 2);  select_512 = None
        select_scatter_109: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_104, 0, 95);  mul_104 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_3: "f32[100]" = torch.ops.aten.add.Tensor(add_2, select_scatter_109);  add_2 = select_scatter_109 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_515: "f32[]" = torch.ops.aten.select.int(select_scatter_108, 0, 94)
        select_scatter_110: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_108, full_default_1, 0, 94);  select_scatter_108 = None
        mul_105: "f32[]" = torch.ops.aten.mul.Tensor(select_515, 2);  select_515 = None
        select_scatter_111: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_105, 0, 94);  mul_105 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_4: "f32[100]" = torch.ops.aten.add.Tensor(add_3, select_scatter_111);  add_3 = select_scatter_111 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_518: "f32[]" = torch.ops.aten.select.int(select_scatter_110, 0, 93)
        select_scatter_112: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_110, full_default_1, 0, 93);  select_scatter_110 = None
        mul_106: "f32[]" = torch.ops.aten.mul.Tensor(select_518, 2);  select_518 = None
        select_scatter_113: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_106, 0, 93);  mul_106 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_5: "f32[100]" = torch.ops.aten.add.Tensor(add_4, select_scatter_113);  add_4 = select_scatter_113 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_521: "f32[]" = torch.ops.aten.select.int(select_scatter_112, 0, 92)
        select_scatter_114: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_112, full_default_1, 0, 92);  select_scatter_112 = None
        mul_107: "f32[]" = torch.ops.aten.mul.Tensor(select_521, 2);  select_521 = None
        select_scatter_115: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_107, 0, 92);  mul_107 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_6: "f32[100]" = torch.ops.aten.add.Tensor(add_5, select_scatter_115);  add_5 = select_scatter_115 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_524: "f32[]" = torch.ops.aten.select.int(select_scatter_114, 0, 91)
        select_scatter_116: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_114, full_default_1, 0, 91);  select_scatter_114 = None
        mul_108: "f32[]" = torch.ops.aten.mul.Tensor(select_524, 2);  select_524 = None
        select_scatter_117: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_108, 0, 91);  mul_108 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_7: "f32[100]" = torch.ops.aten.add.Tensor(add_6, select_scatter_117);  add_6 = select_scatter_117 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_527: "f32[]" = torch.ops.aten.select.int(select_scatter_116, 0, 90)
        select_scatter_118: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_116, full_default_1, 0, 90);  select_scatter_116 = None
        mul_109: "f32[]" = torch.ops.aten.mul.Tensor(select_527, 2);  select_527 = None
        select_scatter_119: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_109, 0, 90);  mul_109 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_8: "f32[100]" = torch.ops.aten.add.Tensor(add_7, select_scatter_119);  add_7 = select_scatter_119 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_530: "f32[]" = torch.ops.aten.select.int(select_scatter_118, 0, 89)
        select_scatter_120: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_118, full_default_1, 0, 89);  select_scatter_118 = None
        mul_110: "f32[]" = torch.ops.aten.mul.Tensor(select_530, 2);  select_530 = None
        select_scatter_121: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_110, 0, 89);  mul_110 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_9: "f32[100]" = torch.ops.aten.add.Tensor(add_8, select_scatter_121);  add_8 = select_scatter_121 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_533: "f32[]" = torch.ops.aten.select.int(select_scatter_120, 0, 88)
        select_scatter_122: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_120, full_default_1, 0, 88);  select_scatter_120 = None
        mul_111: "f32[]" = torch.ops.aten.mul.Tensor(select_533, 2);  select_533 = None
        select_scatter_123: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_111, 0, 88);  mul_111 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_10: "f32[100]" = torch.ops.aten.add.Tensor(add_9, select_scatter_123);  add_9 = select_scatter_123 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_536: "f32[]" = torch.ops.aten.select.int(select_scatter_122, 0, 87)
        select_scatter_124: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_122, full_default_1, 0, 87);  select_scatter_122 = None
        mul_112: "f32[]" = torch.ops.aten.mul.Tensor(select_536, 2);  select_536 = None
        select_scatter_125: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_112, 0, 87);  mul_112 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_11: "f32[100]" = torch.ops.aten.add.Tensor(add_10, select_scatter_125);  add_10 = select_scatter_125 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_539: "f32[]" = torch.ops.aten.select.int(select_scatter_124, 0, 86)
        select_scatter_126: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_124, full_default_1, 0, 86);  select_scatter_124 = None
        mul_113: "f32[]" = torch.ops.aten.mul.Tensor(select_539, 2);  select_539 = None
        select_scatter_127: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_113, 0, 86);  mul_113 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_12: "f32[100]" = torch.ops.aten.add.Tensor(add_11, select_scatter_127);  add_11 = select_scatter_127 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_542: "f32[]" = torch.ops.aten.select.int(select_scatter_126, 0, 85)
        select_scatter_128: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_126, full_default_1, 0, 85);  select_scatter_126 = None
        mul_114: "f32[]" = torch.ops.aten.mul.Tensor(select_542, 2);  select_542 = None
        select_scatter_129: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_114, 0, 85);  mul_114 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_13: "f32[100]" = torch.ops.aten.add.Tensor(add_12, select_scatter_129);  add_12 = select_scatter_129 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_545: "f32[]" = torch.ops.aten.select.int(select_scatter_128, 0, 84)
        select_scatter_130: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_128, full_default_1, 0, 84);  select_scatter_128 = None
        mul_115: "f32[]" = torch.ops.aten.mul.Tensor(select_545, 2);  select_545 = None
        select_scatter_131: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_115, 0, 84);  mul_115 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_14: "f32[100]" = torch.ops.aten.add.Tensor(add_13, select_scatter_131);  add_13 = select_scatter_131 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_548: "f32[]" = torch.ops.aten.select.int(select_scatter_130, 0, 83)
        select_scatter_132: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_130, full_default_1, 0, 83);  select_scatter_130 = None
        mul_116: "f32[]" = torch.ops.aten.mul.Tensor(select_548, 2);  select_548 = None
        select_scatter_133: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_116, 0, 83);  mul_116 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_15: "f32[100]" = torch.ops.aten.add.Tensor(add_14, select_scatter_133);  add_14 = select_scatter_133 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_551: "f32[]" = torch.ops.aten.select.int(select_scatter_132, 0, 82)
        select_scatter_134: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_132, full_default_1, 0, 82);  select_scatter_132 = None
        mul_117: "f32[]" = torch.ops.aten.mul.Tensor(select_551, 2);  select_551 = None
        select_scatter_135: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_117, 0, 82);  mul_117 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_16: "f32[100]" = torch.ops.aten.add.Tensor(add_15, select_scatter_135);  add_15 = select_scatter_135 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_554: "f32[]" = torch.ops.aten.select.int(select_scatter_134, 0, 81)
        select_scatter_136: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_134, full_default_1, 0, 81);  select_scatter_134 = None
        mul_118: "f32[]" = torch.ops.aten.mul.Tensor(select_554, 2);  select_554 = None
        select_scatter_137: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_118, 0, 81);  mul_118 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_17: "f32[100]" = torch.ops.aten.add.Tensor(add_16, select_scatter_137);  add_16 = select_scatter_137 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_557: "f32[]" = torch.ops.aten.select.int(select_scatter_136, 0, 80)
        select_scatter_138: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_136, full_default_1, 0, 80);  select_scatter_136 = None
        mul_119: "f32[]" = torch.ops.aten.mul.Tensor(select_557, 2);  select_557 = None
        select_scatter_139: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_119, 0, 80);  mul_119 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_18: "f32[100]" = torch.ops.aten.add.Tensor(add_17, select_scatter_139);  add_17 = select_scatter_139 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_560: "f32[]" = torch.ops.aten.select.int(select_scatter_138, 0, 79)
        select_scatter_140: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_138, full_default_1, 0, 79);  select_scatter_138 = None
        mul_120: "f32[]" = torch.ops.aten.mul.Tensor(select_560, 2);  select_560 = None
        select_scatter_141: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_120, 0, 79);  mul_120 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_19: "f32[100]" = torch.ops.aten.add.Tensor(add_18, select_scatter_141);  add_18 = select_scatter_141 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_563: "f32[]" = torch.ops.aten.select.int(select_scatter_140, 0, 78)
        select_scatter_142: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_140, full_default_1, 0, 78);  select_scatter_140 = None
        mul_121: "f32[]" = torch.ops.aten.mul.Tensor(select_563, 2);  select_563 = None
        select_scatter_143: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_121, 0, 78);  mul_121 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_20: "f32[100]" = torch.ops.aten.add.Tensor(add_19, select_scatter_143);  add_19 = select_scatter_143 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_566: "f32[]" = torch.ops.aten.select.int(select_scatter_142, 0, 77)
        select_scatter_144: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_142, full_default_1, 0, 77);  select_scatter_142 = None
        mul_122: "f32[]" = torch.ops.aten.mul.Tensor(select_566, 2);  select_566 = None
        select_scatter_145: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_122, 0, 77);  mul_122 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_21: "f32[100]" = torch.ops.aten.add.Tensor(add_20, select_scatter_145);  add_20 = select_scatter_145 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_569: "f32[]" = torch.ops.aten.select.int(select_scatter_144, 0, 76)
        select_scatter_146: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_144, full_default_1, 0, 76);  select_scatter_144 = None
        mul_123: "f32[]" = torch.ops.aten.mul.Tensor(select_569, 2);  select_569 = None
        select_scatter_147: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_123, 0, 76);  mul_123 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_22: "f32[100]" = torch.ops.aten.add.Tensor(add_21, select_scatter_147);  add_21 = select_scatter_147 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_572: "f32[]" = torch.ops.aten.select.int(select_scatter_146, 0, 75)
        select_scatter_148: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_146, full_default_1, 0, 75);  select_scatter_146 = None
        mul_124: "f32[]" = torch.ops.aten.mul.Tensor(select_572, 2);  select_572 = None
        select_scatter_149: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_124, 0, 75);  mul_124 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_23: "f32[100]" = torch.ops.aten.add.Tensor(add_22, select_scatter_149);  add_22 = select_scatter_149 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_575: "f32[]" = torch.ops.aten.select.int(select_scatter_148, 0, 74)
        select_scatter_150: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_148, full_default_1, 0, 74);  select_scatter_148 = None
        mul_125: "f32[]" = torch.ops.aten.mul.Tensor(select_575, 2);  select_575 = None
        select_scatter_151: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_125, 0, 74);  mul_125 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_24: "f32[100]" = torch.ops.aten.add.Tensor(add_23, select_scatter_151);  add_23 = select_scatter_151 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_578: "f32[]" = torch.ops.aten.select.int(select_scatter_150, 0, 73)
        select_scatter_152: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_150, full_default_1, 0, 73);  select_scatter_150 = None
        mul_126: "f32[]" = torch.ops.aten.mul.Tensor(select_578, 2);  select_578 = None
        select_scatter_153: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_126, 0, 73);  mul_126 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_25: "f32[100]" = torch.ops.aten.add.Tensor(add_24, select_scatter_153);  add_24 = select_scatter_153 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_581: "f32[]" = torch.ops.aten.select.int(select_scatter_152, 0, 72)
        select_scatter_154: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_152, full_default_1, 0, 72);  select_scatter_152 = None
        mul_127: "f32[]" = torch.ops.aten.mul.Tensor(select_581, 2);  select_581 = None
        select_scatter_155: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_127, 0, 72);  mul_127 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_26: "f32[100]" = torch.ops.aten.add.Tensor(add_25, select_scatter_155);  add_25 = select_scatter_155 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_584: "f32[]" = torch.ops.aten.select.int(select_scatter_154, 0, 71)
        select_scatter_156: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_154, full_default_1, 0, 71);  select_scatter_154 = None
        mul_128: "f32[]" = torch.ops.aten.mul.Tensor(select_584, 2);  select_584 = None
        select_scatter_157: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_128, 0, 71);  mul_128 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_27: "f32[100]" = torch.ops.aten.add.Tensor(add_26, select_scatter_157);  add_26 = select_scatter_157 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_587: "f32[]" = torch.ops.aten.select.int(select_scatter_156, 0, 70)
        select_scatter_158: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_156, full_default_1, 0, 70);  select_scatter_156 = None
        mul_129: "f32[]" = torch.ops.aten.mul.Tensor(select_587, 2);  select_587 = None
        select_scatter_159: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_129, 0, 70);  mul_129 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_28: "f32[100]" = torch.ops.aten.add.Tensor(add_27, select_scatter_159);  add_27 = select_scatter_159 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_590: "f32[]" = torch.ops.aten.select.int(select_scatter_158, 0, 69)
        select_scatter_160: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_158, full_default_1, 0, 69);  select_scatter_158 = None
        mul_130: "f32[]" = torch.ops.aten.mul.Tensor(select_590, 2);  select_590 = None
        select_scatter_161: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_130, 0, 69);  mul_130 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_29: "f32[100]" = torch.ops.aten.add.Tensor(add_28, select_scatter_161);  add_28 = select_scatter_161 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_593: "f32[]" = torch.ops.aten.select.int(select_scatter_160, 0, 68)
        select_scatter_162: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_160, full_default_1, 0, 68);  select_scatter_160 = None
        mul_131: "f32[]" = torch.ops.aten.mul.Tensor(select_593, 2);  select_593 = None
        select_scatter_163: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_131, 0, 68);  mul_131 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_30: "f32[100]" = torch.ops.aten.add.Tensor(add_29, select_scatter_163);  add_29 = select_scatter_163 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_596: "f32[]" = torch.ops.aten.select.int(select_scatter_162, 0, 67)
        select_scatter_164: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_162, full_default_1, 0, 67);  select_scatter_162 = None
        mul_132: "f32[]" = torch.ops.aten.mul.Tensor(select_596, 2);  select_596 = None
        select_scatter_165: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_132, 0, 67);  mul_132 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_31: "f32[100]" = torch.ops.aten.add.Tensor(add_30, select_scatter_165);  add_30 = select_scatter_165 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_599: "f32[]" = torch.ops.aten.select.int(select_scatter_164, 0, 66)
        select_scatter_166: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_164, full_default_1, 0, 66);  select_scatter_164 = None
        mul_133: "f32[]" = torch.ops.aten.mul.Tensor(select_599, 2);  select_599 = None
        select_scatter_167: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_133, 0, 66);  mul_133 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_32: "f32[100]" = torch.ops.aten.add.Tensor(add_31, select_scatter_167);  add_31 = select_scatter_167 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_602: "f32[]" = torch.ops.aten.select.int(select_scatter_166, 0, 65)
        select_scatter_168: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_166, full_default_1, 0, 65);  select_scatter_166 = None
        mul_134: "f32[]" = torch.ops.aten.mul.Tensor(select_602, 2);  select_602 = None
        select_scatter_169: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_134, 0, 65);  mul_134 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_33: "f32[100]" = torch.ops.aten.add.Tensor(add_32, select_scatter_169);  add_32 = select_scatter_169 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_605: "f32[]" = torch.ops.aten.select.int(select_scatter_168, 0, 64)
        select_scatter_170: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_168, full_default_1, 0, 64);  select_scatter_168 = None
        mul_135: "f32[]" = torch.ops.aten.mul.Tensor(select_605, 2);  select_605 = None
        select_scatter_171: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_135, 0, 64);  mul_135 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_34: "f32[100]" = torch.ops.aten.add.Tensor(add_33, select_scatter_171);  add_33 = select_scatter_171 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_608: "f32[]" = torch.ops.aten.select.int(select_scatter_170, 0, 63)
        select_scatter_172: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_170, full_default_1, 0, 63);  select_scatter_170 = None
        mul_136: "f32[]" = torch.ops.aten.mul.Tensor(select_608, 2);  select_608 = None
        select_scatter_173: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_136, 0, 63);  mul_136 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_35: "f32[100]" = torch.ops.aten.add.Tensor(add_34, select_scatter_173);  add_34 = select_scatter_173 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_611: "f32[]" = torch.ops.aten.select.int(select_scatter_172, 0, 62)
        select_scatter_174: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_172, full_default_1, 0, 62);  select_scatter_172 = None
        mul_137: "f32[]" = torch.ops.aten.mul.Tensor(select_611, 2);  select_611 = None
        select_scatter_175: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_137, 0, 62);  mul_137 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_36: "f32[100]" = torch.ops.aten.add.Tensor(add_35, select_scatter_175);  add_35 = select_scatter_175 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_614: "f32[]" = torch.ops.aten.select.int(select_scatter_174, 0, 61)
        select_scatter_176: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_174, full_default_1, 0, 61);  select_scatter_174 = None
        mul_138: "f32[]" = torch.ops.aten.mul.Tensor(select_614, 2);  select_614 = None
        select_scatter_177: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_138, 0, 61);  mul_138 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_37: "f32[100]" = torch.ops.aten.add.Tensor(add_36, select_scatter_177);  add_36 = select_scatter_177 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_617: "f32[]" = torch.ops.aten.select.int(select_scatter_176, 0, 60)
        select_scatter_178: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_176, full_default_1, 0, 60);  select_scatter_176 = None
        mul_139: "f32[]" = torch.ops.aten.mul.Tensor(select_617, 2);  select_617 = None
        select_scatter_179: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_139, 0, 60);  mul_139 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_38: "f32[100]" = torch.ops.aten.add.Tensor(add_37, select_scatter_179);  add_37 = select_scatter_179 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_620: "f32[]" = torch.ops.aten.select.int(select_scatter_178, 0, 59)
        select_scatter_180: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_178, full_default_1, 0, 59);  select_scatter_178 = None
        mul_140: "f32[]" = torch.ops.aten.mul.Tensor(select_620, 2);  select_620 = None
        select_scatter_181: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_140, 0, 59);  mul_140 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_39: "f32[100]" = torch.ops.aten.add.Tensor(add_38, select_scatter_181);  add_38 = select_scatter_181 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_623: "f32[]" = torch.ops.aten.select.int(select_scatter_180, 0, 58)
        select_scatter_182: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_180, full_default_1, 0, 58);  select_scatter_180 = None
        mul_141: "f32[]" = torch.ops.aten.mul.Tensor(select_623, 2);  select_623 = None
        select_scatter_183: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_141, 0, 58);  mul_141 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_40: "f32[100]" = torch.ops.aten.add.Tensor(add_39, select_scatter_183);  add_39 = select_scatter_183 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_626: "f32[]" = torch.ops.aten.select.int(select_scatter_182, 0, 57)
        select_scatter_184: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_182, full_default_1, 0, 57);  select_scatter_182 = None
        mul_142: "f32[]" = torch.ops.aten.mul.Tensor(select_626, 2);  select_626 = None
        select_scatter_185: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_142, 0, 57);  mul_142 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_41: "f32[100]" = torch.ops.aten.add.Tensor(add_40, select_scatter_185);  add_40 = select_scatter_185 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_629: "f32[]" = torch.ops.aten.select.int(select_scatter_184, 0, 56)
        select_scatter_186: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_184, full_default_1, 0, 56);  select_scatter_184 = None
        mul_143: "f32[]" = torch.ops.aten.mul.Tensor(select_629, 2);  select_629 = None
        select_scatter_187: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_143, 0, 56);  mul_143 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_42: "f32[100]" = torch.ops.aten.add.Tensor(add_41, select_scatter_187);  add_41 = select_scatter_187 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_632: "f32[]" = torch.ops.aten.select.int(select_scatter_186, 0, 55)
        select_scatter_188: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_186, full_default_1, 0, 55);  select_scatter_186 = None
        mul_144: "f32[]" = torch.ops.aten.mul.Tensor(select_632, 2);  select_632 = None
        select_scatter_189: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_144, 0, 55);  mul_144 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_43: "f32[100]" = torch.ops.aten.add.Tensor(add_42, select_scatter_189);  add_42 = select_scatter_189 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_635: "f32[]" = torch.ops.aten.select.int(select_scatter_188, 0, 54)
        select_scatter_190: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_188, full_default_1, 0, 54);  select_scatter_188 = None
        mul_145: "f32[]" = torch.ops.aten.mul.Tensor(select_635, 2);  select_635 = None
        select_scatter_191: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_145, 0, 54);  mul_145 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_44: "f32[100]" = torch.ops.aten.add.Tensor(add_43, select_scatter_191);  add_43 = select_scatter_191 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_638: "f32[]" = torch.ops.aten.select.int(select_scatter_190, 0, 53)
        select_scatter_192: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_190, full_default_1, 0, 53);  select_scatter_190 = None
        mul_146: "f32[]" = torch.ops.aten.mul.Tensor(select_638, 2);  select_638 = None
        select_scatter_193: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_146, 0, 53);  mul_146 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_45: "f32[100]" = torch.ops.aten.add.Tensor(add_44, select_scatter_193);  add_44 = select_scatter_193 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_641: "f32[]" = torch.ops.aten.select.int(select_scatter_192, 0, 52)
        select_scatter_194: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_192, full_default_1, 0, 52);  select_scatter_192 = None
        mul_147: "f32[]" = torch.ops.aten.mul.Tensor(select_641, 2);  select_641 = None
        select_scatter_195: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_147, 0, 52);  mul_147 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_46: "f32[100]" = torch.ops.aten.add.Tensor(add_45, select_scatter_195);  add_45 = select_scatter_195 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_644: "f32[]" = torch.ops.aten.select.int(select_scatter_194, 0, 51)
        select_scatter_196: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_194, full_default_1, 0, 51);  select_scatter_194 = None
        mul_148: "f32[]" = torch.ops.aten.mul.Tensor(select_644, 2);  select_644 = None
        select_scatter_197: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_148, 0, 51);  mul_148 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_47: "f32[100]" = torch.ops.aten.add.Tensor(add_46, select_scatter_197);  add_46 = select_scatter_197 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_647: "f32[]" = torch.ops.aten.select.int(select_scatter_196, 0, 50)
        select_scatter_198: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_196, full_default_1, 0, 50);  select_scatter_196 = None
        mul_149: "f32[]" = torch.ops.aten.mul.Tensor(select_647, 2);  select_647 = None
        select_scatter_199: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_149, 0, 50);  mul_149 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_48: "f32[100]" = torch.ops.aten.add.Tensor(add_47, select_scatter_199);  add_47 = select_scatter_199 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_650: "f32[]" = torch.ops.aten.select.int(select_scatter_198, 0, 49)
        select_scatter_200: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_198, full_default_1, 0, 49);  select_scatter_198 = None
        mul_150: "f32[]" = torch.ops.aten.mul.Tensor(select_650, 2);  select_650 = None
        select_scatter_201: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_150, 0, 49);  mul_150 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_49: "f32[100]" = torch.ops.aten.add.Tensor(add_48, select_scatter_201);  add_48 = select_scatter_201 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_653: "f32[]" = torch.ops.aten.select.int(select_scatter_200, 0, 48)
        select_scatter_202: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_200, full_default_1, 0, 48);  select_scatter_200 = None
        mul_151: "f32[]" = torch.ops.aten.mul.Tensor(select_653, 2);  select_653 = None
        select_scatter_203: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_151, 0, 48);  mul_151 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_50: "f32[100]" = torch.ops.aten.add.Tensor(add_49, select_scatter_203);  add_49 = select_scatter_203 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_656: "f32[]" = torch.ops.aten.select.int(select_scatter_202, 0, 47)
        select_scatter_204: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_202, full_default_1, 0, 47);  select_scatter_202 = None
        mul_152: "f32[]" = torch.ops.aten.mul.Tensor(select_656, 2);  select_656 = None
        select_scatter_205: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_152, 0, 47);  mul_152 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_51: "f32[100]" = torch.ops.aten.add.Tensor(add_50, select_scatter_205);  add_50 = select_scatter_205 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_659: "f32[]" = torch.ops.aten.select.int(select_scatter_204, 0, 46)
        select_scatter_206: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_204, full_default_1, 0, 46);  select_scatter_204 = None
        mul_153: "f32[]" = torch.ops.aten.mul.Tensor(select_659, 2);  select_659 = None
        select_scatter_207: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_153, 0, 46);  mul_153 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_52: "f32[100]" = torch.ops.aten.add.Tensor(add_51, select_scatter_207);  add_51 = select_scatter_207 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_662: "f32[]" = torch.ops.aten.select.int(select_scatter_206, 0, 45)
        select_scatter_208: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_206, full_default_1, 0, 45);  select_scatter_206 = None
        mul_154: "f32[]" = torch.ops.aten.mul.Tensor(select_662, 2);  select_662 = None
        select_scatter_209: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_154, 0, 45);  mul_154 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_53: "f32[100]" = torch.ops.aten.add.Tensor(add_52, select_scatter_209);  add_52 = select_scatter_209 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_665: "f32[]" = torch.ops.aten.select.int(select_scatter_208, 0, 44)
        select_scatter_210: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_208, full_default_1, 0, 44);  select_scatter_208 = None
        mul_155: "f32[]" = torch.ops.aten.mul.Tensor(select_665, 2);  select_665 = None
        select_scatter_211: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_155, 0, 44);  mul_155 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_54: "f32[100]" = torch.ops.aten.add.Tensor(add_53, select_scatter_211);  add_53 = select_scatter_211 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_668: "f32[]" = torch.ops.aten.select.int(select_scatter_210, 0, 43)
        select_scatter_212: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_210, full_default_1, 0, 43);  select_scatter_210 = None
        mul_156: "f32[]" = torch.ops.aten.mul.Tensor(select_668, 2);  select_668 = None
        select_scatter_213: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_156, 0, 43);  mul_156 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_55: "f32[100]" = torch.ops.aten.add.Tensor(add_54, select_scatter_213);  add_54 = select_scatter_213 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_671: "f32[]" = torch.ops.aten.select.int(select_scatter_212, 0, 42)
        select_scatter_214: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_212, full_default_1, 0, 42);  select_scatter_212 = None
        mul_157: "f32[]" = torch.ops.aten.mul.Tensor(select_671, 2);  select_671 = None
        select_scatter_215: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_157, 0, 42);  mul_157 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_56: "f32[100]" = torch.ops.aten.add.Tensor(add_55, select_scatter_215);  add_55 = select_scatter_215 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_674: "f32[]" = torch.ops.aten.select.int(select_scatter_214, 0, 41)
        select_scatter_216: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_214, full_default_1, 0, 41);  select_scatter_214 = None
        mul_158: "f32[]" = torch.ops.aten.mul.Tensor(select_674, 2);  select_674 = None
        select_scatter_217: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_158, 0, 41);  mul_158 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_57: "f32[100]" = torch.ops.aten.add.Tensor(add_56, select_scatter_217);  add_56 = select_scatter_217 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_677: "f32[]" = torch.ops.aten.select.int(select_scatter_216, 0, 40)
        select_scatter_218: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_216, full_default_1, 0, 40);  select_scatter_216 = None
        mul_159: "f32[]" = torch.ops.aten.mul.Tensor(select_677, 2);  select_677 = None
        select_scatter_219: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_159, 0, 40);  mul_159 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_58: "f32[100]" = torch.ops.aten.add.Tensor(add_57, select_scatter_219);  add_57 = select_scatter_219 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_680: "f32[]" = torch.ops.aten.select.int(select_scatter_218, 0, 39)
        select_scatter_220: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_218, full_default_1, 0, 39);  select_scatter_218 = None
        mul_160: "f32[]" = torch.ops.aten.mul.Tensor(select_680, 2);  select_680 = None
        select_scatter_221: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_160, 0, 39);  mul_160 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_59: "f32[100]" = torch.ops.aten.add.Tensor(add_58, select_scatter_221);  add_58 = select_scatter_221 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_683: "f32[]" = torch.ops.aten.select.int(select_scatter_220, 0, 38)
        select_scatter_222: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_220, full_default_1, 0, 38);  select_scatter_220 = None
        mul_161: "f32[]" = torch.ops.aten.mul.Tensor(select_683, 2);  select_683 = None
        select_scatter_223: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_161, 0, 38);  mul_161 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_60: "f32[100]" = torch.ops.aten.add.Tensor(add_59, select_scatter_223);  add_59 = select_scatter_223 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_686: "f32[]" = torch.ops.aten.select.int(select_scatter_222, 0, 37)
        select_scatter_224: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_222, full_default_1, 0, 37);  select_scatter_222 = None
        mul_162: "f32[]" = torch.ops.aten.mul.Tensor(select_686, 2);  select_686 = None
        select_scatter_225: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_162, 0, 37);  mul_162 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_61: "f32[100]" = torch.ops.aten.add.Tensor(add_60, select_scatter_225);  add_60 = select_scatter_225 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_689: "f32[]" = torch.ops.aten.select.int(select_scatter_224, 0, 36)
        select_scatter_226: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_224, full_default_1, 0, 36);  select_scatter_224 = None
        mul_163: "f32[]" = torch.ops.aten.mul.Tensor(select_689, 2);  select_689 = None
        select_scatter_227: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_163, 0, 36);  mul_163 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_62: "f32[100]" = torch.ops.aten.add.Tensor(add_61, select_scatter_227);  add_61 = select_scatter_227 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_692: "f32[]" = torch.ops.aten.select.int(select_scatter_226, 0, 35)
        select_scatter_228: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_226, full_default_1, 0, 35);  select_scatter_226 = None
        mul_164: "f32[]" = torch.ops.aten.mul.Tensor(select_692, 2);  select_692 = None
        select_scatter_229: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_164, 0, 35);  mul_164 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_63: "f32[100]" = torch.ops.aten.add.Tensor(add_62, select_scatter_229);  add_62 = select_scatter_229 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_695: "f32[]" = torch.ops.aten.select.int(select_scatter_228, 0, 34)
        select_scatter_230: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_228, full_default_1, 0, 34);  select_scatter_228 = None
        mul_165: "f32[]" = torch.ops.aten.mul.Tensor(select_695, 2);  select_695 = None
        select_scatter_231: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_165, 0, 34);  mul_165 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_64: "f32[100]" = torch.ops.aten.add.Tensor(add_63, select_scatter_231);  add_63 = select_scatter_231 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_698: "f32[]" = torch.ops.aten.select.int(select_scatter_230, 0, 33)
        select_scatter_232: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_230, full_default_1, 0, 33);  select_scatter_230 = None
        mul_166: "f32[]" = torch.ops.aten.mul.Tensor(select_698, 2);  select_698 = None
        select_scatter_233: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_166, 0, 33);  mul_166 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_65: "f32[100]" = torch.ops.aten.add.Tensor(add_64, select_scatter_233);  add_64 = select_scatter_233 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_701: "f32[]" = torch.ops.aten.select.int(select_scatter_232, 0, 32)
        select_scatter_234: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_232, full_default_1, 0, 32);  select_scatter_232 = None
        mul_167: "f32[]" = torch.ops.aten.mul.Tensor(select_701, 2);  select_701 = None
        select_scatter_235: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_167, 0, 32);  mul_167 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_66: "f32[100]" = torch.ops.aten.add.Tensor(add_65, select_scatter_235);  add_65 = select_scatter_235 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_704: "f32[]" = torch.ops.aten.select.int(select_scatter_234, 0, 31)
        select_scatter_236: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_234, full_default_1, 0, 31);  select_scatter_234 = None
        mul_168: "f32[]" = torch.ops.aten.mul.Tensor(select_704, 2);  select_704 = None
        select_scatter_237: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_168, 0, 31);  mul_168 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_67: "f32[100]" = torch.ops.aten.add.Tensor(add_66, select_scatter_237);  add_66 = select_scatter_237 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_707: "f32[]" = torch.ops.aten.select.int(select_scatter_236, 0, 30)
        select_scatter_238: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_236, full_default_1, 0, 30);  select_scatter_236 = None
        mul_169: "f32[]" = torch.ops.aten.mul.Tensor(select_707, 2);  select_707 = None
        select_scatter_239: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_169, 0, 30);  mul_169 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_68: "f32[100]" = torch.ops.aten.add.Tensor(add_67, select_scatter_239);  add_67 = select_scatter_239 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_710: "f32[]" = torch.ops.aten.select.int(select_scatter_238, 0, 29)
        select_scatter_240: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_238, full_default_1, 0, 29);  select_scatter_238 = None
        mul_170: "f32[]" = torch.ops.aten.mul.Tensor(select_710, 2);  select_710 = None
        select_scatter_241: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_170, 0, 29);  mul_170 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_69: "f32[100]" = torch.ops.aten.add.Tensor(add_68, select_scatter_241);  add_68 = select_scatter_241 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_713: "f32[]" = torch.ops.aten.select.int(select_scatter_240, 0, 28)
        select_scatter_242: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_240, full_default_1, 0, 28);  select_scatter_240 = None
        mul_171: "f32[]" = torch.ops.aten.mul.Tensor(select_713, 2);  select_713 = None
        select_scatter_243: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_171, 0, 28);  mul_171 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_70: "f32[100]" = torch.ops.aten.add.Tensor(add_69, select_scatter_243);  add_69 = select_scatter_243 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_716: "f32[]" = torch.ops.aten.select.int(select_scatter_242, 0, 27)
        select_scatter_244: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_242, full_default_1, 0, 27);  select_scatter_242 = None
        mul_172: "f32[]" = torch.ops.aten.mul.Tensor(select_716, 2);  select_716 = None
        select_scatter_245: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_172, 0, 27);  mul_172 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_71: "f32[100]" = torch.ops.aten.add.Tensor(add_70, select_scatter_245);  add_70 = select_scatter_245 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_719: "f32[]" = torch.ops.aten.select.int(select_scatter_244, 0, 26)
        select_scatter_246: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_244, full_default_1, 0, 26);  select_scatter_244 = None
        mul_173: "f32[]" = torch.ops.aten.mul.Tensor(select_719, 2);  select_719 = None
        select_scatter_247: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_173, 0, 26);  mul_173 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_72: "f32[100]" = torch.ops.aten.add.Tensor(add_71, select_scatter_247);  add_71 = select_scatter_247 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_722: "f32[]" = torch.ops.aten.select.int(select_scatter_246, 0, 25)
        select_scatter_248: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_246, full_default_1, 0, 25);  select_scatter_246 = None
        mul_174: "f32[]" = torch.ops.aten.mul.Tensor(select_722, 2);  select_722 = None
        select_scatter_249: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_174, 0, 25);  mul_174 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_73: "f32[100]" = torch.ops.aten.add.Tensor(add_72, select_scatter_249);  add_72 = select_scatter_249 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_725: "f32[]" = torch.ops.aten.select.int(select_scatter_248, 0, 24)
        select_scatter_250: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_248, full_default_1, 0, 24);  select_scatter_248 = None
        mul_175: "f32[]" = torch.ops.aten.mul.Tensor(select_725, 2);  select_725 = None
        select_scatter_251: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_175, 0, 24);  mul_175 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_74: "f32[100]" = torch.ops.aten.add.Tensor(add_73, select_scatter_251);  add_73 = select_scatter_251 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_728: "f32[]" = torch.ops.aten.select.int(select_scatter_250, 0, 23)
        select_scatter_252: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_250, full_default_1, 0, 23);  select_scatter_250 = None
        mul_176: "f32[]" = torch.ops.aten.mul.Tensor(select_728, 2);  select_728 = None
        select_scatter_253: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_176, 0, 23);  mul_176 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_75: "f32[100]" = torch.ops.aten.add.Tensor(add_74, select_scatter_253);  add_74 = select_scatter_253 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_731: "f32[]" = torch.ops.aten.select.int(select_scatter_252, 0, 22)
        select_scatter_254: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_252, full_default_1, 0, 22);  select_scatter_252 = None
        mul_177: "f32[]" = torch.ops.aten.mul.Tensor(select_731, 2);  select_731 = None
        select_scatter_255: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_177, 0, 22);  mul_177 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_76: "f32[100]" = torch.ops.aten.add.Tensor(add_75, select_scatter_255);  add_75 = select_scatter_255 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_734: "f32[]" = torch.ops.aten.select.int(select_scatter_254, 0, 21)
        select_scatter_256: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_254, full_default_1, 0, 21);  select_scatter_254 = None
        mul_178: "f32[]" = torch.ops.aten.mul.Tensor(select_734, 2);  select_734 = None
        select_scatter_257: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_178, 0, 21);  mul_178 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_77: "f32[100]" = torch.ops.aten.add.Tensor(add_76, select_scatter_257);  add_76 = select_scatter_257 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_737: "f32[]" = torch.ops.aten.select.int(select_scatter_256, 0, 20)
        select_scatter_258: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_256, full_default_1, 0, 20);  select_scatter_256 = None
        mul_179: "f32[]" = torch.ops.aten.mul.Tensor(select_737, 2);  select_737 = None
        select_scatter_259: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_179, 0, 20);  mul_179 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_78: "f32[100]" = torch.ops.aten.add.Tensor(add_77, select_scatter_259);  add_77 = select_scatter_259 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_740: "f32[]" = torch.ops.aten.select.int(select_scatter_258, 0, 19)
        select_scatter_260: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_258, full_default_1, 0, 19);  select_scatter_258 = None
        mul_180: "f32[]" = torch.ops.aten.mul.Tensor(select_740, 2);  select_740 = None
        select_scatter_261: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_180, 0, 19);  mul_180 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_79: "f32[100]" = torch.ops.aten.add.Tensor(add_78, select_scatter_261);  add_78 = select_scatter_261 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_743: "f32[]" = torch.ops.aten.select.int(select_scatter_260, 0, 18)
        select_scatter_262: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_260, full_default_1, 0, 18);  select_scatter_260 = None
        mul_181: "f32[]" = torch.ops.aten.mul.Tensor(select_743, 2);  select_743 = None
        select_scatter_263: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_181, 0, 18);  mul_181 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_80: "f32[100]" = torch.ops.aten.add.Tensor(add_79, select_scatter_263);  add_79 = select_scatter_263 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_746: "f32[]" = torch.ops.aten.select.int(select_scatter_262, 0, 17)
        select_scatter_264: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_262, full_default_1, 0, 17);  select_scatter_262 = None
        mul_182: "f32[]" = torch.ops.aten.mul.Tensor(select_746, 2);  select_746 = None
        select_scatter_265: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_182, 0, 17);  mul_182 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_81: "f32[100]" = torch.ops.aten.add.Tensor(add_80, select_scatter_265);  add_80 = select_scatter_265 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_749: "f32[]" = torch.ops.aten.select.int(select_scatter_264, 0, 16)
        select_scatter_266: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_264, full_default_1, 0, 16);  select_scatter_264 = None
        mul_183: "f32[]" = torch.ops.aten.mul.Tensor(select_749, 2);  select_749 = None
        select_scatter_267: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_183, 0, 16);  mul_183 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_82: "f32[100]" = torch.ops.aten.add.Tensor(add_81, select_scatter_267);  add_81 = select_scatter_267 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_752: "f32[]" = torch.ops.aten.select.int(select_scatter_266, 0, 15)
        select_scatter_268: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_266, full_default_1, 0, 15);  select_scatter_266 = None
        mul_184: "f32[]" = torch.ops.aten.mul.Tensor(select_752, 2);  select_752 = None
        select_scatter_269: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_184, 0, 15);  mul_184 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_83: "f32[100]" = torch.ops.aten.add.Tensor(add_82, select_scatter_269);  add_82 = select_scatter_269 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_755: "f32[]" = torch.ops.aten.select.int(select_scatter_268, 0, 14)
        select_scatter_270: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_268, full_default_1, 0, 14);  select_scatter_268 = None
        mul_185: "f32[]" = torch.ops.aten.mul.Tensor(select_755, 2);  select_755 = None
        select_scatter_271: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_185, 0, 14);  mul_185 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_84: "f32[100]" = torch.ops.aten.add.Tensor(add_83, select_scatter_271);  add_83 = select_scatter_271 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_758: "f32[]" = torch.ops.aten.select.int(select_scatter_270, 0, 13)
        select_scatter_272: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_270, full_default_1, 0, 13);  select_scatter_270 = None
        mul_186: "f32[]" = torch.ops.aten.mul.Tensor(select_758, 2);  select_758 = None
        select_scatter_273: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_186, 0, 13);  mul_186 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_85: "f32[100]" = torch.ops.aten.add.Tensor(add_84, select_scatter_273);  add_84 = select_scatter_273 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_761: "f32[]" = torch.ops.aten.select.int(select_scatter_272, 0, 12)
        select_scatter_274: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_272, full_default_1, 0, 12);  select_scatter_272 = None
        mul_187: "f32[]" = torch.ops.aten.mul.Tensor(select_761, 2);  select_761 = None
        select_scatter_275: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_187, 0, 12);  mul_187 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_86: "f32[100]" = torch.ops.aten.add.Tensor(add_85, select_scatter_275);  add_85 = select_scatter_275 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_764: "f32[]" = torch.ops.aten.select.int(select_scatter_274, 0, 11)
        select_scatter_276: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_274, full_default_1, 0, 11);  select_scatter_274 = None
        mul_188: "f32[]" = torch.ops.aten.mul.Tensor(select_764, 2);  select_764 = None
        select_scatter_277: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_188, 0, 11);  mul_188 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_87: "f32[100]" = torch.ops.aten.add.Tensor(add_86, select_scatter_277);  add_86 = select_scatter_277 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_767: "f32[]" = torch.ops.aten.select.int(select_scatter_276, 0, 10)
        select_scatter_278: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_276, full_default_1, 0, 10);  select_scatter_276 = None
        mul_189: "f32[]" = torch.ops.aten.mul.Tensor(select_767, 2);  select_767 = None
        select_scatter_279: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_189, 0, 10);  mul_189 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_88: "f32[100]" = torch.ops.aten.add.Tensor(add_87, select_scatter_279);  add_87 = select_scatter_279 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_770: "f32[]" = torch.ops.aten.select.int(select_scatter_278, 0, 9)
        select_scatter_280: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_278, full_default_1, 0, 9);  select_scatter_278 = None
        mul_190: "f32[]" = torch.ops.aten.mul.Tensor(select_770, 2);  select_770 = None
        select_scatter_281: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_190, 0, 9);  mul_190 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_89: "f32[100]" = torch.ops.aten.add.Tensor(add_88, select_scatter_281);  add_88 = select_scatter_281 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_773: "f32[]" = torch.ops.aten.select.int(select_scatter_280, 0, 8)
        select_scatter_282: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_280, full_default_1, 0, 8);  select_scatter_280 = None
        mul_191: "f32[]" = torch.ops.aten.mul.Tensor(select_773, 2);  select_773 = None
        select_scatter_283: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_191, 0, 8);  mul_191 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_90: "f32[100]" = torch.ops.aten.add.Tensor(add_89, select_scatter_283);  add_89 = select_scatter_283 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_776: "f32[]" = torch.ops.aten.select.int(select_scatter_282, 0, 7)
        select_scatter_284: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_282, full_default_1, 0, 7);  select_scatter_282 = None
        mul_192: "f32[]" = torch.ops.aten.mul.Tensor(select_776, 2);  select_776 = None
        select_scatter_285: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_192, 0, 7);  mul_192 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_91: "f32[100]" = torch.ops.aten.add.Tensor(add_90, select_scatter_285);  add_90 = select_scatter_285 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_779: "f32[]" = torch.ops.aten.select.int(select_scatter_284, 0, 6)
        select_scatter_286: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_284, full_default_1, 0, 6);  select_scatter_284 = None
        mul_193: "f32[]" = torch.ops.aten.mul.Tensor(select_779, 2);  select_779 = None
        select_scatter_287: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_193, 0, 6);  mul_193 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_92: "f32[100]" = torch.ops.aten.add.Tensor(add_91, select_scatter_287);  add_91 = select_scatter_287 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_782: "f32[]" = torch.ops.aten.select.int(select_scatter_286, 0, 5)
        select_scatter_288: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_286, full_default_1, 0, 5);  select_scatter_286 = None
        mul_194: "f32[]" = torch.ops.aten.mul.Tensor(select_782, 2);  select_782 = None
        select_scatter_289: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_194, 0, 5);  mul_194 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_93: "f32[100]" = torch.ops.aten.add.Tensor(add_92, select_scatter_289);  add_92 = select_scatter_289 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_785: "f32[]" = torch.ops.aten.select.int(select_scatter_288, 0, 4)
        select_scatter_290: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_288, full_default_1, 0, 4);  select_scatter_288 = None
        mul_195: "f32[]" = torch.ops.aten.mul.Tensor(select_785, 2);  select_785 = None
        select_scatter_291: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_195, 0, 4);  mul_195 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_94: "f32[100]" = torch.ops.aten.add.Tensor(add_93, select_scatter_291);  add_93 = select_scatter_291 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_788: "f32[]" = torch.ops.aten.select.int(select_scatter_290, 0, 3)
        select_scatter_292: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_290, full_default_1, 0, 3);  select_scatter_290 = None
        mul_196: "f32[]" = torch.ops.aten.mul.Tensor(select_788, 2);  select_788 = None
        select_scatter_293: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_196, 0, 3);  mul_196 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_95: "f32[100]" = torch.ops.aten.add.Tensor(add_94, select_scatter_293);  add_94 = select_scatter_293 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_791: "f32[]" = torch.ops.aten.select.int(select_scatter_292, 0, 2)
        select_scatter_294: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_292, full_default_1, 0, 2);  select_scatter_292 = None
        mul_197: "f32[]" = torch.ops.aten.mul.Tensor(select_791, 2);  select_791 = None
        select_scatter_295: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_197, 0, 2);  mul_197 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_96: "f32[100]" = torch.ops.aten.add.Tensor(add_95, select_scatter_295);  add_95 = select_scatter_295 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_794: "f32[]" = torch.ops.aten.select.int(select_scatter_294, 0, 1)
        select_scatter_296: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_294, full_default_1, 0, 1);  select_scatter_294 = full_default_1 = None
        mul_198: "f32[]" = torch.ops.aten.mul.Tensor(select_794, 2);  select_794 = None
        select_scatter_297: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_198, 0, 1);  mul_198 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_97: "f32[100]" = torch.ops.aten.add.Tensor(add_96, select_scatter_297);  add_96 = select_scatter_297 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_797: "f32[]" = torch.ops.aten.select.int(select_scatter_296, 0, 0);  select_scatter_296 = None
        mul_199: "f32[]" = torch.ops.aten.mul.Tensor(select_797, 2);  select_797 = None
        select_scatter_298: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_199, 0, 0);  full_default = mul_199 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_98: "f32[100]" = torch.ops.aten.add.Tensor(add_97, select_scatter_298);  add_97 = select_scatter_298 = None
        return (add_98,)
        