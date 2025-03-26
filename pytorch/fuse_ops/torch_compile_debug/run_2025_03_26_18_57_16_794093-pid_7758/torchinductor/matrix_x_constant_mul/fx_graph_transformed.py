class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:4 in explicit_loop, code: out = torch.zeros_like(x)
        full_default: "f32[100]" = torch.ops.aten.full.default([100], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 0)
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select, 2);  select = None
        
        # No stacktrace found for following nodes
        select_scatter_default: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul, 0, 0);  mul = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_4: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 1)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(select_4, 2);  select_4 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_1: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default, mul_1, 0, 1);  select_scatter_default = mul_1 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_9: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 2)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(select_9, 2);  select_9 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_2: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_1, mul_2, 0, 2);  select_scatter_default_1 = mul_2 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_14: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 3)
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(select_14, 2);  select_14 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_3: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_2, mul_3, 0, 3);  select_scatter_default_2 = mul_3 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_19: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 4)
        mul_4: "f32[]" = torch.ops.aten.mul.Tensor(select_19, 2);  select_19 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_4: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_3, mul_4, 0, 4);  select_scatter_default_3 = mul_4 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_24: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 5)
        mul_5: "f32[]" = torch.ops.aten.mul.Tensor(select_24, 2);  select_24 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_5: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_4, mul_5, 0, 5);  select_scatter_default_4 = mul_5 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_29: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 6)
        mul_6: "f32[]" = torch.ops.aten.mul.Tensor(select_29, 2);  select_29 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_6: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_5, mul_6, 0, 6);  select_scatter_default_5 = mul_6 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_34: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 7)
        mul_7: "f32[]" = torch.ops.aten.mul.Tensor(select_34, 2);  select_34 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_7: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_6, mul_7, 0, 7);  select_scatter_default_6 = mul_7 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_39: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 8)
        mul_8: "f32[]" = torch.ops.aten.mul.Tensor(select_39, 2);  select_39 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_8: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_7, mul_8, 0, 8);  select_scatter_default_7 = mul_8 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_44: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 9)
        mul_9: "f32[]" = torch.ops.aten.mul.Tensor(select_44, 2);  select_44 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_9: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_8, mul_9, 0, 9);  select_scatter_default_8 = mul_9 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_49: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 10)
        mul_10: "f32[]" = torch.ops.aten.mul.Tensor(select_49, 2);  select_49 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_10: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_9, mul_10, 0, 10);  select_scatter_default_9 = mul_10 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_54: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 11)
        mul_11: "f32[]" = torch.ops.aten.mul.Tensor(select_54, 2);  select_54 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_11: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_10, mul_11, 0, 11);  select_scatter_default_10 = mul_11 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_59: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 12)
        mul_12: "f32[]" = torch.ops.aten.mul.Tensor(select_59, 2);  select_59 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_12: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_11, mul_12, 0, 12);  select_scatter_default_11 = mul_12 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_64: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 13)
        mul_13: "f32[]" = torch.ops.aten.mul.Tensor(select_64, 2);  select_64 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_13: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_12, mul_13, 0, 13);  select_scatter_default_12 = mul_13 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_69: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 14)
        mul_14: "f32[]" = torch.ops.aten.mul.Tensor(select_69, 2);  select_69 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_14: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_13, mul_14, 0, 14);  select_scatter_default_13 = mul_14 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_74: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 15)
        mul_15: "f32[]" = torch.ops.aten.mul.Tensor(select_74, 2);  select_74 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_15: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_14, mul_15, 0, 15);  select_scatter_default_14 = mul_15 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_79: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 16)
        mul_16: "f32[]" = torch.ops.aten.mul.Tensor(select_79, 2);  select_79 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_16: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_15, mul_16, 0, 16);  select_scatter_default_15 = mul_16 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_84: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 17)
        mul_17: "f32[]" = torch.ops.aten.mul.Tensor(select_84, 2);  select_84 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_17: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_16, mul_17, 0, 17);  select_scatter_default_16 = mul_17 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_89: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 18)
        mul_18: "f32[]" = torch.ops.aten.mul.Tensor(select_89, 2);  select_89 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_18: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_17, mul_18, 0, 18);  select_scatter_default_17 = mul_18 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_94: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 19)
        mul_19: "f32[]" = torch.ops.aten.mul.Tensor(select_94, 2);  select_94 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_19: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_18, mul_19, 0, 19);  select_scatter_default_18 = mul_19 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_99: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 20)
        mul_20: "f32[]" = torch.ops.aten.mul.Tensor(select_99, 2);  select_99 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_20: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_19, mul_20, 0, 20);  select_scatter_default_19 = mul_20 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_104: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 21)
        mul_21: "f32[]" = torch.ops.aten.mul.Tensor(select_104, 2);  select_104 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_21: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_20, mul_21, 0, 21);  select_scatter_default_20 = mul_21 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_109: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 22)
        mul_22: "f32[]" = torch.ops.aten.mul.Tensor(select_109, 2);  select_109 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_22: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_21, mul_22, 0, 22);  select_scatter_default_21 = mul_22 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_114: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 23)
        mul_23: "f32[]" = torch.ops.aten.mul.Tensor(select_114, 2);  select_114 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_23: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_22, mul_23, 0, 23);  select_scatter_default_22 = mul_23 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_119: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 24)
        mul_24: "f32[]" = torch.ops.aten.mul.Tensor(select_119, 2);  select_119 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_24: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_23, mul_24, 0, 24);  select_scatter_default_23 = mul_24 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_124: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 25)
        mul_25: "f32[]" = torch.ops.aten.mul.Tensor(select_124, 2);  select_124 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_25: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_24, mul_25, 0, 25);  select_scatter_default_24 = mul_25 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_129: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 26)
        mul_26: "f32[]" = torch.ops.aten.mul.Tensor(select_129, 2);  select_129 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_26: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_25, mul_26, 0, 26);  select_scatter_default_25 = mul_26 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_134: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 27)
        mul_27: "f32[]" = torch.ops.aten.mul.Tensor(select_134, 2);  select_134 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_27: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_26, mul_27, 0, 27);  select_scatter_default_26 = mul_27 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_139: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 28)
        mul_28: "f32[]" = torch.ops.aten.mul.Tensor(select_139, 2);  select_139 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_28: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_27, mul_28, 0, 28);  select_scatter_default_27 = mul_28 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_144: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 29)
        mul_29: "f32[]" = torch.ops.aten.mul.Tensor(select_144, 2);  select_144 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_29: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_28, mul_29, 0, 29);  select_scatter_default_28 = mul_29 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_149: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 30)
        mul_30: "f32[]" = torch.ops.aten.mul.Tensor(select_149, 2);  select_149 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_30: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_29, mul_30, 0, 30);  select_scatter_default_29 = mul_30 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_154: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 31)
        mul_31: "f32[]" = torch.ops.aten.mul.Tensor(select_154, 2);  select_154 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_31: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_30, mul_31, 0, 31);  select_scatter_default_30 = mul_31 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_159: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 32)
        mul_32: "f32[]" = torch.ops.aten.mul.Tensor(select_159, 2);  select_159 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_32: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_31, mul_32, 0, 32);  select_scatter_default_31 = mul_32 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_164: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 33)
        mul_33: "f32[]" = torch.ops.aten.mul.Tensor(select_164, 2);  select_164 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_33: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_32, mul_33, 0, 33);  select_scatter_default_32 = mul_33 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_169: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 34)
        mul_34: "f32[]" = torch.ops.aten.mul.Tensor(select_169, 2);  select_169 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_34: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_33, mul_34, 0, 34);  select_scatter_default_33 = mul_34 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_174: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 35)
        mul_35: "f32[]" = torch.ops.aten.mul.Tensor(select_174, 2);  select_174 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_35: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_34, mul_35, 0, 35);  select_scatter_default_34 = mul_35 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_179: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 36)
        mul_36: "f32[]" = torch.ops.aten.mul.Tensor(select_179, 2);  select_179 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_36: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_35, mul_36, 0, 36);  select_scatter_default_35 = mul_36 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_184: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 37)
        mul_37: "f32[]" = torch.ops.aten.mul.Tensor(select_184, 2);  select_184 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_37: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_36, mul_37, 0, 37);  select_scatter_default_36 = mul_37 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_189: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 38)
        mul_38: "f32[]" = torch.ops.aten.mul.Tensor(select_189, 2);  select_189 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_38: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_37, mul_38, 0, 38);  select_scatter_default_37 = mul_38 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_194: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 39)
        mul_39: "f32[]" = torch.ops.aten.mul.Tensor(select_194, 2);  select_194 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_39: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_38, mul_39, 0, 39);  select_scatter_default_38 = mul_39 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_199: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 40)
        mul_40: "f32[]" = torch.ops.aten.mul.Tensor(select_199, 2);  select_199 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_40: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_39, mul_40, 0, 40);  select_scatter_default_39 = mul_40 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_204: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 41)
        mul_41: "f32[]" = torch.ops.aten.mul.Tensor(select_204, 2);  select_204 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_41: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_40, mul_41, 0, 41);  select_scatter_default_40 = mul_41 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_209: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 42)
        mul_42: "f32[]" = torch.ops.aten.mul.Tensor(select_209, 2);  select_209 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_42: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_41, mul_42, 0, 42);  select_scatter_default_41 = mul_42 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_214: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 43)
        mul_43: "f32[]" = torch.ops.aten.mul.Tensor(select_214, 2);  select_214 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_43: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_42, mul_43, 0, 43);  select_scatter_default_42 = mul_43 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_219: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 44)
        mul_44: "f32[]" = torch.ops.aten.mul.Tensor(select_219, 2);  select_219 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_44: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_43, mul_44, 0, 44);  select_scatter_default_43 = mul_44 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_224: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 45)
        mul_45: "f32[]" = torch.ops.aten.mul.Tensor(select_224, 2);  select_224 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_45: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_44, mul_45, 0, 45);  select_scatter_default_44 = mul_45 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_229: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 46)
        mul_46: "f32[]" = torch.ops.aten.mul.Tensor(select_229, 2);  select_229 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_46: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_45, mul_46, 0, 46);  select_scatter_default_45 = mul_46 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_234: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 47)
        mul_47: "f32[]" = torch.ops.aten.mul.Tensor(select_234, 2);  select_234 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_47: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_46, mul_47, 0, 47);  select_scatter_default_46 = mul_47 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_239: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 48)
        mul_48: "f32[]" = torch.ops.aten.mul.Tensor(select_239, 2);  select_239 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_48: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_47, mul_48, 0, 48);  select_scatter_default_47 = mul_48 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_244: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 49)
        mul_49: "f32[]" = torch.ops.aten.mul.Tensor(select_244, 2);  select_244 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_49: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_48, mul_49, 0, 49);  select_scatter_default_48 = mul_49 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_249: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 50)
        mul_50: "f32[]" = torch.ops.aten.mul.Tensor(select_249, 2);  select_249 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_50: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_49, mul_50, 0, 50);  select_scatter_default_49 = mul_50 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_254: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 51)
        mul_51: "f32[]" = torch.ops.aten.mul.Tensor(select_254, 2);  select_254 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_51: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_50, mul_51, 0, 51);  select_scatter_default_50 = mul_51 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_259: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 52)
        mul_52: "f32[]" = torch.ops.aten.mul.Tensor(select_259, 2);  select_259 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_52: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_51, mul_52, 0, 52);  select_scatter_default_51 = mul_52 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_264: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 53)
        mul_53: "f32[]" = torch.ops.aten.mul.Tensor(select_264, 2);  select_264 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_53: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_52, mul_53, 0, 53);  select_scatter_default_52 = mul_53 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_269: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 54)
        mul_54: "f32[]" = torch.ops.aten.mul.Tensor(select_269, 2);  select_269 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_54: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_53, mul_54, 0, 54);  select_scatter_default_53 = mul_54 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_274: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 55)
        mul_55: "f32[]" = torch.ops.aten.mul.Tensor(select_274, 2);  select_274 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_55: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_54, mul_55, 0, 55);  select_scatter_default_54 = mul_55 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_279: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 56)
        mul_56: "f32[]" = torch.ops.aten.mul.Tensor(select_279, 2);  select_279 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_56: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_55, mul_56, 0, 56);  select_scatter_default_55 = mul_56 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_284: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 57)
        mul_57: "f32[]" = torch.ops.aten.mul.Tensor(select_284, 2);  select_284 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_57: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_56, mul_57, 0, 57);  select_scatter_default_56 = mul_57 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_289: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 58)
        mul_58: "f32[]" = torch.ops.aten.mul.Tensor(select_289, 2);  select_289 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_58: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_57, mul_58, 0, 58);  select_scatter_default_57 = mul_58 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_294: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 59)
        mul_59: "f32[]" = torch.ops.aten.mul.Tensor(select_294, 2);  select_294 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_59: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_58, mul_59, 0, 59);  select_scatter_default_58 = mul_59 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_299: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 60)
        mul_60: "f32[]" = torch.ops.aten.mul.Tensor(select_299, 2);  select_299 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_60: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_59, mul_60, 0, 60);  select_scatter_default_59 = mul_60 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_304: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 61)
        mul_61: "f32[]" = torch.ops.aten.mul.Tensor(select_304, 2);  select_304 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_61: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_60, mul_61, 0, 61);  select_scatter_default_60 = mul_61 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_309: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 62)
        mul_62: "f32[]" = torch.ops.aten.mul.Tensor(select_309, 2);  select_309 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_62: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_61, mul_62, 0, 62);  select_scatter_default_61 = mul_62 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_314: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 63)
        mul_63: "f32[]" = torch.ops.aten.mul.Tensor(select_314, 2);  select_314 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_63: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_62, mul_63, 0, 63);  select_scatter_default_62 = mul_63 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_319: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 64)
        mul_64: "f32[]" = torch.ops.aten.mul.Tensor(select_319, 2);  select_319 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_64: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_63, mul_64, 0, 64);  select_scatter_default_63 = mul_64 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_324: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 65)
        mul_65: "f32[]" = torch.ops.aten.mul.Tensor(select_324, 2);  select_324 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_65: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_64, mul_65, 0, 65);  select_scatter_default_64 = mul_65 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_329: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 66)
        mul_66: "f32[]" = torch.ops.aten.mul.Tensor(select_329, 2);  select_329 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_66: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_65, mul_66, 0, 66);  select_scatter_default_65 = mul_66 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_334: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 67)
        mul_67: "f32[]" = torch.ops.aten.mul.Tensor(select_334, 2);  select_334 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_67: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_66, mul_67, 0, 67);  select_scatter_default_66 = mul_67 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_339: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 68)
        mul_68: "f32[]" = torch.ops.aten.mul.Tensor(select_339, 2);  select_339 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_68: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_67, mul_68, 0, 68);  select_scatter_default_67 = mul_68 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_344: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 69)
        mul_69: "f32[]" = torch.ops.aten.mul.Tensor(select_344, 2);  select_344 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_69: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_68, mul_69, 0, 69);  select_scatter_default_68 = mul_69 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_349: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 70)
        mul_70: "f32[]" = torch.ops.aten.mul.Tensor(select_349, 2);  select_349 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_70: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_69, mul_70, 0, 70);  select_scatter_default_69 = mul_70 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_354: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 71)
        mul_71: "f32[]" = torch.ops.aten.mul.Tensor(select_354, 2);  select_354 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_71: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_70, mul_71, 0, 71);  select_scatter_default_70 = mul_71 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_359: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 72)
        mul_72: "f32[]" = torch.ops.aten.mul.Tensor(select_359, 2);  select_359 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_72: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_71, mul_72, 0, 72);  select_scatter_default_71 = mul_72 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_364: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 73)
        mul_73: "f32[]" = torch.ops.aten.mul.Tensor(select_364, 2);  select_364 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_73: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_72, mul_73, 0, 73);  select_scatter_default_72 = mul_73 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_369: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 74)
        mul_74: "f32[]" = torch.ops.aten.mul.Tensor(select_369, 2);  select_369 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_74: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_73, mul_74, 0, 74);  select_scatter_default_73 = mul_74 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_374: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 75)
        mul_75: "f32[]" = torch.ops.aten.mul.Tensor(select_374, 2);  select_374 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_75: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_74, mul_75, 0, 75);  select_scatter_default_74 = mul_75 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_379: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 76)
        mul_76: "f32[]" = torch.ops.aten.mul.Tensor(select_379, 2);  select_379 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_76: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_75, mul_76, 0, 76);  select_scatter_default_75 = mul_76 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_384: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 77)
        mul_77: "f32[]" = torch.ops.aten.mul.Tensor(select_384, 2);  select_384 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_77: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_76, mul_77, 0, 77);  select_scatter_default_76 = mul_77 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_389: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 78)
        mul_78: "f32[]" = torch.ops.aten.mul.Tensor(select_389, 2);  select_389 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_78: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_77, mul_78, 0, 78);  select_scatter_default_77 = mul_78 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_394: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 79)
        mul_79: "f32[]" = torch.ops.aten.mul.Tensor(select_394, 2);  select_394 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_79: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_78, mul_79, 0, 79);  select_scatter_default_78 = mul_79 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_399: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 80)
        mul_80: "f32[]" = torch.ops.aten.mul.Tensor(select_399, 2);  select_399 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_80: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_79, mul_80, 0, 80);  select_scatter_default_79 = mul_80 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_404: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 81)
        mul_81: "f32[]" = torch.ops.aten.mul.Tensor(select_404, 2);  select_404 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_81: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_80, mul_81, 0, 81);  select_scatter_default_80 = mul_81 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_409: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 82)
        mul_82: "f32[]" = torch.ops.aten.mul.Tensor(select_409, 2);  select_409 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_82: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_81, mul_82, 0, 82);  select_scatter_default_81 = mul_82 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_414: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 83)
        mul_83: "f32[]" = torch.ops.aten.mul.Tensor(select_414, 2);  select_414 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_83: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_82, mul_83, 0, 83);  select_scatter_default_82 = mul_83 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_419: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 84)
        mul_84: "f32[]" = torch.ops.aten.mul.Tensor(select_419, 2);  select_419 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_84: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_83, mul_84, 0, 84);  select_scatter_default_83 = mul_84 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_424: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 85)
        mul_85: "f32[]" = torch.ops.aten.mul.Tensor(select_424, 2);  select_424 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_85: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_84, mul_85, 0, 85);  select_scatter_default_84 = mul_85 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_429: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 86)
        mul_86: "f32[]" = torch.ops.aten.mul.Tensor(select_429, 2);  select_429 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_86: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_85, mul_86, 0, 86);  select_scatter_default_85 = mul_86 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_434: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 87)
        mul_87: "f32[]" = torch.ops.aten.mul.Tensor(select_434, 2);  select_434 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_87: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_86, mul_87, 0, 87);  select_scatter_default_86 = mul_87 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_439: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 88)
        mul_88: "f32[]" = torch.ops.aten.mul.Tensor(select_439, 2);  select_439 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_88: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_87, mul_88, 0, 88);  select_scatter_default_87 = mul_88 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_444: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 89)
        mul_89: "f32[]" = torch.ops.aten.mul.Tensor(select_444, 2);  select_444 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_89: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_88, mul_89, 0, 89);  select_scatter_default_88 = mul_89 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_449: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 90)
        mul_90: "f32[]" = torch.ops.aten.mul.Tensor(select_449, 2);  select_449 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_90: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_89, mul_90, 0, 90);  select_scatter_default_89 = mul_90 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_454: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 91)
        mul_91: "f32[]" = torch.ops.aten.mul.Tensor(select_454, 2);  select_454 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_91: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_90, mul_91, 0, 91);  select_scatter_default_90 = mul_91 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_459: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 92)
        mul_92: "f32[]" = torch.ops.aten.mul.Tensor(select_459, 2);  select_459 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_92: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_91, mul_92, 0, 92);  select_scatter_default_91 = mul_92 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_464: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 93)
        mul_93: "f32[]" = torch.ops.aten.mul.Tensor(select_464, 2);  select_464 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_93: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_92, mul_93, 0, 93);  select_scatter_default_92 = mul_93 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_469: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 94)
        mul_94: "f32[]" = torch.ops.aten.mul.Tensor(select_469, 2);  select_469 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_94: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_93, mul_94, 0, 94);  select_scatter_default_93 = mul_94 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_474: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 95)
        mul_95: "f32[]" = torch.ops.aten.mul.Tensor(select_474, 2);  select_474 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_95: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_94, mul_95, 0, 95);  select_scatter_default_94 = mul_95 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_479: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 96)
        mul_96: "f32[]" = torch.ops.aten.mul.Tensor(select_479, 2);  select_479 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_96: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_95, mul_96, 0, 96);  select_scatter_default_95 = mul_96 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_484: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 97)
        mul_97: "f32[]" = torch.ops.aten.mul.Tensor(select_484, 2);  select_484 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_97: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_96, mul_97, 0, 97);  select_scatter_default_96 = mul_97 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_489: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 98)
        mul_98: "f32[]" = torch.ops.aten.mul.Tensor(select_489, 2);  select_489 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_98: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_97, mul_98, 0, 98);  select_scatter_default_97 = mul_98 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_494: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 99);  primals_1 = None
        mul_99: "f32[]" = torch.ops.aten.mul.Tensor(select_494, 2);  select_494 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_99: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_98, mul_99, 0, 99);  select_scatter_default_98 = mul_99 = None
        return (select_scatter_default_99, full_default)
        