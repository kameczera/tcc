class GraphModule(torch.nn.Module):
    def forward(self, full_default: "f32[100]", tangents_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_500: "f32[]" = torch.ops.aten.select.int(tangents_1, 0, 99)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        select_scatter_default: "f32[100]" = torch.ops.aten.select_scatter.default(tangents_1, full_default_1, 0, 99);  tangents_1 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_100: "f32[]" = torch.ops.aten.mul.Tensor(select_500, 2);  select_500 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_1: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_100, 0, 99);  mul_100 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_503: "f32[]" = torch.ops.aten.select.int(select_scatter_default, 0, 98)
        
        # No stacktrace found for following nodes
        select_scatter_default_2: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default, full_default_1, 0, 98);  select_scatter_default = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_101: "f32[]" = torch.ops.aten.mul.Tensor(select_503, 2);  select_503 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_3: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_101, 0, 98);  mul_101 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add: "f32[100]" = torch.ops.aten.add.Tensor(select_scatter_default_1, select_scatter_default_3);  select_scatter_default_1 = select_scatter_default_3 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_506: "f32[]" = torch.ops.aten.select.int(select_scatter_default_2, 0, 97)
        
        # No stacktrace found for following nodes
        select_scatter_default_4: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_2, full_default_1, 0, 97);  select_scatter_default_2 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_102: "f32[]" = torch.ops.aten.mul.Tensor(select_506, 2);  select_506 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_5: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_102, 0, 97);  mul_102 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_1: "f32[100]" = torch.ops.aten.add.Tensor(add, select_scatter_default_5);  add = select_scatter_default_5 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_509: "f32[]" = torch.ops.aten.select.int(select_scatter_default_4, 0, 96)
        
        # No stacktrace found for following nodes
        select_scatter_default_6: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_4, full_default_1, 0, 96);  select_scatter_default_4 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_103: "f32[]" = torch.ops.aten.mul.Tensor(select_509, 2);  select_509 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_7: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_103, 0, 96);  mul_103 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_2: "f32[100]" = torch.ops.aten.add.Tensor(add_1, select_scatter_default_7);  add_1 = select_scatter_default_7 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_512: "f32[]" = torch.ops.aten.select.int(select_scatter_default_6, 0, 95)
        
        # No stacktrace found for following nodes
        select_scatter_default_8: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_6, full_default_1, 0, 95);  select_scatter_default_6 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_104: "f32[]" = torch.ops.aten.mul.Tensor(select_512, 2);  select_512 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_9: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_104, 0, 95);  mul_104 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_3: "f32[100]" = torch.ops.aten.add.Tensor(add_2, select_scatter_default_9);  add_2 = select_scatter_default_9 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_515: "f32[]" = torch.ops.aten.select.int(select_scatter_default_8, 0, 94)
        
        # No stacktrace found for following nodes
        select_scatter_default_10: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_8, full_default_1, 0, 94);  select_scatter_default_8 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_105: "f32[]" = torch.ops.aten.mul.Tensor(select_515, 2);  select_515 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_11: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_105, 0, 94);  mul_105 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_4: "f32[100]" = torch.ops.aten.add.Tensor(add_3, select_scatter_default_11);  add_3 = select_scatter_default_11 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_518: "f32[]" = torch.ops.aten.select.int(select_scatter_default_10, 0, 93)
        
        # No stacktrace found for following nodes
        select_scatter_default_12: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_10, full_default_1, 0, 93);  select_scatter_default_10 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_106: "f32[]" = torch.ops.aten.mul.Tensor(select_518, 2);  select_518 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_13: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_106, 0, 93);  mul_106 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_5: "f32[100]" = torch.ops.aten.add.Tensor(add_4, select_scatter_default_13);  add_4 = select_scatter_default_13 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_521: "f32[]" = torch.ops.aten.select.int(select_scatter_default_12, 0, 92)
        
        # No stacktrace found for following nodes
        select_scatter_default_14: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_12, full_default_1, 0, 92);  select_scatter_default_12 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_107: "f32[]" = torch.ops.aten.mul.Tensor(select_521, 2);  select_521 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_15: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_107, 0, 92);  mul_107 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_6: "f32[100]" = torch.ops.aten.add.Tensor(add_5, select_scatter_default_15);  add_5 = select_scatter_default_15 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_524: "f32[]" = torch.ops.aten.select.int(select_scatter_default_14, 0, 91)
        
        # No stacktrace found for following nodes
        select_scatter_default_16: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_14, full_default_1, 0, 91);  select_scatter_default_14 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_108: "f32[]" = torch.ops.aten.mul.Tensor(select_524, 2);  select_524 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_17: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_108, 0, 91);  mul_108 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_7: "f32[100]" = torch.ops.aten.add.Tensor(add_6, select_scatter_default_17);  add_6 = select_scatter_default_17 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_527: "f32[]" = torch.ops.aten.select.int(select_scatter_default_16, 0, 90)
        
        # No stacktrace found for following nodes
        select_scatter_default_18: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_16, full_default_1, 0, 90);  select_scatter_default_16 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_109: "f32[]" = torch.ops.aten.mul.Tensor(select_527, 2);  select_527 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_19: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_109, 0, 90);  mul_109 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_8: "f32[100]" = torch.ops.aten.add.Tensor(add_7, select_scatter_default_19);  add_7 = select_scatter_default_19 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_530: "f32[]" = torch.ops.aten.select.int(select_scatter_default_18, 0, 89)
        
        # No stacktrace found for following nodes
        select_scatter_default_20: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_18, full_default_1, 0, 89);  select_scatter_default_18 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_110: "f32[]" = torch.ops.aten.mul.Tensor(select_530, 2);  select_530 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_21: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_110, 0, 89);  mul_110 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_9: "f32[100]" = torch.ops.aten.add.Tensor(add_8, select_scatter_default_21);  add_8 = select_scatter_default_21 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_533: "f32[]" = torch.ops.aten.select.int(select_scatter_default_20, 0, 88)
        
        # No stacktrace found for following nodes
        select_scatter_default_22: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_20, full_default_1, 0, 88);  select_scatter_default_20 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_111: "f32[]" = torch.ops.aten.mul.Tensor(select_533, 2);  select_533 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_23: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_111, 0, 88);  mul_111 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_10: "f32[100]" = torch.ops.aten.add.Tensor(add_9, select_scatter_default_23);  add_9 = select_scatter_default_23 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_536: "f32[]" = torch.ops.aten.select.int(select_scatter_default_22, 0, 87)
        
        # No stacktrace found for following nodes
        select_scatter_default_24: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_22, full_default_1, 0, 87);  select_scatter_default_22 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_112: "f32[]" = torch.ops.aten.mul.Tensor(select_536, 2);  select_536 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_25: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_112, 0, 87);  mul_112 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_11: "f32[100]" = torch.ops.aten.add.Tensor(add_10, select_scatter_default_25);  add_10 = select_scatter_default_25 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_539: "f32[]" = torch.ops.aten.select.int(select_scatter_default_24, 0, 86)
        
        # No stacktrace found for following nodes
        select_scatter_default_26: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_24, full_default_1, 0, 86);  select_scatter_default_24 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_113: "f32[]" = torch.ops.aten.mul.Tensor(select_539, 2);  select_539 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_27: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_113, 0, 86);  mul_113 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_12: "f32[100]" = torch.ops.aten.add.Tensor(add_11, select_scatter_default_27);  add_11 = select_scatter_default_27 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_542: "f32[]" = torch.ops.aten.select.int(select_scatter_default_26, 0, 85)
        
        # No stacktrace found for following nodes
        select_scatter_default_28: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_26, full_default_1, 0, 85);  select_scatter_default_26 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_114: "f32[]" = torch.ops.aten.mul.Tensor(select_542, 2);  select_542 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_29: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_114, 0, 85);  mul_114 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_13: "f32[100]" = torch.ops.aten.add.Tensor(add_12, select_scatter_default_29);  add_12 = select_scatter_default_29 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_545: "f32[]" = torch.ops.aten.select.int(select_scatter_default_28, 0, 84)
        
        # No stacktrace found for following nodes
        select_scatter_default_30: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_28, full_default_1, 0, 84);  select_scatter_default_28 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_115: "f32[]" = torch.ops.aten.mul.Tensor(select_545, 2);  select_545 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_31: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_115, 0, 84);  mul_115 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_14: "f32[100]" = torch.ops.aten.add.Tensor(add_13, select_scatter_default_31);  add_13 = select_scatter_default_31 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_548: "f32[]" = torch.ops.aten.select.int(select_scatter_default_30, 0, 83)
        
        # No stacktrace found for following nodes
        select_scatter_default_32: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_30, full_default_1, 0, 83);  select_scatter_default_30 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_116: "f32[]" = torch.ops.aten.mul.Tensor(select_548, 2);  select_548 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_33: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_116, 0, 83);  mul_116 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_15: "f32[100]" = torch.ops.aten.add.Tensor(add_14, select_scatter_default_33);  add_14 = select_scatter_default_33 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_551: "f32[]" = torch.ops.aten.select.int(select_scatter_default_32, 0, 82)
        
        # No stacktrace found for following nodes
        select_scatter_default_34: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_32, full_default_1, 0, 82);  select_scatter_default_32 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_117: "f32[]" = torch.ops.aten.mul.Tensor(select_551, 2);  select_551 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_35: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_117, 0, 82);  mul_117 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_16: "f32[100]" = torch.ops.aten.add.Tensor(add_15, select_scatter_default_35);  add_15 = select_scatter_default_35 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_554: "f32[]" = torch.ops.aten.select.int(select_scatter_default_34, 0, 81)
        
        # No stacktrace found for following nodes
        select_scatter_default_36: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_34, full_default_1, 0, 81);  select_scatter_default_34 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_118: "f32[]" = torch.ops.aten.mul.Tensor(select_554, 2);  select_554 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_37: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_118, 0, 81);  mul_118 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_17: "f32[100]" = torch.ops.aten.add.Tensor(add_16, select_scatter_default_37);  add_16 = select_scatter_default_37 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_557: "f32[]" = torch.ops.aten.select.int(select_scatter_default_36, 0, 80)
        
        # No stacktrace found for following nodes
        select_scatter_default_38: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_36, full_default_1, 0, 80);  select_scatter_default_36 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_119: "f32[]" = torch.ops.aten.mul.Tensor(select_557, 2);  select_557 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_39: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_119, 0, 80);  mul_119 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_18: "f32[100]" = torch.ops.aten.add.Tensor(add_17, select_scatter_default_39);  add_17 = select_scatter_default_39 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_560: "f32[]" = torch.ops.aten.select.int(select_scatter_default_38, 0, 79)
        
        # No stacktrace found for following nodes
        select_scatter_default_40: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_38, full_default_1, 0, 79);  select_scatter_default_38 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_120: "f32[]" = torch.ops.aten.mul.Tensor(select_560, 2);  select_560 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_41: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_120, 0, 79);  mul_120 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_19: "f32[100]" = torch.ops.aten.add.Tensor(add_18, select_scatter_default_41);  add_18 = select_scatter_default_41 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_563: "f32[]" = torch.ops.aten.select.int(select_scatter_default_40, 0, 78)
        
        # No stacktrace found for following nodes
        select_scatter_default_42: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_40, full_default_1, 0, 78);  select_scatter_default_40 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_121: "f32[]" = torch.ops.aten.mul.Tensor(select_563, 2);  select_563 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_43: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_121, 0, 78);  mul_121 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_20: "f32[100]" = torch.ops.aten.add.Tensor(add_19, select_scatter_default_43);  add_19 = select_scatter_default_43 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_566: "f32[]" = torch.ops.aten.select.int(select_scatter_default_42, 0, 77)
        
        # No stacktrace found for following nodes
        select_scatter_default_44: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_42, full_default_1, 0, 77);  select_scatter_default_42 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_122: "f32[]" = torch.ops.aten.mul.Tensor(select_566, 2);  select_566 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_45: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_122, 0, 77);  mul_122 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_21: "f32[100]" = torch.ops.aten.add.Tensor(add_20, select_scatter_default_45);  add_20 = select_scatter_default_45 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_569: "f32[]" = torch.ops.aten.select.int(select_scatter_default_44, 0, 76)
        
        # No stacktrace found for following nodes
        select_scatter_default_46: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_44, full_default_1, 0, 76);  select_scatter_default_44 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_123: "f32[]" = torch.ops.aten.mul.Tensor(select_569, 2);  select_569 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_47: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_123, 0, 76);  mul_123 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_22: "f32[100]" = torch.ops.aten.add.Tensor(add_21, select_scatter_default_47);  add_21 = select_scatter_default_47 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_572: "f32[]" = torch.ops.aten.select.int(select_scatter_default_46, 0, 75)
        
        # No stacktrace found for following nodes
        select_scatter_default_48: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_46, full_default_1, 0, 75);  select_scatter_default_46 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_124: "f32[]" = torch.ops.aten.mul.Tensor(select_572, 2);  select_572 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_49: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_124, 0, 75);  mul_124 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_23: "f32[100]" = torch.ops.aten.add.Tensor(add_22, select_scatter_default_49);  add_22 = select_scatter_default_49 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_575: "f32[]" = torch.ops.aten.select.int(select_scatter_default_48, 0, 74)
        
        # No stacktrace found for following nodes
        select_scatter_default_50: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_48, full_default_1, 0, 74);  select_scatter_default_48 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_125: "f32[]" = torch.ops.aten.mul.Tensor(select_575, 2);  select_575 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_51: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_125, 0, 74);  mul_125 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_24: "f32[100]" = torch.ops.aten.add.Tensor(add_23, select_scatter_default_51);  add_23 = select_scatter_default_51 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_578: "f32[]" = torch.ops.aten.select.int(select_scatter_default_50, 0, 73)
        
        # No stacktrace found for following nodes
        select_scatter_default_52: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_50, full_default_1, 0, 73);  select_scatter_default_50 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_126: "f32[]" = torch.ops.aten.mul.Tensor(select_578, 2);  select_578 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_53: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_126, 0, 73);  mul_126 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_25: "f32[100]" = torch.ops.aten.add.Tensor(add_24, select_scatter_default_53);  add_24 = select_scatter_default_53 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_581: "f32[]" = torch.ops.aten.select.int(select_scatter_default_52, 0, 72)
        
        # No stacktrace found for following nodes
        select_scatter_default_54: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_52, full_default_1, 0, 72);  select_scatter_default_52 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_127: "f32[]" = torch.ops.aten.mul.Tensor(select_581, 2);  select_581 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_55: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_127, 0, 72);  mul_127 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_26: "f32[100]" = torch.ops.aten.add.Tensor(add_25, select_scatter_default_55);  add_25 = select_scatter_default_55 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_584: "f32[]" = torch.ops.aten.select.int(select_scatter_default_54, 0, 71)
        
        # No stacktrace found for following nodes
        select_scatter_default_56: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_54, full_default_1, 0, 71);  select_scatter_default_54 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_128: "f32[]" = torch.ops.aten.mul.Tensor(select_584, 2);  select_584 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_57: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_128, 0, 71);  mul_128 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_27: "f32[100]" = torch.ops.aten.add.Tensor(add_26, select_scatter_default_57);  add_26 = select_scatter_default_57 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_587: "f32[]" = torch.ops.aten.select.int(select_scatter_default_56, 0, 70)
        
        # No stacktrace found for following nodes
        select_scatter_default_58: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_56, full_default_1, 0, 70);  select_scatter_default_56 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_129: "f32[]" = torch.ops.aten.mul.Tensor(select_587, 2);  select_587 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_59: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_129, 0, 70);  mul_129 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_28: "f32[100]" = torch.ops.aten.add.Tensor(add_27, select_scatter_default_59);  add_27 = select_scatter_default_59 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_590: "f32[]" = torch.ops.aten.select.int(select_scatter_default_58, 0, 69)
        
        # No stacktrace found for following nodes
        select_scatter_default_60: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_58, full_default_1, 0, 69);  select_scatter_default_58 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_130: "f32[]" = torch.ops.aten.mul.Tensor(select_590, 2);  select_590 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_61: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_130, 0, 69);  mul_130 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_29: "f32[100]" = torch.ops.aten.add.Tensor(add_28, select_scatter_default_61);  add_28 = select_scatter_default_61 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_593: "f32[]" = torch.ops.aten.select.int(select_scatter_default_60, 0, 68)
        
        # No stacktrace found for following nodes
        select_scatter_default_62: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_60, full_default_1, 0, 68);  select_scatter_default_60 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_131: "f32[]" = torch.ops.aten.mul.Tensor(select_593, 2);  select_593 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_63: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_131, 0, 68);  mul_131 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_30: "f32[100]" = torch.ops.aten.add.Tensor(add_29, select_scatter_default_63);  add_29 = select_scatter_default_63 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_596: "f32[]" = torch.ops.aten.select.int(select_scatter_default_62, 0, 67)
        
        # No stacktrace found for following nodes
        select_scatter_default_64: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_62, full_default_1, 0, 67);  select_scatter_default_62 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_132: "f32[]" = torch.ops.aten.mul.Tensor(select_596, 2);  select_596 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_65: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_132, 0, 67);  mul_132 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_31: "f32[100]" = torch.ops.aten.add.Tensor(add_30, select_scatter_default_65);  add_30 = select_scatter_default_65 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_599: "f32[]" = torch.ops.aten.select.int(select_scatter_default_64, 0, 66)
        
        # No stacktrace found for following nodes
        select_scatter_default_66: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_64, full_default_1, 0, 66);  select_scatter_default_64 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_133: "f32[]" = torch.ops.aten.mul.Tensor(select_599, 2);  select_599 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_67: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_133, 0, 66);  mul_133 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_32: "f32[100]" = torch.ops.aten.add.Tensor(add_31, select_scatter_default_67);  add_31 = select_scatter_default_67 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_602: "f32[]" = torch.ops.aten.select.int(select_scatter_default_66, 0, 65)
        
        # No stacktrace found for following nodes
        select_scatter_default_68: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_66, full_default_1, 0, 65);  select_scatter_default_66 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_134: "f32[]" = torch.ops.aten.mul.Tensor(select_602, 2);  select_602 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_69: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_134, 0, 65);  mul_134 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_33: "f32[100]" = torch.ops.aten.add.Tensor(add_32, select_scatter_default_69);  add_32 = select_scatter_default_69 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_605: "f32[]" = torch.ops.aten.select.int(select_scatter_default_68, 0, 64)
        
        # No stacktrace found for following nodes
        select_scatter_default_70: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_68, full_default_1, 0, 64);  select_scatter_default_68 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_135: "f32[]" = torch.ops.aten.mul.Tensor(select_605, 2);  select_605 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_71: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_135, 0, 64);  mul_135 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_34: "f32[100]" = torch.ops.aten.add.Tensor(add_33, select_scatter_default_71);  add_33 = select_scatter_default_71 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_608: "f32[]" = torch.ops.aten.select.int(select_scatter_default_70, 0, 63)
        
        # No stacktrace found for following nodes
        select_scatter_default_72: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_70, full_default_1, 0, 63);  select_scatter_default_70 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_136: "f32[]" = torch.ops.aten.mul.Tensor(select_608, 2);  select_608 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_73: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_136, 0, 63);  mul_136 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_35: "f32[100]" = torch.ops.aten.add.Tensor(add_34, select_scatter_default_73);  add_34 = select_scatter_default_73 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_611: "f32[]" = torch.ops.aten.select.int(select_scatter_default_72, 0, 62)
        
        # No stacktrace found for following nodes
        select_scatter_default_74: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_72, full_default_1, 0, 62);  select_scatter_default_72 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_137: "f32[]" = torch.ops.aten.mul.Tensor(select_611, 2);  select_611 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_75: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_137, 0, 62);  mul_137 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_36: "f32[100]" = torch.ops.aten.add.Tensor(add_35, select_scatter_default_75);  add_35 = select_scatter_default_75 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_614: "f32[]" = torch.ops.aten.select.int(select_scatter_default_74, 0, 61)
        
        # No stacktrace found for following nodes
        select_scatter_default_76: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_74, full_default_1, 0, 61);  select_scatter_default_74 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_138: "f32[]" = torch.ops.aten.mul.Tensor(select_614, 2);  select_614 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_77: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_138, 0, 61);  mul_138 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_37: "f32[100]" = torch.ops.aten.add.Tensor(add_36, select_scatter_default_77);  add_36 = select_scatter_default_77 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_617: "f32[]" = torch.ops.aten.select.int(select_scatter_default_76, 0, 60)
        
        # No stacktrace found for following nodes
        select_scatter_default_78: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_76, full_default_1, 0, 60);  select_scatter_default_76 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_139: "f32[]" = torch.ops.aten.mul.Tensor(select_617, 2);  select_617 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_79: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_139, 0, 60);  mul_139 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_38: "f32[100]" = torch.ops.aten.add.Tensor(add_37, select_scatter_default_79);  add_37 = select_scatter_default_79 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_620: "f32[]" = torch.ops.aten.select.int(select_scatter_default_78, 0, 59)
        
        # No stacktrace found for following nodes
        select_scatter_default_80: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_78, full_default_1, 0, 59);  select_scatter_default_78 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_140: "f32[]" = torch.ops.aten.mul.Tensor(select_620, 2);  select_620 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_81: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_140, 0, 59);  mul_140 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_39: "f32[100]" = torch.ops.aten.add.Tensor(add_38, select_scatter_default_81);  add_38 = select_scatter_default_81 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_623: "f32[]" = torch.ops.aten.select.int(select_scatter_default_80, 0, 58)
        
        # No stacktrace found for following nodes
        select_scatter_default_82: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_80, full_default_1, 0, 58);  select_scatter_default_80 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_141: "f32[]" = torch.ops.aten.mul.Tensor(select_623, 2);  select_623 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_83: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_141, 0, 58);  mul_141 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_40: "f32[100]" = torch.ops.aten.add.Tensor(add_39, select_scatter_default_83);  add_39 = select_scatter_default_83 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_626: "f32[]" = torch.ops.aten.select.int(select_scatter_default_82, 0, 57)
        
        # No stacktrace found for following nodes
        select_scatter_default_84: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_82, full_default_1, 0, 57);  select_scatter_default_82 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_142: "f32[]" = torch.ops.aten.mul.Tensor(select_626, 2);  select_626 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_85: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_142, 0, 57);  mul_142 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_41: "f32[100]" = torch.ops.aten.add.Tensor(add_40, select_scatter_default_85);  add_40 = select_scatter_default_85 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_629: "f32[]" = torch.ops.aten.select.int(select_scatter_default_84, 0, 56)
        
        # No stacktrace found for following nodes
        select_scatter_default_86: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_84, full_default_1, 0, 56);  select_scatter_default_84 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_143: "f32[]" = torch.ops.aten.mul.Tensor(select_629, 2);  select_629 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_87: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_143, 0, 56);  mul_143 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_42: "f32[100]" = torch.ops.aten.add.Tensor(add_41, select_scatter_default_87);  add_41 = select_scatter_default_87 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_632: "f32[]" = torch.ops.aten.select.int(select_scatter_default_86, 0, 55)
        
        # No stacktrace found for following nodes
        select_scatter_default_88: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_86, full_default_1, 0, 55);  select_scatter_default_86 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_144: "f32[]" = torch.ops.aten.mul.Tensor(select_632, 2);  select_632 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_89: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_144, 0, 55);  mul_144 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_43: "f32[100]" = torch.ops.aten.add.Tensor(add_42, select_scatter_default_89);  add_42 = select_scatter_default_89 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_635: "f32[]" = torch.ops.aten.select.int(select_scatter_default_88, 0, 54)
        
        # No stacktrace found for following nodes
        select_scatter_default_90: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_88, full_default_1, 0, 54);  select_scatter_default_88 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_145: "f32[]" = torch.ops.aten.mul.Tensor(select_635, 2);  select_635 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_91: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_145, 0, 54);  mul_145 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_44: "f32[100]" = torch.ops.aten.add.Tensor(add_43, select_scatter_default_91);  add_43 = select_scatter_default_91 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_638: "f32[]" = torch.ops.aten.select.int(select_scatter_default_90, 0, 53)
        
        # No stacktrace found for following nodes
        select_scatter_default_92: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_90, full_default_1, 0, 53);  select_scatter_default_90 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_146: "f32[]" = torch.ops.aten.mul.Tensor(select_638, 2);  select_638 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_93: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_146, 0, 53);  mul_146 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_45: "f32[100]" = torch.ops.aten.add.Tensor(add_44, select_scatter_default_93);  add_44 = select_scatter_default_93 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_641: "f32[]" = torch.ops.aten.select.int(select_scatter_default_92, 0, 52)
        
        # No stacktrace found for following nodes
        select_scatter_default_94: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_92, full_default_1, 0, 52);  select_scatter_default_92 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_147: "f32[]" = torch.ops.aten.mul.Tensor(select_641, 2);  select_641 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_95: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_147, 0, 52);  mul_147 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_46: "f32[100]" = torch.ops.aten.add.Tensor(add_45, select_scatter_default_95);  add_45 = select_scatter_default_95 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_644: "f32[]" = torch.ops.aten.select.int(select_scatter_default_94, 0, 51)
        
        # No stacktrace found for following nodes
        select_scatter_default_96: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_94, full_default_1, 0, 51);  select_scatter_default_94 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_148: "f32[]" = torch.ops.aten.mul.Tensor(select_644, 2);  select_644 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_97: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_148, 0, 51);  mul_148 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_47: "f32[100]" = torch.ops.aten.add.Tensor(add_46, select_scatter_default_97);  add_46 = select_scatter_default_97 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_647: "f32[]" = torch.ops.aten.select.int(select_scatter_default_96, 0, 50)
        
        # No stacktrace found for following nodes
        select_scatter_default_98: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_96, full_default_1, 0, 50);  select_scatter_default_96 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_149: "f32[]" = torch.ops.aten.mul.Tensor(select_647, 2);  select_647 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_99: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_149, 0, 50);  mul_149 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_48: "f32[100]" = torch.ops.aten.add.Tensor(add_47, select_scatter_default_99);  add_47 = select_scatter_default_99 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_650: "f32[]" = torch.ops.aten.select.int(select_scatter_default_98, 0, 49)
        
        # No stacktrace found for following nodes
        select_scatter_default_100: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_98, full_default_1, 0, 49);  select_scatter_default_98 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_150: "f32[]" = torch.ops.aten.mul.Tensor(select_650, 2);  select_650 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_101: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_150, 0, 49);  mul_150 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_49: "f32[100]" = torch.ops.aten.add.Tensor(add_48, select_scatter_default_101);  add_48 = select_scatter_default_101 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_653: "f32[]" = torch.ops.aten.select.int(select_scatter_default_100, 0, 48)
        
        # No stacktrace found for following nodes
        select_scatter_default_102: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_100, full_default_1, 0, 48);  select_scatter_default_100 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_151: "f32[]" = torch.ops.aten.mul.Tensor(select_653, 2);  select_653 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_103: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_151, 0, 48);  mul_151 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_50: "f32[100]" = torch.ops.aten.add.Tensor(add_49, select_scatter_default_103);  add_49 = select_scatter_default_103 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_656: "f32[]" = torch.ops.aten.select.int(select_scatter_default_102, 0, 47)
        
        # No stacktrace found for following nodes
        select_scatter_default_104: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_102, full_default_1, 0, 47);  select_scatter_default_102 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_152: "f32[]" = torch.ops.aten.mul.Tensor(select_656, 2);  select_656 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_105: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_152, 0, 47);  mul_152 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_51: "f32[100]" = torch.ops.aten.add.Tensor(add_50, select_scatter_default_105);  add_50 = select_scatter_default_105 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_659: "f32[]" = torch.ops.aten.select.int(select_scatter_default_104, 0, 46)
        
        # No stacktrace found for following nodes
        select_scatter_default_106: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_104, full_default_1, 0, 46);  select_scatter_default_104 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_153: "f32[]" = torch.ops.aten.mul.Tensor(select_659, 2);  select_659 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_107: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_153, 0, 46);  mul_153 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_52: "f32[100]" = torch.ops.aten.add.Tensor(add_51, select_scatter_default_107);  add_51 = select_scatter_default_107 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_662: "f32[]" = torch.ops.aten.select.int(select_scatter_default_106, 0, 45)
        
        # No stacktrace found for following nodes
        select_scatter_default_108: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_106, full_default_1, 0, 45);  select_scatter_default_106 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_154: "f32[]" = torch.ops.aten.mul.Tensor(select_662, 2);  select_662 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_109: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_154, 0, 45);  mul_154 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_53: "f32[100]" = torch.ops.aten.add.Tensor(add_52, select_scatter_default_109);  add_52 = select_scatter_default_109 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_665: "f32[]" = torch.ops.aten.select.int(select_scatter_default_108, 0, 44)
        
        # No stacktrace found for following nodes
        select_scatter_default_110: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_108, full_default_1, 0, 44);  select_scatter_default_108 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_155: "f32[]" = torch.ops.aten.mul.Tensor(select_665, 2);  select_665 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_111: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_155, 0, 44);  mul_155 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_54: "f32[100]" = torch.ops.aten.add.Tensor(add_53, select_scatter_default_111);  add_53 = select_scatter_default_111 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_668: "f32[]" = torch.ops.aten.select.int(select_scatter_default_110, 0, 43)
        
        # No stacktrace found for following nodes
        select_scatter_default_112: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_110, full_default_1, 0, 43);  select_scatter_default_110 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_156: "f32[]" = torch.ops.aten.mul.Tensor(select_668, 2);  select_668 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_113: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_156, 0, 43);  mul_156 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_55: "f32[100]" = torch.ops.aten.add.Tensor(add_54, select_scatter_default_113);  add_54 = select_scatter_default_113 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_671: "f32[]" = torch.ops.aten.select.int(select_scatter_default_112, 0, 42)
        
        # No stacktrace found for following nodes
        select_scatter_default_114: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_112, full_default_1, 0, 42);  select_scatter_default_112 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_157: "f32[]" = torch.ops.aten.mul.Tensor(select_671, 2);  select_671 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_115: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_157, 0, 42);  mul_157 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_56: "f32[100]" = torch.ops.aten.add.Tensor(add_55, select_scatter_default_115);  add_55 = select_scatter_default_115 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_674: "f32[]" = torch.ops.aten.select.int(select_scatter_default_114, 0, 41)
        
        # No stacktrace found for following nodes
        select_scatter_default_116: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_114, full_default_1, 0, 41);  select_scatter_default_114 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_158: "f32[]" = torch.ops.aten.mul.Tensor(select_674, 2);  select_674 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_117: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_158, 0, 41);  mul_158 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_57: "f32[100]" = torch.ops.aten.add.Tensor(add_56, select_scatter_default_117);  add_56 = select_scatter_default_117 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_677: "f32[]" = torch.ops.aten.select.int(select_scatter_default_116, 0, 40)
        
        # No stacktrace found for following nodes
        select_scatter_default_118: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_116, full_default_1, 0, 40);  select_scatter_default_116 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_159: "f32[]" = torch.ops.aten.mul.Tensor(select_677, 2);  select_677 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_119: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_159, 0, 40);  mul_159 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_58: "f32[100]" = torch.ops.aten.add.Tensor(add_57, select_scatter_default_119);  add_57 = select_scatter_default_119 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_680: "f32[]" = torch.ops.aten.select.int(select_scatter_default_118, 0, 39)
        
        # No stacktrace found for following nodes
        select_scatter_default_120: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_118, full_default_1, 0, 39);  select_scatter_default_118 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_160: "f32[]" = torch.ops.aten.mul.Tensor(select_680, 2);  select_680 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_121: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_160, 0, 39);  mul_160 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_59: "f32[100]" = torch.ops.aten.add.Tensor(add_58, select_scatter_default_121);  add_58 = select_scatter_default_121 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_683: "f32[]" = torch.ops.aten.select.int(select_scatter_default_120, 0, 38)
        
        # No stacktrace found for following nodes
        select_scatter_default_122: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_120, full_default_1, 0, 38);  select_scatter_default_120 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_161: "f32[]" = torch.ops.aten.mul.Tensor(select_683, 2);  select_683 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_123: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_161, 0, 38);  mul_161 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_60: "f32[100]" = torch.ops.aten.add.Tensor(add_59, select_scatter_default_123);  add_59 = select_scatter_default_123 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_686: "f32[]" = torch.ops.aten.select.int(select_scatter_default_122, 0, 37)
        
        # No stacktrace found for following nodes
        select_scatter_default_124: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_122, full_default_1, 0, 37);  select_scatter_default_122 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_162: "f32[]" = torch.ops.aten.mul.Tensor(select_686, 2);  select_686 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_125: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_162, 0, 37);  mul_162 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_61: "f32[100]" = torch.ops.aten.add.Tensor(add_60, select_scatter_default_125);  add_60 = select_scatter_default_125 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_689: "f32[]" = torch.ops.aten.select.int(select_scatter_default_124, 0, 36)
        
        # No stacktrace found for following nodes
        select_scatter_default_126: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_124, full_default_1, 0, 36);  select_scatter_default_124 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_163: "f32[]" = torch.ops.aten.mul.Tensor(select_689, 2);  select_689 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_127: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_163, 0, 36);  mul_163 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_62: "f32[100]" = torch.ops.aten.add.Tensor(add_61, select_scatter_default_127);  add_61 = select_scatter_default_127 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_692: "f32[]" = torch.ops.aten.select.int(select_scatter_default_126, 0, 35)
        
        # No stacktrace found for following nodes
        select_scatter_default_128: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_126, full_default_1, 0, 35);  select_scatter_default_126 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_164: "f32[]" = torch.ops.aten.mul.Tensor(select_692, 2);  select_692 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_129: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_164, 0, 35);  mul_164 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_63: "f32[100]" = torch.ops.aten.add.Tensor(add_62, select_scatter_default_129);  add_62 = select_scatter_default_129 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_695: "f32[]" = torch.ops.aten.select.int(select_scatter_default_128, 0, 34)
        
        # No stacktrace found for following nodes
        select_scatter_default_130: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_128, full_default_1, 0, 34);  select_scatter_default_128 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_165: "f32[]" = torch.ops.aten.mul.Tensor(select_695, 2);  select_695 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_131: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_165, 0, 34);  mul_165 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_64: "f32[100]" = torch.ops.aten.add.Tensor(add_63, select_scatter_default_131);  add_63 = select_scatter_default_131 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_698: "f32[]" = torch.ops.aten.select.int(select_scatter_default_130, 0, 33)
        
        # No stacktrace found for following nodes
        select_scatter_default_132: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_130, full_default_1, 0, 33);  select_scatter_default_130 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_166: "f32[]" = torch.ops.aten.mul.Tensor(select_698, 2);  select_698 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_133: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_166, 0, 33);  mul_166 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_65: "f32[100]" = torch.ops.aten.add.Tensor(add_64, select_scatter_default_133);  add_64 = select_scatter_default_133 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_701: "f32[]" = torch.ops.aten.select.int(select_scatter_default_132, 0, 32)
        
        # No stacktrace found for following nodes
        select_scatter_default_134: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_132, full_default_1, 0, 32);  select_scatter_default_132 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_167: "f32[]" = torch.ops.aten.mul.Tensor(select_701, 2);  select_701 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_135: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_167, 0, 32);  mul_167 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_66: "f32[100]" = torch.ops.aten.add.Tensor(add_65, select_scatter_default_135);  add_65 = select_scatter_default_135 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_704: "f32[]" = torch.ops.aten.select.int(select_scatter_default_134, 0, 31)
        
        # No stacktrace found for following nodes
        select_scatter_default_136: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_134, full_default_1, 0, 31);  select_scatter_default_134 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_168: "f32[]" = torch.ops.aten.mul.Tensor(select_704, 2);  select_704 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_137: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_168, 0, 31);  mul_168 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_67: "f32[100]" = torch.ops.aten.add.Tensor(add_66, select_scatter_default_137);  add_66 = select_scatter_default_137 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_707: "f32[]" = torch.ops.aten.select.int(select_scatter_default_136, 0, 30)
        
        # No stacktrace found for following nodes
        select_scatter_default_138: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_136, full_default_1, 0, 30);  select_scatter_default_136 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_169: "f32[]" = torch.ops.aten.mul.Tensor(select_707, 2);  select_707 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_139: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_169, 0, 30);  mul_169 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_68: "f32[100]" = torch.ops.aten.add.Tensor(add_67, select_scatter_default_139);  add_67 = select_scatter_default_139 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_710: "f32[]" = torch.ops.aten.select.int(select_scatter_default_138, 0, 29)
        
        # No stacktrace found for following nodes
        select_scatter_default_140: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_138, full_default_1, 0, 29);  select_scatter_default_138 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_170: "f32[]" = torch.ops.aten.mul.Tensor(select_710, 2);  select_710 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_141: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_170, 0, 29);  mul_170 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_69: "f32[100]" = torch.ops.aten.add.Tensor(add_68, select_scatter_default_141);  add_68 = select_scatter_default_141 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_713: "f32[]" = torch.ops.aten.select.int(select_scatter_default_140, 0, 28)
        
        # No stacktrace found for following nodes
        select_scatter_default_142: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_140, full_default_1, 0, 28);  select_scatter_default_140 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_171: "f32[]" = torch.ops.aten.mul.Tensor(select_713, 2);  select_713 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_143: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_171, 0, 28);  mul_171 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_70: "f32[100]" = torch.ops.aten.add.Tensor(add_69, select_scatter_default_143);  add_69 = select_scatter_default_143 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_716: "f32[]" = torch.ops.aten.select.int(select_scatter_default_142, 0, 27)
        
        # No stacktrace found for following nodes
        select_scatter_default_144: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_142, full_default_1, 0, 27);  select_scatter_default_142 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_172: "f32[]" = torch.ops.aten.mul.Tensor(select_716, 2);  select_716 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_145: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_172, 0, 27);  mul_172 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_71: "f32[100]" = torch.ops.aten.add.Tensor(add_70, select_scatter_default_145);  add_70 = select_scatter_default_145 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_719: "f32[]" = torch.ops.aten.select.int(select_scatter_default_144, 0, 26)
        
        # No stacktrace found for following nodes
        select_scatter_default_146: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_144, full_default_1, 0, 26);  select_scatter_default_144 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_173: "f32[]" = torch.ops.aten.mul.Tensor(select_719, 2);  select_719 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_147: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_173, 0, 26);  mul_173 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_72: "f32[100]" = torch.ops.aten.add.Tensor(add_71, select_scatter_default_147);  add_71 = select_scatter_default_147 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_722: "f32[]" = torch.ops.aten.select.int(select_scatter_default_146, 0, 25)
        
        # No stacktrace found for following nodes
        select_scatter_default_148: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_146, full_default_1, 0, 25);  select_scatter_default_146 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_174: "f32[]" = torch.ops.aten.mul.Tensor(select_722, 2);  select_722 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_149: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_174, 0, 25);  mul_174 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_73: "f32[100]" = torch.ops.aten.add.Tensor(add_72, select_scatter_default_149);  add_72 = select_scatter_default_149 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_725: "f32[]" = torch.ops.aten.select.int(select_scatter_default_148, 0, 24)
        
        # No stacktrace found for following nodes
        select_scatter_default_150: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_148, full_default_1, 0, 24);  select_scatter_default_148 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_175: "f32[]" = torch.ops.aten.mul.Tensor(select_725, 2);  select_725 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_151: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_175, 0, 24);  mul_175 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_74: "f32[100]" = torch.ops.aten.add.Tensor(add_73, select_scatter_default_151);  add_73 = select_scatter_default_151 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_728: "f32[]" = torch.ops.aten.select.int(select_scatter_default_150, 0, 23)
        
        # No stacktrace found for following nodes
        select_scatter_default_152: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_150, full_default_1, 0, 23);  select_scatter_default_150 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_176: "f32[]" = torch.ops.aten.mul.Tensor(select_728, 2);  select_728 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_153: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_176, 0, 23);  mul_176 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_75: "f32[100]" = torch.ops.aten.add.Tensor(add_74, select_scatter_default_153);  add_74 = select_scatter_default_153 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_731: "f32[]" = torch.ops.aten.select.int(select_scatter_default_152, 0, 22)
        
        # No stacktrace found for following nodes
        select_scatter_default_154: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_152, full_default_1, 0, 22);  select_scatter_default_152 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_177: "f32[]" = torch.ops.aten.mul.Tensor(select_731, 2);  select_731 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_155: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_177, 0, 22);  mul_177 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_76: "f32[100]" = torch.ops.aten.add.Tensor(add_75, select_scatter_default_155);  add_75 = select_scatter_default_155 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_734: "f32[]" = torch.ops.aten.select.int(select_scatter_default_154, 0, 21)
        
        # No stacktrace found for following nodes
        select_scatter_default_156: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_154, full_default_1, 0, 21);  select_scatter_default_154 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_178: "f32[]" = torch.ops.aten.mul.Tensor(select_734, 2);  select_734 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_157: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_178, 0, 21);  mul_178 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_77: "f32[100]" = torch.ops.aten.add.Tensor(add_76, select_scatter_default_157);  add_76 = select_scatter_default_157 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_737: "f32[]" = torch.ops.aten.select.int(select_scatter_default_156, 0, 20)
        
        # No stacktrace found for following nodes
        select_scatter_default_158: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_156, full_default_1, 0, 20);  select_scatter_default_156 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_179: "f32[]" = torch.ops.aten.mul.Tensor(select_737, 2);  select_737 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_159: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_179, 0, 20);  mul_179 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_78: "f32[100]" = torch.ops.aten.add.Tensor(add_77, select_scatter_default_159);  add_77 = select_scatter_default_159 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_740: "f32[]" = torch.ops.aten.select.int(select_scatter_default_158, 0, 19)
        
        # No stacktrace found for following nodes
        select_scatter_default_160: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_158, full_default_1, 0, 19);  select_scatter_default_158 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_180: "f32[]" = torch.ops.aten.mul.Tensor(select_740, 2);  select_740 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_161: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_180, 0, 19);  mul_180 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_79: "f32[100]" = torch.ops.aten.add.Tensor(add_78, select_scatter_default_161);  add_78 = select_scatter_default_161 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_743: "f32[]" = torch.ops.aten.select.int(select_scatter_default_160, 0, 18)
        
        # No stacktrace found for following nodes
        select_scatter_default_162: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_160, full_default_1, 0, 18);  select_scatter_default_160 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_181: "f32[]" = torch.ops.aten.mul.Tensor(select_743, 2);  select_743 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_163: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_181, 0, 18);  mul_181 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_80: "f32[100]" = torch.ops.aten.add.Tensor(add_79, select_scatter_default_163);  add_79 = select_scatter_default_163 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_746: "f32[]" = torch.ops.aten.select.int(select_scatter_default_162, 0, 17)
        
        # No stacktrace found for following nodes
        select_scatter_default_164: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_162, full_default_1, 0, 17);  select_scatter_default_162 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_182: "f32[]" = torch.ops.aten.mul.Tensor(select_746, 2);  select_746 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_165: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_182, 0, 17);  mul_182 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_81: "f32[100]" = torch.ops.aten.add.Tensor(add_80, select_scatter_default_165);  add_80 = select_scatter_default_165 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_749: "f32[]" = torch.ops.aten.select.int(select_scatter_default_164, 0, 16)
        
        # No stacktrace found for following nodes
        select_scatter_default_166: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_164, full_default_1, 0, 16);  select_scatter_default_164 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_183: "f32[]" = torch.ops.aten.mul.Tensor(select_749, 2);  select_749 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_167: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_183, 0, 16);  mul_183 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_82: "f32[100]" = torch.ops.aten.add.Tensor(add_81, select_scatter_default_167);  add_81 = select_scatter_default_167 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_752: "f32[]" = torch.ops.aten.select.int(select_scatter_default_166, 0, 15)
        
        # No stacktrace found for following nodes
        select_scatter_default_168: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_166, full_default_1, 0, 15);  select_scatter_default_166 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_184: "f32[]" = torch.ops.aten.mul.Tensor(select_752, 2);  select_752 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_169: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_184, 0, 15);  mul_184 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_83: "f32[100]" = torch.ops.aten.add.Tensor(add_82, select_scatter_default_169);  add_82 = select_scatter_default_169 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_755: "f32[]" = torch.ops.aten.select.int(select_scatter_default_168, 0, 14)
        
        # No stacktrace found for following nodes
        select_scatter_default_170: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_168, full_default_1, 0, 14);  select_scatter_default_168 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_185: "f32[]" = torch.ops.aten.mul.Tensor(select_755, 2);  select_755 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_171: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_185, 0, 14);  mul_185 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_84: "f32[100]" = torch.ops.aten.add.Tensor(add_83, select_scatter_default_171);  add_83 = select_scatter_default_171 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_758: "f32[]" = torch.ops.aten.select.int(select_scatter_default_170, 0, 13)
        
        # No stacktrace found for following nodes
        select_scatter_default_172: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_170, full_default_1, 0, 13);  select_scatter_default_170 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_186: "f32[]" = torch.ops.aten.mul.Tensor(select_758, 2);  select_758 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_173: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_186, 0, 13);  mul_186 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_85: "f32[100]" = torch.ops.aten.add.Tensor(add_84, select_scatter_default_173);  add_84 = select_scatter_default_173 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_761: "f32[]" = torch.ops.aten.select.int(select_scatter_default_172, 0, 12)
        
        # No stacktrace found for following nodes
        select_scatter_default_174: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_172, full_default_1, 0, 12);  select_scatter_default_172 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_187: "f32[]" = torch.ops.aten.mul.Tensor(select_761, 2);  select_761 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_175: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_187, 0, 12);  mul_187 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_86: "f32[100]" = torch.ops.aten.add.Tensor(add_85, select_scatter_default_175);  add_85 = select_scatter_default_175 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_764: "f32[]" = torch.ops.aten.select.int(select_scatter_default_174, 0, 11)
        
        # No stacktrace found for following nodes
        select_scatter_default_176: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_174, full_default_1, 0, 11);  select_scatter_default_174 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_188: "f32[]" = torch.ops.aten.mul.Tensor(select_764, 2);  select_764 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_177: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_188, 0, 11);  mul_188 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_87: "f32[100]" = torch.ops.aten.add.Tensor(add_86, select_scatter_default_177);  add_86 = select_scatter_default_177 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_767: "f32[]" = torch.ops.aten.select.int(select_scatter_default_176, 0, 10)
        
        # No stacktrace found for following nodes
        select_scatter_default_178: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_176, full_default_1, 0, 10);  select_scatter_default_176 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_189: "f32[]" = torch.ops.aten.mul.Tensor(select_767, 2);  select_767 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_179: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_189, 0, 10);  mul_189 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_88: "f32[100]" = torch.ops.aten.add.Tensor(add_87, select_scatter_default_179);  add_87 = select_scatter_default_179 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_770: "f32[]" = torch.ops.aten.select.int(select_scatter_default_178, 0, 9)
        
        # No stacktrace found for following nodes
        select_scatter_default_180: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_178, full_default_1, 0, 9);  select_scatter_default_178 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_190: "f32[]" = torch.ops.aten.mul.Tensor(select_770, 2);  select_770 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_181: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_190, 0, 9);  mul_190 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_89: "f32[100]" = torch.ops.aten.add.Tensor(add_88, select_scatter_default_181);  add_88 = select_scatter_default_181 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_773: "f32[]" = torch.ops.aten.select.int(select_scatter_default_180, 0, 8)
        
        # No stacktrace found for following nodes
        select_scatter_default_182: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_180, full_default_1, 0, 8);  select_scatter_default_180 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_191: "f32[]" = torch.ops.aten.mul.Tensor(select_773, 2);  select_773 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_183: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_191, 0, 8);  mul_191 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_90: "f32[100]" = torch.ops.aten.add.Tensor(add_89, select_scatter_default_183);  add_89 = select_scatter_default_183 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_776: "f32[]" = torch.ops.aten.select.int(select_scatter_default_182, 0, 7)
        
        # No stacktrace found for following nodes
        select_scatter_default_184: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_182, full_default_1, 0, 7);  select_scatter_default_182 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_192: "f32[]" = torch.ops.aten.mul.Tensor(select_776, 2);  select_776 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_185: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_192, 0, 7);  mul_192 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_91: "f32[100]" = torch.ops.aten.add.Tensor(add_90, select_scatter_default_185);  add_90 = select_scatter_default_185 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_779: "f32[]" = torch.ops.aten.select.int(select_scatter_default_184, 0, 6)
        
        # No stacktrace found for following nodes
        select_scatter_default_186: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_184, full_default_1, 0, 6);  select_scatter_default_184 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_193: "f32[]" = torch.ops.aten.mul.Tensor(select_779, 2);  select_779 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_187: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_193, 0, 6);  mul_193 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_92: "f32[100]" = torch.ops.aten.add.Tensor(add_91, select_scatter_default_187);  add_91 = select_scatter_default_187 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_782: "f32[]" = torch.ops.aten.select.int(select_scatter_default_186, 0, 5)
        
        # No stacktrace found for following nodes
        select_scatter_default_188: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_186, full_default_1, 0, 5);  select_scatter_default_186 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_194: "f32[]" = torch.ops.aten.mul.Tensor(select_782, 2);  select_782 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_189: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_194, 0, 5);  mul_194 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_93: "f32[100]" = torch.ops.aten.add.Tensor(add_92, select_scatter_default_189);  add_92 = select_scatter_default_189 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_785: "f32[]" = torch.ops.aten.select.int(select_scatter_default_188, 0, 4)
        
        # No stacktrace found for following nodes
        select_scatter_default_190: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_188, full_default_1, 0, 4);  select_scatter_default_188 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_195: "f32[]" = torch.ops.aten.mul.Tensor(select_785, 2);  select_785 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_191: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_195, 0, 4);  mul_195 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_94: "f32[100]" = torch.ops.aten.add.Tensor(add_93, select_scatter_default_191);  add_93 = select_scatter_default_191 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_788: "f32[]" = torch.ops.aten.select.int(select_scatter_default_190, 0, 3)
        
        # No stacktrace found for following nodes
        select_scatter_default_192: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_190, full_default_1, 0, 3);  select_scatter_default_190 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_196: "f32[]" = torch.ops.aten.mul.Tensor(select_788, 2);  select_788 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_193: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_196, 0, 3);  mul_196 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_95: "f32[100]" = torch.ops.aten.add.Tensor(add_94, select_scatter_default_193);  add_94 = select_scatter_default_193 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_791: "f32[]" = torch.ops.aten.select.int(select_scatter_default_192, 0, 2)
        
        # No stacktrace found for following nodes
        select_scatter_default_194: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_192, full_default_1, 0, 2);  select_scatter_default_192 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_197: "f32[]" = torch.ops.aten.mul.Tensor(select_791, 2);  select_791 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_195: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_197, 0, 2);  mul_197 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_96: "f32[100]" = torch.ops.aten.add.Tensor(add_95, select_scatter_default_195);  add_95 = select_scatter_default_195 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_794: "f32[]" = torch.ops.aten.select.int(select_scatter_default_194, 0, 1)
        
        # No stacktrace found for following nodes
        select_scatter_default_196: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_default_194, full_default_1, 0, 1);  select_scatter_default_194 = full_default_1 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        mul_198: "f32[]" = torch.ops.aten.mul.Tensor(select_794, 2);  select_794 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_197: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_198, 0, 1);  mul_198 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_97: "f32[100]" = torch.ops.aten.add.Tensor(add_96, select_scatter_default_197);  add_96 = select_scatter_default_197 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select_797: "f32[]" = torch.ops.aten.select.int(select_scatter_default_196, 0, 0);  select_scatter_default_196 = None
        mul_199: "f32[]" = torch.ops.aten.mul.Tensor(select_797, 2);  select_797 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_198: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul_199, 0, 0);  full_default = mul_199 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        add_98: "f32[100]" = torch.ops.aten.add.Tensor(add_97, select_scatter_default_198);  add_97 = select_scatter_default_198 = None
        return (add_98,)
        