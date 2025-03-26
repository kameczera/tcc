class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:4 in explicit_loop, code: out = torch.zeros_like(x)
        full_default: "f32[100]" = torch.ops.aten.full.default([100], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:6 in explicit_loop, code: out[i] = x[i] * 2  # Operação iterativa
        select: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 0)
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select, 2);  select = None
        select_scatter: "f32[100]" = torch.ops.aten.select_scatter.default(full_default, mul, 0, 0);  mul = None
        select_4: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 1)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(select_4, 2);  select_4 = None
        select_scatter_1: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter, mul_1, 0, 1);  select_scatter = mul_1 = None
        select_9: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 2)
        mul_2: "f32[]" = torch.ops.aten.mul.Tensor(select_9, 2);  select_9 = None
        select_scatter_2: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_1, mul_2, 0, 2);  select_scatter_1 = mul_2 = None
        select_14: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 3)
        mul_3: "f32[]" = torch.ops.aten.mul.Tensor(select_14, 2);  select_14 = None
        select_scatter_3: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_2, mul_3, 0, 3);  select_scatter_2 = mul_3 = None
        select_19: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 4)
        mul_4: "f32[]" = torch.ops.aten.mul.Tensor(select_19, 2);  select_19 = None
        select_scatter_4: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_3, mul_4, 0, 4);  select_scatter_3 = mul_4 = None
        select_24: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 5)
        mul_5: "f32[]" = torch.ops.aten.mul.Tensor(select_24, 2);  select_24 = None
        select_scatter_5: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_4, mul_5, 0, 5);  select_scatter_4 = mul_5 = None
        select_29: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 6)
        mul_6: "f32[]" = torch.ops.aten.mul.Tensor(select_29, 2);  select_29 = None
        select_scatter_6: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_5, mul_6, 0, 6);  select_scatter_5 = mul_6 = None
        select_34: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 7)
        mul_7: "f32[]" = torch.ops.aten.mul.Tensor(select_34, 2);  select_34 = None
        select_scatter_7: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_6, mul_7, 0, 7);  select_scatter_6 = mul_7 = None
        select_39: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 8)
        mul_8: "f32[]" = torch.ops.aten.mul.Tensor(select_39, 2);  select_39 = None
        select_scatter_8: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_7, mul_8, 0, 8);  select_scatter_7 = mul_8 = None
        select_44: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 9)
        mul_9: "f32[]" = torch.ops.aten.mul.Tensor(select_44, 2);  select_44 = None
        select_scatter_9: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_8, mul_9, 0, 9);  select_scatter_8 = mul_9 = None
        select_49: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 10)
        mul_10: "f32[]" = torch.ops.aten.mul.Tensor(select_49, 2);  select_49 = None
        select_scatter_10: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_9, mul_10, 0, 10);  select_scatter_9 = mul_10 = None
        select_54: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 11)
        mul_11: "f32[]" = torch.ops.aten.mul.Tensor(select_54, 2);  select_54 = None
        select_scatter_11: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_10, mul_11, 0, 11);  select_scatter_10 = mul_11 = None
        select_59: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 12)
        mul_12: "f32[]" = torch.ops.aten.mul.Tensor(select_59, 2);  select_59 = None
        select_scatter_12: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_11, mul_12, 0, 12);  select_scatter_11 = mul_12 = None
        select_64: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 13)
        mul_13: "f32[]" = torch.ops.aten.mul.Tensor(select_64, 2);  select_64 = None
        select_scatter_13: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_12, mul_13, 0, 13);  select_scatter_12 = mul_13 = None
        select_69: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 14)
        mul_14: "f32[]" = torch.ops.aten.mul.Tensor(select_69, 2);  select_69 = None
        select_scatter_14: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_13, mul_14, 0, 14);  select_scatter_13 = mul_14 = None
        select_74: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 15)
        mul_15: "f32[]" = torch.ops.aten.mul.Tensor(select_74, 2);  select_74 = None
        select_scatter_15: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_14, mul_15, 0, 15);  select_scatter_14 = mul_15 = None
        select_79: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 16)
        mul_16: "f32[]" = torch.ops.aten.mul.Tensor(select_79, 2);  select_79 = None
        select_scatter_16: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_15, mul_16, 0, 16);  select_scatter_15 = mul_16 = None
        select_84: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 17)
        mul_17: "f32[]" = torch.ops.aten.mul.Tensor(select_84, 2);  select_84 = None
        select_scatter_17: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_16, mul_17, 0, 17);  select_scatter_16 = mul_17 = None
        select_89: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 18)
        mul_18: "f32[]" = torch.ops.aten.mul.Tensor(select_89, 2);  select_89 = None
        select_scatter_18: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_17, mul_18, 0, 18);  select_scatter_17 = mul_18 = None
        select_94: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 19)
        mul_19: "f32[]" = torch.ops.aten.mul.Tensor(select_94, 2);  select_94 = None
        select_scatter_19: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_18, mul_19, 0, 19);  select_scatter_18 = mul_19 = None
        select_99: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 20)
        mul_20: "f32[]" = torch.ops.aten.mul.Tensor(select_99, 2);  select_99 = None
        select_scatter_20: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_19, mul_20, 0, 20);  select_scatter_19 = mul_20 = None
        select_104: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 21)
        mul_21: "f32[]" = torch.ops.aten.mul.Tensor(select_104, 2);  select_104 = None
        select_scatter_21: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_20, mul_21, 0, 21);  select_scatter_20 = mul_21 = None
        select_109: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 22)
        mul_22: "f32[]" = torch.ops.aten.mul.Tensor(select_109, 2);  select_109 = None
        select_scatter_22: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_21, mul_22, 0, 22);  select_scatter_21 = mul_22 = None
        select_114: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 23)
        mul_23: "f32[]" = torch.ops.aten.mul.Tensor(select_114, 2);  select_114 = None
        select_scatter_23: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_22, mul_23, 0, 23);  select_scatter_22 = mul_23 = None
        select_119: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 24)
        mul_24: "f32[]" = torch.ops.aten.mul.Tensor(select_119, 2);  select_119 = None
        select_scatter_24: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_23, mul_24, 0, 24);  select_scatter_23 = mul_24 = None
        select_124: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 25)
        mul_25: "f32[]" = torch.ops.aten.mul.Tensor(select_124, 2);  select_124 = None
        select_scatter_25: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_24, mul_25, 0, 25);  select_scatter_24 = mul_25 = None
        select_129: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 26)
        mul_26: "f32[]" = torch.ops.aten.mul.Tensor(select_129, 2);  select_129 = None
        select_scatter_26: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_25, mul_26, 0, 26);  select_scatter_25 = mul_26 = None
        select_134: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 27)
        mul_27: "f32[]" = torch.ops.aten.mul.Tensor(select_134, 2);  select_134 = None
        select_scatter_27: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_26, mul_27, 0, 27);  select_scatter_26 = mul_27 = None
        select_139: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 28)
        mul_28: "f32[]" = torch.ops.aten.mul.Tensor(select_139, 2);  select_139 = None
        select_scatter_28: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_27, mul_28, 0, 28);  select_scatter_27 = mul_28 = None
        select_144: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 29)
        mul_29: "f32[]" = torch.ops.aten.mul.Tensor(select_144, 2);  select_144 = None
        select_scatter_29: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_28, mul_29, 0, 29);  select_scatter_28 = mul_29 = None
        select_149: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 30)
        mul_30: "f32[]" = torch.ops.aten.mul.Tensor(select_149, 2);  select_149 = None
        select_scatter_30: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_29, mul_30, 0, 30);  select_scatter_29 = mul_30 = None
        select_154: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 31)
        mul_31: "f32[]" = torch.ops.aten.mul.Tensor(select_154, 2);  select_154 = None
        select_scatter_31: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_30, mul_31, 0, 31);  select_scatter_30 = mul_31 = None
        select_159: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 32)
        mul_32: "f32[]" = torch.ops.aten.mul.Tensor(select_159, 2);  select_159 = None
        select_scatter_32: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_31, mul_32, 0, 32);  select_scatter_31 = mul_32 = None
        select_164: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 33)
        mul_33: "f32[]" = torch.ops.aten.mul.Tensor(select_164, 2);  select_164 = None
        select_scatter_33: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_32, mul_33, 0, 33);  select_scatter_32 = mul_33 = None
        select_169: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 34)
        mul_34: "f32[]" = torch.ops.aten.mul.Tensor(select_169, 2);  select_169 = None
        select_scatter_34: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_33, mul_34, 0, 34);  select_scatter_33 = mul_34 = None
        select_174: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 35)
        mul_35: "f32[]" = torch.ops.aten.mul.Tensor(select_174, 2);  select_174 = None
        select_scatter_35: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_34, mul_35, 0, 35);  select_scatter_34 = mul_35 = None
        select_179: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 36)
        mul_36: "f32[]" = torch.ops.aten.mul.Tensor(select_179, 2);  select_179 = None
        select_scatter_36: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_35, mul_36, 0, 36);  select_scatter_35 = mul_36 = None
        select_184: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 37)
        mul_37: "f32[]" = torch.ops.aten.mul.Tensor(select_184, 2);  select_184 = None
        select_scatter_37: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_36, mul_37, 0, 37);  select_scatter_36 = mul_37 = None
        select_189: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 38)
        mul_38: "f32[]" = torch.ops.aten.mul.Tensor(select_189, 2);  select_189 = None
        select_scatter_38: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_37, mul_38, 0, 38);  select_scatter_37 = mul_38 = None
        select_194: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 39)
        mul_39: "f32[]" = torch.ops.aten.mul.Tensor(select_194, 2);  select_194 = None
        select_scatter_39: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_38, mul_39, 0, 39);  select_scatter_38 = mul_39 = None
        select_199: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 40)
        mul_40: "f32[]" = torch.ops.aten.mul.Tensor(select_199, 2);  select_199 = None
        select_scatter_40: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_39, mul_40, 0, 40);  select_scatter_39 = mul_40 = None
        select_204: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 41)
        mul_41: "f32[]" = torch.ops.aten.mul.Tensor(select_204, 2);  select_204 = None
        select_scatter_41: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_40, mul_41, 0, 41);  select_scatter_40 = mul_41 = None
        select_209: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 42)
        mul_42: "f32[]" = torch.ops.aten.mul.Tensor(select_209, 2);  select_209 = None
        select_scatter_42: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_41, mul_42, 0, 42);  select_scatter_41 = mul_42 = None
        select_214: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 43)
        mul_43: "f32[]" = torch.ops.aten.mul.Tensor(select_214, 2);  select_214 = None
        select_scatter_43: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_42, mul_43, 0, 43);  select_scatter_42 = mul_43 = None
        select_219: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 44)
        mul_44: "f32[]" = torch.ops.aten.mul.Tensor(select_219, 2);  select_219 = None
        select_scatter_44: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_43, mul_44, 0, 44);  select_scatter_43 = mul_44 = None
        select_224: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 45)
        mul_45: "f32[]" = torch.ops.aten.mul.Tensor(select_224, 2);  select_224 = None
        select_scatter_45: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_44, mul_45, 0, 45);  select_scatter_44 = mul_45 = None
        select_229: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 46)
        mul_46: "f32[]" = torch.ops.aten.mul.Tensor(select_229, 2);  select_229 = None
        select_scatter_46: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_45, mul_46, 0, 46);  select_scatter_45 = mul_46 = None
        select_234: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 47)
        mul_47: "f32[]" = torch.ops.aten.mul.Tensor(select_234, 2);  select_234 = None
        select_scatter_47: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_46, mul_47, 0, 47);  select_scatter_46 = mul_47 = None
        select_239: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 48)
        mul_48: "f32[]" = torch.ops.aten.mul.Tensor(select_239, 2);  select_239 = None
        select_scatter_48: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_47, mul_48, 0, 48);  select_scatter_47 = mul_48 = None
        select_244: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 49)
        mul_49: "f32[]" = torch.ops.aten.mul.Tensor(select_244, 2);  select_244 = None
        select_scatter_49: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_48, mul_49, 0, 49);  select_scatter_48 = mul_49 = None
        select_249: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 50)
        mul_50: "f32[]" = torch.ops.aten.mul.Tensor(select_249, 2);  select_249 = None
        select_scatter_50: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_49, mul_50, 0, 50);  select_scatter_49 = mul_50 = None
        select_254: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 51)
        mul_51: "f32[]" = torch.ops.aten.mul.Tensor(select_254, 2);  select_254 = None
        select_scatter_51: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_50, mul_51, 0, 51);  select_scatter_50 = mul_51 = None
        select_259: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 52)
        mul_52: "f32[]" = torch.ops.aten.mul.Tensor(select_259, 2);  select_259 = None
        select_scatter_52: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_51, mul_52, 0, 52);  select_scatter_51 = mul_52 = None
        select_264: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 53)
        mul_53: "f32[]" = torch.ops.aten.mul.Tensor(select_264, 2);  select_264 = None
        select_scatter_53: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_52, mul_53, 0, 53);  select_scatter_52 = mul_53 = None
        select_269: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 54)
        mul_54: "f32[]" = torch.ops.aten.mul.Tensor(select_269, 2);  select_269 = None
        select_scatter_54: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_53, mul_54, 0, 54);  select_scatter_53 = mul_54 = None
        select_274: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 55)
        mul_55: "f32[]" = torch.ops.aten.mul.Tensor(select_274, 2);  select_274 = None
        select_scatter_55: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_54, mul_55, 0, 55);  select_scatter_54 = mul_55 = None
        select_279: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 56)
        mul_56: "f32[]" = torch.ops.aten.mul.Tensor(select_279, 2);  select_279 = None
        select_scatter_56: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_55, mul_56, 0, 56);  select_scatter_55 = mul_56 = None
        select_284: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 57)
        mul_57: "f32[]" = torch.ops.aten.mul.Tensor(select_284, 2);  select_284 = None
        select_scatter_57: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_56, mul_57, 0, 57);  select_scatter_56 = mul_57 = None
        select_289: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 58)
        mul_58: "f32[]" = torch.ops.aten.mul.Tensor(select_289, 2);  select_289 = None
        select_scatter_58: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_57, mul_58, 0, 58);  select_scatter_57 = mul_58 = None
        select_294: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 59)
        mul_59: "f32[]" = torch.ops.aten.mul.Tensor(select_294, 2);  select_294 = None
        select_scatter_59: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_58, mul_59, 0, 59);  select_scatter_58 = mul_59 = None
        select_299: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 60)
        mul_60: "f32[]" = torch.ops.aten.mul.Tensor(select_299, 2);  select_299 = None
        select_scatter_60: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_59, mul_60, 0, 60);  select_scatter_59 = mul_60 = None
        select_304: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 61)
        mul_61: "f32[]" = torch.ops.aten.mul.Tensor(select_304, 2);  select_304 = None
        select_scatter_61: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_60, mul_61, 0, 61);  select_scatter_60 = mul_61 = None
        select_309: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 62)
        mul_62: "f32[]" = torch.ops.aten.mul.Tensor(select_309, 2);  select_309 = None
        select_scatter_62: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_61, mul_62, 0, 62);  select_scatter_61 = mul_62 = None
        select_314: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 63)
        mul_63: "f32[]" = torch.ops.aten.mul.Tensor(select_314, 2);  select_314 = None
        select_scatter_63: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_62, mul_63, 0, 63);  select_scatter_62 = mul_63 = None
        select_319: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 64)
        mul_64: "f32[]" = torch.ops.aten.mul.Tensor(select_319, 2);  select_319 = None
        select_scatter_64: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_63, mul_64, 0, 64);  select_scatter_63 = mul_64 = None
        select_324: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 65)
        mul_65: "f32[]" = torch.ops.aten.mul.Tensor(select_324, 2);  select_324 = None
        select_scatter_65: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_64, mul_65, 0, 65);  select_scatter_64 = mul_65 = None
        select_329: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 66)
        mul_66: "f32[]" = torch.ops.aten.mul.Tensor(select_329, 2);  select_329 = None
        select_scatter_66: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_65, mul_66, 0, 66);  select_scatter_65 = mul_66 = None
        select_334: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 67)
        mul_67: "f32[]" = torch.ops.aten.mul.Tensor(select_334, 2);  select_334 = None
        select_scatter_67: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_66, mul_67, 0, 67);  select_scatter_66 = mul_67 = None
        select_339: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 68)
        mul_68: "f32[]" = torch.ops.aten.mul.Tensor(select_339, 2);  select_339 = None
        select_scatter_68: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_67, mul_68, 0, 68);  select_scatter_67 = mul_68 = None
        select_344: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 69)
        mul_69: "f32[]" = torch.ops.aten.mul.Tensor(select_344, 2);  select_344 = None
        select_scatter_69: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_68, mul_69, 0, 69);  select_scatter_68 = mul_69 = None
        select_349: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 70)
        mul_70: "f32[]" = torch.ops.aten.mul.Tensor(select_349, 2);  select_349 = None
        select_scatter_70: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_69, mul_70, 0, 70);  select_scatter_69 = mul_70 = None
        select_354: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 71)
        mul_71: "f32[]" = torch.ops.aten.mul.Tensor(select_354, 2);  select_354 = None
        select_scatter_71: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_70, mul_71, 0, 71);  select_scatter_70 = mul_71 = None
        select_359: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 72)
        mul_72: "f32[]" = torch.ops.aten.mul.Tensor(select_359, 2);  select_359 = None
        select_scatter_72: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_71, mul_72, 0, 72);  select_scatter_71 = mul_72 = None
        select_364: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 73)
        mul_73: "f32[]" = torch.ops.aten.mul.Tensor(select_364, 2);  select_364 = None
        select_scatter_73: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_72, mul_73, 0, 73);  select_scatter_72 = mul_73 = None
        select_369: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 74)
        mul_74: "f32[]" = torch.ops.aten.mul.Tensor(select_369, 2);  select_369 = None
        select_scatter_74: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_73, mul_74, 0, 74);  select_scatter_73 = mul_74 = None
        select_374: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 75)
        mul_75: "f32[]" = torch.ops.aten.mul.Tensor(select_374, 2);  select_374 = None
        select_scatter_75: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_74, mul_75, 0, 75);  select_scatter_74 = mul_75 = None
        select_379: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 76)
        mul_76: "f32[]" = torch.ops.aten.mul.Tensor(select_379, 2);  select_379 = None
        select_scatter_76: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_75, mul_76, 0, 76);  select_scatter_75 = mul_76 = None
        select_384: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 77)
        mul_77: "f32[]" = torch.ops.aten.mul.Tensor(select_384, 2);  select_384 = None
        select_scatter_77: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_76, mul_77, 0, 77);  select_scatter_76 = mul_77 = None
        select_389: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 78)
        mul_78: "f32[]" = torch.ops.aten.mul.Tensor(select_389, 2);  select_389 = None
        select_scatter_78: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_77, mul_78, 0, 78);  select_scatter_77 = mul_78 = None
        select_394: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 79)
        mul_79: "f32[]" = torch.ops.aten.mul.Tensor(select_394, 2);  select_394 = None
        select_scatter_79: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_78, mul_79, 0, 79);  select_scatter_78 = mul_79 = None
        select_399: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 80)
        mul_80: "f32[]" = torch.ops.aten.mul.Tensor(select_399, 2);  select_399 = None
        select_scatter_80: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_79, mul_80, 0, 80);  select_scatter_79 = mul_80 = None
        select_404: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 81)
        mul_81: "f32[]" = torch.ops.aten.mul.Tensor(select_404, 2);  select_404 = None
        select_scatter_81: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_80, mul_81, 0, 81);  select_scatter_80 = mul_81 = None
        select_409: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 82)
        mul_82: "f32[]" = torch.ops.aten.mul.Tensor(select_409, 2);  select_409 = None
        select_scatter_82: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_81, mul_82, 0, 82);  select_scatter_81 = mul_82 = None
        select_414: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 83)
        mul_83: "f32[]" = torch.ops.aten.mul.Tensor(select_414, 2);  select_414 = None
        select_scatter_83: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_82, mul_83, 0, 83);  select_scatter_82 = mul_83 = None
        select_419: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 84)
        mul_84: "f32[]" = torch.ops.aten.mul.Tensor(select_419, 2);  select_419 = None
        select_scatter_84: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_83, mul_84, 0, 84);  select_scatter_83 = mul_84 = None
        select_424: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 85)
        mul_85: "f32[]" = torch.ops.aten.mul.Tensor(select_424, 2);  select_424 = None
        select_scatter_85: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_84, mul_85, 0, 85);  select_scatter_84 = mul_85 = None
        select_429: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 86)
        mul_86: "f32[]" = torch.ops.aten.mul.Tensor(select_429, 2);  select_429 = None
        select_scatter_86: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_85, mul_86, 0, 86);  select_scatter_85 = mul_86 = None
        select_434: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 87)
        mul_87: "f32[]" = torch.ops.aten.mul.Tensor(select_434, 2);  select_434 = None
        select_scatter_87: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_86, mul_87, 0, 87);  select_scatter_86 = mul_87 = None
        select_439: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 88)
        mul_88: "f32[]" = torch.ops.aten.mul.Tensor(select_439, 2);  select_439 = None
        select_scatter_88: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_87, mul_88, 0, 88);  select_scatter_87 = mul_88 = None
        select_444: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 89)
        mul_89: "f32[]" = torch.ops.aten.mul.Tensor(select_444, 2);  select_444 = None
        select_scatter_89: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_88, mul_89, 0, 89);  select_scatter_88 = mul_89 = None
        select_449: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 90)
        mul_90: "f32[]" = torch.ops.aten.mul.Tensor(select_449, 2);  select_449 = None
        select_scatter_90: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_89, mul_90, 0, 90);  select_scatter_89 = mul_90 = None
        select_454: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 91)
        mul_91: "f32[]" = torch.ops.aten.mul.Tensor(select_454, 2);  select_454 = None
        select_scatter_91: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_90, mul_91, 0, 91);  select_scatter_90 = mul_91 = None
        select_459: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 92)
        mul_92: "f32[]" = torch.ops.aten.mul.Tensor(select_459, 2);  select_459 = None
        select_scatter_92: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_91, mul_92, 0, 92);  select_scatter_91 = mul_92 = None
        select_464: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 93)
        mul_93: "f32[]" = torch.ops.aten.mul.Tensor(select_464, 2);  select_464 = None
        select_scatter_93: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_92, mul_93, 0, 93);  select_scatter_92 = mul_93 = None
        select_469: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 94)
        mul_94: "f32[]" = torch.ops.aten.mul.Tensor(select_469, 2);  select_469 = None
        select_scatter_94: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_93, mul_94, 0, 94);  select_scatter_93 = mul_94 = None
        select_474: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 95)
        mul_95: "f32[]" = torch.ops.aten.mul.Tensor(select_474, 2);  select_474 = None
        select_scatter_95: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_94, mul_95, 0, 95);  select_scatter_94 = mul_95 = None
        select_479: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 96)
        mul_96: "f32[]" = torch.ops.aten.mul.Tensor(select_479, 2);  select_479 = None
        select_scatter_96: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_95, mul_96, 0, 96);  select_scatter_95 = mul_96 = None
        select_484: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 97)
        mul_97: "f32[]" = torch.ops.aten.mul.Tensor(select_484, 2);  select_484 = None
        select_scatter_97: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_96, mul_97, 0, 97);  select_scatter_96 = mul_97 = None
        select_489: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 98)
        mul_98: "f32[]" = torch.ops.aten.mul.Tensor(select_489, 2);  select_489 = None
        select_scatter_98: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_97, mul_98, 0, 98);  select_scatter_97 = mul_98 = None
        select_494: "f32[]" = torch.ops.aten.select.int(primals_1, 0, 99);  primals_1 = None
        mul_99: "f32[]" = torch.ops.aten.mul.Tensor(select_494, 2);  select_494 = None
        select_scatter_99: "f32[100]" = torch.ops.aten.select_scatter.default(select_scatter_98, mul_99, 0, 99);  select_scatter_98 = mul_99 = None
        return (select_scatter_99, full_default)
        