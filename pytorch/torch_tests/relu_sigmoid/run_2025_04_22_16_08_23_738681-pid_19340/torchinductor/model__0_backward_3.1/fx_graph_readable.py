class GraphModule(torch.nn.Module):
    def forward(self, sigmoid: "f32[100]", le: "b8[100]", tangents_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:8 in f, code: z = torch.sigmoid(y)
        sub: "f32[100]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul: "f32[100]" = torch.ops.aten.mul.Tensor(sigmoid, sub);  sigmoid = sub = None
        mul_1: "f32[100]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:7 in f, code: y = torch.relu(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[100]" = torch.ops.aten.where.self(le, full_default, mul_1);  le = full_default = mul_1 = None
        return (where,)
        