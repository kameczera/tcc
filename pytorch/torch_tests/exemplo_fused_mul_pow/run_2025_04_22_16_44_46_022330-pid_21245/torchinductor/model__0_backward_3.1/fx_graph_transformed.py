class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]", tangents_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:7 in f, code: return x ** 10
        pow_2: "f32[100]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 9.0);  primals_1 = None
        mul: "f32[100]" = torch.ops.aten.mul.Scalar(pow_2, 10.0);  pow_2 = None
        mul_1: "f32[100]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
        return (mul_1,)
        