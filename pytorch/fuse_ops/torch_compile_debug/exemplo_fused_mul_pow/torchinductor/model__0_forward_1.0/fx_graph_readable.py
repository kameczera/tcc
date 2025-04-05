class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test2.py:7 in f, code: return x ** 10
        pow_1: "f32[100]" = torch.ops.aten.pow.Tensor_Scalar(primals_1, 10)
        return (pow_1, primals_1)
        