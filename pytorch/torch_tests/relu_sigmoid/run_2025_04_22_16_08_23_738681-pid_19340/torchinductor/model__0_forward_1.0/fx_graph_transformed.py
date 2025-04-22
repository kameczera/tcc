class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:7 in f, code: y = torch.relu(x)
        relu: "f32[100]" = torch.ops.aten.relu.default(primals_1);  primals_1 = None
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:8 in f, code: z = torch.sigmoid(y)
        sigmoid: "f32[100]" = torch.ops.aten.sigmoid.default(relu)
        
         # File: /home/kamei/Documentos/projects/tcc/pytorch/test_fused_pow_inductor.py:7 in f, code: y = torch.relu(x)
        le: "b8[100]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        return (sigmoid, sigmoid, le)
        