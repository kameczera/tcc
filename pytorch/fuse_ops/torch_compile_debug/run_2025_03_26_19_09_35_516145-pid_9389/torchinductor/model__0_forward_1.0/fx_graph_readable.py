class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[100]"):
         # File: /home/kamei/Documentos/projects/tcc/pytorch/fuse_ops/torchinductor.py:7 in explicit_loop, code: out = x * 2
        mul: "f32[100]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        return (mul,)
        