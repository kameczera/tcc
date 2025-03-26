import torch
from torch.fx import symbolic_trace

def fn(x):
    return torch.sin(x) + torch.cos(x)

x = torch.randn(10, 10)

fx_fn = symbolic_trace(fn)
inductor_fn = torch.compile(fn , backend="inductor")
print(inductor_fn)
print(fx_fn.graph)
