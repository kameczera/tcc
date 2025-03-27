import torch

@torch.compile
def explicit_loop(x):
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] * 2  # Operação iterativa
    return out

x = torch.randn(100)
output = explicit_loop(x)