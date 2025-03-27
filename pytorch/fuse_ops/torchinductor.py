import torch

def explicit_loop(x):
    out = torch.zeros_like(x)
    # for i in range(x.shape[0]):
    #    out[i] = x[i] * 2
    out = x * 2
    return out

torch._dynamo.reset()
compiled_f = torch.compile(explicit_loop, backend='inductor',
                              options={'trace.enabled':True,
                                       'trace.graph_diagram':True})

device = 'cuda'

torch.manual_seed(0)
x = torch.rand(100, requires_grad=True).to(device)
y = torch.ones_like(x)

out = torch.nn.functional.mse_loss(compiled_f(x),y).backward()
