import torch

def f(x):
    y = torch.relu(x)
    z = torch.sigmoid(y)
    return z

torch._dynamo.reset()
compiled_f = torch.compile(f, backend='inductor',
                              options={'trace.enabled':True,
                                       'trace.graph_diagram':True})

device = 'cuda'

torch.manual_seed(0)
x = torch.rand(100, requires_grad=True).to(device)
y = torch.ones_like(x)

out = torch.nn.functional.mse_loss(compiled_f(x),y).backward()