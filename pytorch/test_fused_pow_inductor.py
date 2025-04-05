import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

def f(x):
    return x ** 10

torch._dynamo.reset()
compiled_f = torch.compile(f, backend='inductor',
                              options={'trace.enabled':True,
                                       'trace.graph_diagram':True})

device = 'cuda'

torch.manual_seed(0)
x = torch.rand(100, requires_grad=True).to(device)
y = torch.ones_like(x)

out = torch.nn.functional.mse_loss(compiled_f(x),y).backward()