import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

def f(x):
    return x ** 10

def inspect_backend(gm, sample_inputs): 
    # Forward compiler capture
    def fw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("forward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Backward compiler capture
    def bw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("backward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Call AOTAutograd
    gm_forward = aot_module_simplified(gm,sample_inputs,
                                       fw_compiler=fw,
                                       bw_compiler=bw)

    return gm_forward

torch.manual_seed(0)
x = torch.rand(1000, requires_grad=True).to("cuda")
y = torch.ones_like(x)

torch._dynamo.reset()
compiled_f = torch.compile(f, backend=inspect_backend)
out = torch.nn.functional.mse_loss(compiled_f(x), y).backward()