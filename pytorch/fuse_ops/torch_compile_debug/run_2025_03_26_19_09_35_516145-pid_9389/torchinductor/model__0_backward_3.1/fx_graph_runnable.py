
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.graph_diagram = True




isolate_fails_code_str = None



# torch version: 2.5.1+cu121
# torch cuda version: 12.1
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Sep_12_02:18:05_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.77 
# Build cuda_12.6.r12.6/compiler.34841621_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3050 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, tangents_1):
        mul_1 = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        return (mul_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf0, (100,), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)