import torch
from torch._C import _jit_pass_dce
from typing import List

def fuse_consecutive_ops(graph: torch._C.Graph):
    """
    Otimização para fundir operações consecutivas de soma (aten::add).
    """
    nodes = list(graph.nodes())

    for i in range(len(nodes) - 1, 0, -1):
        current = nodes[i - 1]
        next_node = nodes[i]

        if current.kind() == "aten::add" and next_node.kind() == "aten::add":
            current_input1, current_input2, current_alpha = current.inputs()
            next_input1, next_input2, next_alpha = next_node.inputs()
            print(next_input1 == current.output())
            print(next_alpha)
            if next_input1 == current.output():
                print("ois")
                new_add = graph.create("aten::add", [current_input1, current_input2, current_alpha])
                graph.appendNode(new_add)
                next_node.output().replaceAllUsesWith(new_add.output())

                next_node.destroy()
                current.destroy()

    return graph

def optimize_script(script_module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
    graph = script_module.graph
    fuse_consecutive_ops(graph)

    return script_module

@torch.jit.script
def example_function(x: torch.Tensor):
    a = x + 2 + 4
    b = a + 3
    c = torch.relu(b)
    return c
fn = torch.jit.script(example_function)
print(fn.graph)
optimized_fn = optimize_script(torch.jit.script(example_function))
print("Funcao otimizada:")
print(optimized_fn.graph)




# class ExampleModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.value = torch.ones(1)
    
#     def forward(self, x: torch.Tensor):
#         a = x + 2 + 4
#         b = a + 3
#         c = torch.relu(b)
#         return c

# instance = ExampleModule()
# script = torch.jit.script(instance)
# optimized_module = optimize_script(script)

# print("Classe otimizada:")
# print(optimized_module.graph)