import torch
from torch._C import _jit_pass_dce
from typing import List

def add(op1, op2):
    return op1 + op2

def mul(op1, op2):
    return op1 * op2

def tensor_constant_op(graph: torch._C.Graph, nodes, current, next_node, op):
    if(op == add):
        current_input1, current_input2, current_alpha = current.inputs()
        next_input1, next_input2, next_alpha = next_node.inputs()
    else:
        current_input1, current_input2 = current.inputs()
        next_input1, next_input2 = next_node.inputs()

    if current_input2.node().kind() == "prim::Constant" and next_input2.node().kind() == "prim::Constant":
        if next_input1 == current.output():
            value1 = current_input2.toIValue()
            value2 = next_input2.toIValue()
            new_int = graph.insertConstant(op(value1, value2))
            new_int.node().moveBefore(nodes[0])
            if(op == add): new_op = graph.create("aten::add", [current_input1, new_int, current_alpha])
            else: new_op = graph.create("aten::mul", [current_input1, new_int])
            new_op.insertBefore(current)
            next_node.output().replaceAllUsesWith(new_op.output())

def fuse_consecutive_ops(graph: torch._C.Graph):
    """
    Otimização para fundir operações consecutivas de soma (aten::add).
    """
    nodes = list(graph.nodes())
    changes = True
    while changes:
        changes = False
        for i in range(len(nodes) - 1, 1, -1):
            current = nodes[i - 1]
            next_node = nodes[i]
            if current.kind() == "aten::add" and next_node.kind() == "aten::add":
                tensor_constant_op(graph, nodes, current, next_node, add)
                del nodes[i]
                del nodes[i - 1]
                next_node.destroy()
                current.destroy()
                changes = True
            elif current.kind() == "aten::mul" and next_node.kind() == "aten::mul":
                tensor_constant_op(graph, nodes, current, next_node, mul)
                del nodes[i]
                del nodes[i - 1]
                next_node.destroy()
                current.destroy()
                changes = True
        nodes = list(graph.nodes())
        print(graph)
    return graph

def destroy_useless_variables(graph: torch._C.Graph):
    nodes = list(graph.nodes())
    for current in nodes[:]:
        if not current.hasUses():
            current.destroy()
    return graph

def optimize_script(script_module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
    graph = script_module.graph
    fuse_consecutive_ops(graph)
    destroy_useless_variables(graph)

    return script_module

class ExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.ones(1)
    
    def forward(self, x: torch.Tensor):
        b = 2 + x
        d = 4 + b
        e = d * 5
        f = e * 5
        c = torch.relu(f)
        return c

instance = ExampleModule()
script = torch.jit.script(instance)
print("Classe original:")
print(script.graph)
optimized_module = optimize_script(script)

print("Classe otimizada:")
print(optimized_module.graph)