import torch
import time
# from torch._C import _jit_pass_dce
# from torch._C import _jit_pass_constant_propagation
# from torch._C import _jit_pass_peephole
# from torch._C import _jit_pass_inline
# from torch._C import _jit_pass_eliminate_dead_code
# from torch._C import _jit_pass_fuse_linear


@torch.jit.script
def complex_tensor_op(tensor):
    # Apply some fancy transformations to the tensor before passing to complex_tensor_op
    # 1. Normalize the tensor
    tensor = tensor / (torch.norm(tensor, dim=-1, keepdim=True) + 1e-8)
    
    # 2. Apply attention-like mechanism
    attention_weights = torch.softmax(tensor @ tensor.transpose(-2, -1), dim=-1)
    tensor = attention_weights @ tensor
    
    # 3. Add residual connection with layer normalization
    tensor_mean = tensor.mean(dim=-1, keepdim=True)
    tensor_var = ((tensor - tensor_mean) ** 2).mean(dim=-1, keepdim=True)
    tensor_normalized = (tensor - tensor_mean) / torch.sqrt(tensor_var + 1e-5)
    tensor = tensor + tensor_normalized * torch.randint(1, 11, (1,)).item()
    
    # 4. Apply non-linear activation
    tensor = torch.relu(tensor) * torch.sigmoid(tensor)

    return tensor

@torch.jit.script
def medium_tensor_op(tensor):
    # Complex tensor operation with multiple math operators
    return torch.sin(torch.matmul(tensor, tensor.transpose(-2, -1))) + torch.tanh(tensor.pow(0.9879865)) * 0.5

@torch.jit.script
def small_tensor_op(tensor):
    # Simpler operation for small tensors
    return tensor + torch.sigmoid(tensor)

@torch.jit.script
def higher_function(tensor):
    size =  tensor.shape[0]
    result = torch.zeros(1)
    if size >= 1000:
        result = complex_tensor_op(tensor)
    elif size >= 100:
        result = medium_tensor_op(tensor)
    elif size >= 10:
        result = small_tensor_op(tensor)

    return result

class program(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        print(f"Size of tensor inside program: {size}")

    def forward(self):
        tensor = torch.randn(self.size, self.size)
        size =  tensor.shape[0]
        result = torch.zeros(1)
        if size >= 1000:
            result = complex_tensor_op(tensor)
        elif size >= 100:
            result = medium_tensor_op(tensor)
        elif size >= 10:
            result = small_tensor_op(tensor)

        return result

def benchmark(model, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        model()
        end = time.time()
        times.append(end - start)
    return sum(times) / iterations

if __name__ == "__main__":
    # ================================
    # JIT script
    # ================================
    test_cases = [50, 500, 2000]

    for size in test_cases:
        instance = program(size)
        script = torch.jit.script(instance)

        print("==================")
        print(f"Dynamic graph for input size: {size}")
        print("==================")

        print("==================")
        print("Before optimization:")
        print("==================")

        print(script.graph)
        print("TorchScript code:")
        print(script.code)

        print("==================")
        print("\nAfter optimization:")
        print("==================")

        opt_script = torch.jit.optimize_for_inference(script)
        print(opt_script.forward.graph_for(size))

        print("\n2. TorchScript code:")
        print(opt_script.code)

        iterations = 10
        time_script = benchmark(script.forward, iterations)
        time_opt_script = benchmark(opt_script.forward, iterations)

        print(f"Tempo médio Script: {time_script:.6f} s")
        print(f"Tempo médio Opt_Script: {time_opt_script:.6f} s")
