import torch

def f(tensor):
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

torch._dynamo.reset()

device = 'cuda'
torch.manual_seed(0)
x = torch.rand(100, requires_grad=True).to(device)
y = torch.ones_like(x)

# Explicação da compilação
explanation = torch._dynamo.explain(f)(x)

print("=== TorchDynamo Explanation ===")
print("Total Graphs Compiled:", explanation.graph_count)

if hasattr(explanation, "break_reasons"):
    print("\nGraph Break Reasons:")
    for i, reason in enumerate(explanation.break_reasons, 1):
        print(f"{i}. {reason}")
else:
    print("Nenhuma razão de quebra encontrada.")

