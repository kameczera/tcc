import torch

def explicit_loop(x):
    out = torch.zeros_like(x)
    out = x * 2
    return out

torch._dynamo.reset()
compiled_f = torch.compile(explicit_loop, backend='inductor')

device = 'cuda'
torch.manual_seed(0)
x = torch.rand(10, requires_grad=True).to(device)
y = torch.ones_like(x)

# Criando eventos CUDA para marcação do tempo
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Sincroniza antes de medir o tempo
torch.cuda.synchronize()
start_event.record()

# Execução da função compilada
out = torch.nn.functional.mse_loss(compiled_f(x), y).backward()

# Marca o fim da execução
end_event.record()
torch.cuda.synchronize()  # Aguarda todas as operações terminarem

# Calcula o tempo total
elapsed_time = start_event.elapsed_time(end_event)  # Tempo em milissegundos

print(f"Tempo de execução: {elapsed_time:.3f} ms")
