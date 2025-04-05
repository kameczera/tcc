import torch
import time
import matplotlib.pyplot as plt

def explicit_loop_iterative(x):
    out = torch.zeros_like(x)
    out = x * x * x * x
    return out

def explicit_loop_vectorized(x):
    return x * 2

# Compilação
torch._dynamo.reset()
compiled_iterative = torch.compile(explicit_loop_iterative, backend='inductor')
compiled_vectorized = torch.compile(explicit_loop_vectorized, backend='inductor')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sizes = [100, 200, 300, 400, 500, 600]
times_iterative = []
times_vectorized = []

def measure_time(func, x):
    func(x)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func(x)  # Chamada real da função
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

for size in sizes:
    x = torch.rand(size, requires_grad=False, device=device)

    time_iter = measure_time(compiled_iterative, x)
    time_vect = measure_time(compiled_vectorized, x)

    times_iterative.append(time_iter)
    times_vectorized.append(time_vect)

plt.figure(figsize=(8, 5))
plt.plot(sizes, times_iterative, label="Iterativo (Loop)", marker='o', linestyle='--', color='r')
plt.plot(sizes, times_vectorized, label="Vetorizado (Multiplicação Direta)", marker='s', linestyle='-', color='g')

plt.xlabel("Tamanho do Tensor")
plt.ylabel("Tempo de Execução (ms)")
plt.title("Comparação de Tempo de Execução - Iterativo vs Vetorizado")
plt.legend()
plt.grid()

plt.savefig("comparacao_tempo_execucao.png", dpi=300, bbox_inches='tight')
