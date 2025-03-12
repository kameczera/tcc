import torch
import time
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma GPU encontrada'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------- Função scriptada ----------------------- #
@torch.jit.script
def script_fn(x):
    if x.sum() > 0:
        return 2 * x + 1
    else:
        return x
    
# ---------------------------------------------------------------- #

# ------------------------ Função traçada ------------------------ #
def traced_fn(x):
    return 2 * x + 1

traced_fn = torch.jit.trace(traced_fn, (torch.rand(100, 100),))

# traced_fn = torch.jit.trace(lambda x: 2 * x + 1, (torch.rand(200, 200, 200),)) Tentativa de inline (não deu certo)

@torch.jit.script
def mixed_fn(x):
    if x.sum() > 0:
        return traced_fn(x)
    else:
        return x

# ---------------------------------------------------------------- #

def measure_time(fn, input_tensor, num_iterations=1000):
    fn(input_tensor)
    start_time = time.time()
    for _ in range(num_iterations):
        fn(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_iterations

input_tensor = torch.rand(100, 100)

num_iterations = 1000

script_time = measure_time(script_fn, input_tensor, num_iterations)
print(f"Tempo médio de execução da função scriptada: {script_time:.6f} segundos")

mixed_time = measure_time(mixed_fn, input_tensor, num_iterations)
print(f"Tempo médio de execução da função mixada: {mixed_time:.6f} segundos")
