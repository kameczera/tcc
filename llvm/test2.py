import torch
import time

# ----------------------- Função scriptada ----------------------- #
@torch.jit.script
def scripted_fn(x, y):
    result = x @ y + torch.sin(x)
    if result.mean() > 0:
        return result / result.std()
    else:
        return result

# ---------------------------------------------------------------- #

# ------------------------ Função traçada ------------------------ #
def traced_fn(x, y):
    return x @ y + torch.sin(x)

example_x = torch.rand(100, 100)
example_y = torch.rand(100, 100)

traced_fn = torch.jit.trace(traced_fn, (example_x, example_y))

@torch.jit.script
def mixed_fn(x, y):
    if x.sum() > y.sum():
        return traced_fn(x, y)
    else:
        return x + y

# ---------------------------------------------------------------- #

def measure_time(fn, it_1, it_2, num_iterations=1000):
    fn(it_1, it_2)
    start_time = time.time()
    for _ in range(num_iterations):
        fn(it_1, it_2)
    end_time = time.time()
    return (end_time - start_time) / num_iterations

num_iterations = 1000

script_time = measure_time(scripted_fn, example_x, example_y, num_iterations)
print(f"Tempo médio de execução da função scriptada: {script_time:.6f} segundos")

mixed_time = measure_time(mixed_fn, example_x, example_y, num_iterations)
print(f"Tempo médio de execução da função mixada: {mixed_time:.6f} segundos")