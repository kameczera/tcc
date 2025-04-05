import torch
import time
import logging
import os

# 1. Forçar uso de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Definir a função alvo
def tensor_power_10(x):
    return x ** 10

# 3. Resetar e ativar logs detalhados
torch._dynamo.reset()
torch._logging.set_logs(
    dynamo=logging.DEBUG,
    inductor=logging.DEBUG,
    aot=logging.DEBUG
)

# 4. Compilar com TorchInductor
compiled_fn = torch.compile(tensor_power_10, backend="inductor", mode="default")

# 5. Criar um tensor grande o bastante (para evitar fallback)
x = torch.randn(1024 * 1024 * 4, device=device)  # ~4 milhões de floats

# 6. Executar e medir tempo
start = time.time()
result = compiled_fn(x)
torch.cuda.synchronize()
end = time.time()

print(f"Tempo de execução: {end - start:.6f} segundos")
print(f"Resultado (primeiros 5 elementos): {result[:5]}")

# 7. Salvar debug info
debug_dir = "torch_compile_debug"
torch._dynamo.debug_utils.save_debug_info(debug_dir)

# 8. Verificar se algo foi salvo
print(f"\nArquivos salvos em '{debug_dir}':")
for root, dirs, files in os.walk(debug_dir):
    for f in files:
        print(os.path.join(root, f))
    for d in dirs:
        print(f"[DIR] {os.path.join(root, d)}")
