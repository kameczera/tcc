import torch
import time

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

def passagem_personalizada(grafo):
    """
    Passagem de transformação personalizada do grafo que incorpora `inline_function`
    dentro de `my_function`.
    """
    print("Antes da Passagem:")
    print(grafo)

    # Obtém o grafo da função inline
    grafo_inline = inline_function.graph

    # Itera sobre os nós no grafo
    for no in grafo.nodes():
        if no.kind() == "aten::sin":  # Exemplo: Substitui sin() com inline_function
            print(f"Encontrado {no.kind()}, substituindo por inline_function")
            
            # Clona nós de inline_function para o grafo principal
            for no_inl in grafo_inline.nodes():
                novo_no = grafo.createClone(no_inl, lambda x: x)
                grafo.appendNode(novo_no)

            # Remove o nó antigo
            no.destroy()

    print("\nDepois da Passagem:")
    print(grafo)

# Aplica a passagem
passagem_personalizada(script_fn.graph)


