import torch
import torch.nn.functional as F
import time
from torch.compile import compile

# Configuração do modelo
batch_size, seq_len, d_model, num_heads = 4, 512, 768, 12
head_dim = d_model // num_heads

# Simulação de Q, K, V
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# Máscara de atenção causal (triangular inferior)
def causal_mask(size):
    return torch.triu(torch.ones(size, size, device='cuda', dtype=torch.bool), diagonal=1)
mask = causal_mask(seq_len)

# Implementação ingênua
def naive_attention(Q, K, V, mask):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_scores.masked_fill_(mask, float('-inf'))
    attn_probs = F.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, V)

# Implementação otimizada (FlashAttention)
try:
    from flash_attn.flash_attn_func import flash_attn_func
    def optimized_attention(Q, K, V, mask):
        return flash_attn_func(Q, K, V, dropout_p=0.0, causal=True)
    flash_available = True
except ImportError:
    print("FlashAttention não encontrado, rodando apenas naive_attention.")
    flash_available = False

# Benchmark
for fn, label in [(naive_attention, "Naïve"), (optimized_attention, "Flash")] if flash_available else [(naive_attention, "Naïve")]:
    compiled_fn = compile(fn)  # TorchInductor
    
    # Execução e tempo médio
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        compiled_fn(Q, K, V, mask)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    print(f"{label} Attention (TorchInductor): {sum(times) / len(times):.6f} s")
