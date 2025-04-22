import torch
import triton
import triton.language as tl

# Dados de entrada
x = torch.randn(100, device='cuda', dtype=torch.float32)
buf0 = torch.empty_like(x)
buf1 = torch.empty(100, device='cuda', dtype=torch.bool)

# ========= KERNELS ========= #

@triton.jit
def kernel_pre_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0)
    tmp3 = tl.sigmoid(tmp2)
    tl.store(out_ptr0 + x0, tmp3, xmask)

@triton.jit
def kernel_pre_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0)
    tmp3 = 0.0
    tmp4 = tmp2 <= tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)

@triton.jit
def kernel_post_fusion(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 0.0
    tmp5 = tmp2 <= tmp4
    tl.store(out_ptr0 + x0, tmp3, xmask)
    tl.store(out_ptr1 + x0, tmp5, xmask)

# ========= Benchmark ========= #

def benchmark_kernel(fn, *args, grid):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn[grid]( *args, x.numel(), 64 )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

# Executa os dois kernels pré-fusão
t_pre_0 = benchmark_kernel(kernel_pre_0, x, buf0, grid=(2,))
t_pre_1 = benchmark_kernel(kernel_pre_1, x, buf1, grid=(2,))
t_pre_total = t_pre_0 + t_pre_1

# Executa o kernel pós-fusão
buf0_post = torch.empty_like(x)
buf1_post = torch.empty(100, device='cuda', dtype=torch.bool)
t_post = benchmark_kernel(kernel_post_fusion, x, buf0_post, buf1_post, grid=(2,))

# ========= Resultados ========= #
print(f"Tempo Pré-Fusão total: {t_pre_total:.3f} ms")
print(f"Tempo Pós-Fusão: {t_post:.3f} ms")

# Valida se o resultado é o mesmo
assert torch.allclose(buf0, buf0_post, atol=1e-4)
assert torch.equal(buf1, buf1_post)
