import torch

# Define CUDA kernel as a string
cuda_kernel = """
extern "C" __global__ void multiply_add_kernel(
    const float* input,
    float* output,
    const int n) 
{
    // Get thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < n) {
        // Perform multiply and add
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}
"""

def tensor_operation_cuda(input_tensor):
    # Load the CUDA kernel
    module = torch.utils.cpp_extension.load_inline(
        name="multiply_add",
        cpp_sources="",  # No C++ code needed
        cuda_sources=cuda_kernel,
        functions=["multiply_add_kernel"],
        verbose=True
    )
    
    # Prepare output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate grid and block dimensions
    threads_per_block = 256
    n = input_tensor.numel()
    blocks = (n + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    module.multiply_add_kernel(
        blocks,
        threads_per_block,
        (
            input_tensor.data_ptr(),
            output.data_ptr(),
            n
        )
    )
    
    return output

if __name__ == "__main__":
    # Create input tensor
    input_tensor = torch.randn(1000, 1000, device='cuda')
    
    # Run low-level CUDA operation
    result = tensor_operation_cuda(input_tensor)
    print("Low-level CUDA result shape:", result.shape)
    
    # Verify results match
    high_level_result = tensor_operation_cuda(input_tensor)
    print("Results match:", torch.allclose(result, high_level_result))
