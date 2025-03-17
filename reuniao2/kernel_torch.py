import torch

def tensor_operation_high_level(input_tensor):
    # High-level PyTorch operation
    # This performs element-wise multiplication and addition
    output = input_tensor * 2 + 1
    return output

if __name__ == "__main__":
    # Create input tensor
    input_tensor = torch.randn(1000, 1000, device='cuda')
    
    # Run high-level operation
    result = tensor_operation_high_level(input_tensor)
    print("High-level result shape:", result.shape)
