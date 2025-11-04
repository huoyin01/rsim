import torch

def check_gpu_info():
    """
    Checks and prints information about available GPUs using PyTorch.
    """
    print("Checking GPU information using PyTorch...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s).")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - CUDA Capability: {torch.cuda.get_device_capability(i)}")
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    - Total Memory: {total_mem:.2f} GB")
    else:
        print("No GPU found. PyTorch is running on CPU.")

if __name__ == "__main__":
    check_gpu_info()