import torch
import gc

def select_smart_device():
    # Define 1GB in bytes
    ONE_GB = 2**30
    
    # Check if any CUDA is available at all
    if not torch.cuda.is_available():
        print("No CUDA found. Defaulting to CPU.")
        return torch.device("cpu")

    try:
        # Get (free_memory, total_memory) for cuda:0
        free_0, total_0 = torch.cuda.mem_get_info(0)
        
        print(f"CUDA:0 Free Memory: {free_0 / ONE_GB:.2f} GB")

        # Your Logic: If cuda:0 is low (<1GB), try cuda:1
        if free_0 < ONE_GB:
            print("CUDA:0 is low on memory. Checking CUDA:1...")
            
            if torch.cuda.device_count() > 1:
                free_1, _ = torch.cuda.mem_get_info(1)
                if free_1 > ONE_GB:
                    print("Using CUDA:1.")
                    return torch.device("cuda:1")
                else:
                    print("CUDA:1 is also full.")
            else:
                print("CUDA:1 is not available.")
            
            # If we get here, both GPUs are full or 1 doesn't exist
            print("Falling back to CPU.")
            return torch.device("cpu")
            
        else:
            # If CUDA:0 has enough space (>=1GB)
            print("CUDA:0 has sufficient space. Using CUDA:0.")
            return torch.device("cuda:0")

    except Exception as e:
        print(f"Error checking memory: {e}. Defaulting to CPU.")
        return torch.device("cpu")


def clean_all_gpu_memory():
    # 1. Clear out Python references
    # Note: If you have specific large variables (like 'model' or 'patterns'), 
    # you should 'del' them before calling this function.
    gc.collect()

    # 2. Check how many GPUs are visible
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        # 3. Set the current device to the specific GPU
        torch.cuda.set_device(i)
        
        # 4. Wait for all kernels to finish so we don't clear memory in use
        torch.cuda.synchronize()
        
        # 5. Release the cached (reserved) memory back to the OS
        torch.cuda.empty_cache()
        
        print(f"✅ Cleared VRAM for GPU {i}: {torch.cuda.get_device_name(i)}")