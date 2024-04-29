from zeus.monitor import ZeusMonitor
from zeus.device import get_gpus
import pdb
import time
import torch

# pdb.set_trace()

gpus = get_gpus()

if torch.cuda.is_available():
    print("CUDA is available. Listing available GPUs:")
    
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} (Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB)")

monitor = ZeusMonitor()

monitor.begin_window("test")

time.sleep(2)

result = monitor.end_window("test")

print(f"Training took {result.time} seconds.")
print(f"Training consumed {result.total_energy} Joules.")

for gpu_idx, gpu_energy in result.energy.items():
    print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")