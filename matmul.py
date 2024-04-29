import torch
from zeus.monitor import ZeusMonitor

NUM_ITER = 10
N = 10000

def gpu_matrix_multiplication():
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    print(f"Using {device} device")

    # Size of the square matrices

    # Generate random matrices
    A = torch.rand(N, N, device=device)
    B = torch.rand(N, N, device=device)

    print("Starting matrix multiplication...")
    monitor = ZeusMonitor()
    monitor.begin_window("matmul")

    # Perform matrix multiplication
    with torch.no_grad():  # No need to track gradients for this operation
        for i in range(NUM_ITER):
            C = torch.matmul(A, B)
            A = torch.matmul(A, C)
            B = torch.matmul(B, C)

    print("Resulting matrix C has dimensions:", C.shape)

    result = monitor.end_window("matmul")
    print(f"matmul took {result.time} seconds.")
    print(f"matmul consumed {result.total_energy} Joules.")

if __name__ == "__main__":
    gpu_matrix_multiplication()
