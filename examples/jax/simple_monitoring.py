import time
import jax
import jax.numpy as jnp
from zeus.monitor import ZeusMonitor

@jax.jit
def mat_prod(B):
    """ Dummy example to make GPU warm with a jitting"""
    A = jnp.ones((1000, 1000))
    return A @ B

if __name__ == "__main__":
    # Time/Energy measurements for four GPUs will begin and end at the same time.
    gpu_indices = [0]

    monitor = ZeusMonitor(gpu_indices=gpu_indices, backend="jax")

    # Mark the beginning of a measurement window. You can use any string
    # as the window name, but make sure it's unique.
    monitor.begin_window("all_computations")

    # Actual work
    key = jax.random.PRNGKey(0)
    B = jax.random.uniform(key, (1000, 1000))
    for i in range(100):
        B = mat_prod(B)

    # Mark the end of a measurement window and retrieve the measurment result.
    result = monitor.end_window("all_computations")

    # Print the measurement result.
    print(f"Training took {result.time} seconds.")
    print(f"Training consumed {result.total_energy} Joules.")
    for gpu_idx, gpu_energy in result.gpu_energy.items():
        print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
