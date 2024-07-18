import jax
import jax.numpy as jnp

from zeus.monitor import ZeusMonitor

@jax.jit
def mat_prod(B):
    A = jnp.ones((1000, 1000))
    return A @ B

def main():
    # Monitor the GPU with index 0.
    # The monitor will use a JAX-specific method to wait for the GPU
    # to finish computations when `end_window` is called.
    monitor = ZeusMonitor(gpu_indices=[0], sync_execution_with="jax")

    # Mark the beginning of a measurement window.
    monitor.begin_window("all_computations")

    # Actual work
    key = jax.random.PRNGKey(0)
    B = jax.random.uniform(key, (1000, 1000))
    for i in range(50000):
        B = mat_prod(B)

    # Mark the end of a measurement window and retrieve the measurment result.
    measurement = monitor.end_window("all_computations")

    # Print the measurement result.
    print("Measurement object:", measurement)
    print(f"Took {measurement.time} seconds.")
    for gpu_idx, gpu_energy in measurement.gpu_energy.items():
        print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")


if __name__ == "__main__":
    main()
