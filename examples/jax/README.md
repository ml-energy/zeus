# Measuring the energy consumption of JAX

`ZeusMonitor` officially supports JAX:

```python
monitor = ZeusMonitor(sync_execution_with="jax")

monitor.begin_window("computations")
# Run computation
measurement = monitor.end_window("computations")
```

The `sync_execution_with` parameter in `ZeusMonitor` tells the monitor that it should use JAX mechanisms to wait for GPU computations to complete.
GPU computations typically run asynchronously with your Python code (in both PyTorch and JAX), so waiting for GPU computations to complete is important to ensure that we measure the right set of computations.

## Running the example

Install dependencies:

```sh
pip install -r requirements.txt
```

Run the example:

```sh
python measure_energy.py
```
