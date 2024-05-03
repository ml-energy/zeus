### `ZeusMonitor`

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] makes it very simple to measure the GPU time and energy consumption of arbitrary Python code blocks.

```python hl_lines="4 11-13"
from zeus.monitor import ZeusMonitor

# All GPUs are measured simultaneously if `gpu_indices` is not given.
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

for epoch in range(100):
    monitor.begin_window("epoch")

    measurements = []
    for x, y in train_loader:
        monitor.begin_window("step")
        train_one_step(x, y)
        result = monitor.end_window("step")
        measurements.append(result)

    result = monitor.end_window("epoch")
    print(f"Epoch {epoch} consumed {result.time} s and {result.total_energy} J.")

    avg_time = sum(map(lambda m: m.time, measurements)) / len(measurements)
    avg_energy = sum(map(lambda m: m.total_energy, measurements)) / len(measurements)
    print(f"One step took {avg_time} s and {avg_energy} J on average.")
```

### CLI power and energy monitor

```console linenums="0"
$ python -m zeus.monitor power
[2023-08-22 22:39:59,787] [PowerMonitor](power.py:134) Monitoring power usage of GPUs [0, 1, 2, 3]
2023-08-22 22:40:00.800576
{'GPU0': 66.176, 'GPU1': 68.792, 'GPU2': 66.898, 'GPU3': 67.53}
2023-08-22 22:40:01.842590
{'GPU0': 66.078, 'GPU1': 68.595, 'GPU2': 66.996, 'GPU3': 67.138}
2023-08-22 22:40:02.845734
{'GPU0': 66.078, 'GPU1': 68.693, 'GPU2': 66.898, 'GPU3': 67.236}
2023-08-22 22:40:03.848818
{'GPU0': 66.177, 'GPU1': 68.675, 'GPU2': 67.094, 'GPU3': 66.926}
^C
Total time (s): 4.421529293060303
Total energy (J):
{'GPU0': 198.52566362297537, 'GPU1': 206.22215216255188, 'GPU2': 201.08565518283845, 'GPU3': 201.79834523367884}
```

```console linenums="0"
$ python -m zeus.monitor energy
[2023-08-22 22:44:45,106] [ZeusMonitor](energy.py:157) Monitoring GPU [0, 1, 2, 3].
[2023-08-22 22:44:46,210] [zeus.utils.framework](framework.py:38) PyTorch with CUDA support is available.
[2023-08-22 22:44:46,760] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' started.
^C[2023-08-22 22:44:50,205] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' ended.
Total energy (J):
Measurement(time=3.4480526447296143, energy={0: 224.2969999909401, 1: 232.83799999952316, 2: 233.3100000023842, 3: 234.53700000047684})
```
