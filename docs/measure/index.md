# Measuring Energy

!!! Important
    This page assumes that your environment is already set up. Please refer to the [Getting Started](../getting_started/index.md) guide if not.

Zeus makes it very easy to measure time, power, and energy both programmatically in Python and also on the command line.
Measuring power and energy is also very low overhead, typically taking less than 10 ms for each call.

## Programmatic measurement

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] makes it very simple to measure the GPU time and energy consumption of arbitrary Python code blocks.

A *measurement window* is defined by a code block wrapped with [`begin_window`][zeus.monitor.ZeusMonitor.begin_window] and [`end_window`][zeus.monitor.ZeusMonitor.end_window].
[`end_window`][zeus.monitor.ZeusMonitor.end_window] will return a [`Measurement`][zeus.monitor.energy.Measurement] object, which holds the time and energy consumption of the window.
Users can specify and measure multiple measurement windows at the same time, and they can be arbitrarily nested or overlapping as long as they are given different names.

```python hl_lines="5 12-14"
from zeus.monitor import ZeusMonitor

if __name__ == "__main__":
    # All GPUs are measured simultaneously if `gpu_indices` is not given.
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

    for epoch in range(100):
        monitor.begin_window("epoch")

        steps = []
        for x, y in train_loader:
            monitor.begin_window("step")
            train_one_step(x, y)
            result = monitor.end_window("step")
            steps.append(result)

        mes = monitor.end_window("epoch")
        print(f"Epoch {epoch} consumed {mes.time} s and {mes.total_energy} J.")

        avg_time = sum(map(lambda m: m.time, steps)) / len(steps)
        avg_energy = sum(map(lambda m: m.total_energy, steps)) / len(steps)
        print(f"One step took {avg_time} s and {avg_energy} J on average.")
```

!!! Tip "[`zeus.monitor.PowerMonitor`][zeus.monitor.power.PowerMonitor]"
    This monitor spawns a process that polls the instantaneous GPU power consumption API and exposes two methods: [`get_power`][zeus.monitor.power.PowerMonitor.get_power] and [`get_energy`][zeus.monitor.power.PowerMonitor.get_energy].
    For GPUs older than Volta that do not support querying energy directly, [`ZeusMonitor`][zeus.monitor.ZeusMonitor] automatically uses the [`PowerMonitor`][zeus.monitor.power.PowerMonitor] internally.

!!! Warning "Use of global variables on GPUs older than Volta"
    On GPUs older than Volta, **you should not** instantiate [`ZeusMonitor`][zeus.monitor.ZeusMonitor] as a global variable without protecting it with `if __name__ == "__main__"`.
    It's because the energy query API is only available on Volta or newer NVIDIA GPU microarchitectures, and for older GPUs, a separate process that polls the power API has to be spawned (i.e., [`PowerMonitor`][zeus.monitor.power.PowerMonitor]).
    In this case, global code that spawns the process should be guarded with `if __name__ == "__main__"`.
    More details in [Python docs](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods){.external}.

!!! Tip "`gpu_indices` and `CUDA_VISIBLE_DEVICES`"
    Zeus always respects `CUDA_VISIBLE_DEVICES` if set.
    In other words, if `CUDA_VISIBLE_DEVICES=1,3` and `gpu_indices=[1]`, Zeus will understand that as GPU 3 in the system.

!!! Important "`gpu_indices` and optimization"
    In general, energy optimizers measure the energy of the GPU through a [`ZeusMonitor`][zeus.monitor.ZeusMonitor] instance that is passed to their constructor.
    Thus, only the GPUs specified by `gpu_indices` will be the target of optimization.

### Synchronizing CPU and GPU computations

Deep learning frameworks typically run actual computation on GPUs in an asynchronous fashion.
That is, the CPU (Python interpreter) asynchronously dispatches computations to run on the GPU and moves on to dispatch the next computation without waiting for the GPU to finish.
This helps GPUs achieve higher utilization with less idle time.

Due to this asynchronous nature of Deep Learning frameworks, we need to be careful when we want to take time and energy measurements of GPU execution.
We want *only and all of* the computations dispatched between `begin_window` and `end_window` to be captured by our time and energy measurement.
That's what the `sync_execution_with` paramter in [`ZeusMonitor`][zeus.monitor.ZeusMonitor] and `sync_execution` paramter in [`begin_window`][zeus.monitor.ZeusMonitor.begin_window] and [`end_window`][zeus.monitor.ZeusMonitor.end_window] are for.
Depending on the Deep Learning framework you're using (currently PyTorch and JAX are supported), [`ZeusMonitor`][zeus.monitor.ZeusMonitor] will automatically synchronize CPU and GPU execution to make sure all and only the computations dispatched between the window are captured.

!!! Tip
    Zeus has one function used globally across the codebase for device synchronization: [`sync_execution`][zeus.utils.framework.sync_execution].

!!! Warning
    [`ZeusMonitor`][zeus.monitor.ZeusMonitor] covers only the common and simple case of device synchronization, when GPU indices (`gpu_indices`) correspond to one whole physical device.
    This is usually what you want, except when using more advanced device partitioning (e.g., using `--xla_force_host_platform_device_count` in JAX to partition CPUs into more pieces).
    In such cases, you probably want to opt out from using this function and handle synchronization manually at the appropriate granularity.

## CPU measurements using Intel RAPL

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] supports CPU/DRAM energy measurement as well!

The RAPL interface for CPU energy measurement is available for the majority of Intel and AMD CPUs.
DRAM energy measurement are available on some CPUs as well.
To check support, refer to [Verifying installation](../getting_started/index.md#verifying-installation).

To only measure the energy consumption of the CPU used by the current Python process, you can use the [`get_current_cpu_index`][zeus.device.cpu.get_current_cpu_index] function, which retrieves the CPU index where the specified process ID is running.

You can pass in `cpu_indices=[]` or `gpu_indices=[]` to [`ZeusMonitor`][zeus.monitor.ZeusMonitor] to disable either CPU or GPU measurements.

```python hl_lines="2 5-7"
from zeus.monitor import ZeusMonitor
from zues.device.cpu import get_current_cpu_index

if __name__ == "__main__":
    # Get the CPU index of the current process
    current_cpu_index = get_current_cpu_index()
    monitor = ZeusMonitor(cpu_indices=[current_cpu_index], gpu_indices=[])

    for epoch in range(100):
        monitor.begin_window("epoch")

        steps = []
        for x, y in train_loader:
            monitor.begin_window("step")
            train_one_step(x, y)
            result = monitor.end_window("step")
            steps.append(result)

        mes = monitor.end_window("epoch")
        print(f"Epoch {epoch} consumed {mes.time} s and {mes.total_energy} J.")

        avg_time = sum(map(lambda m: m.time, steps)) / len(steps)
        avg_energy = sum(map(lambda m: m.total_energy, steps)) / len(steps)
        print(f"One step takes {avg_time} s and {avg_energy} J for the CPU.")
```

## CLI power and energy monitor

The energy monitor measures the total energy consumed by the GPU during the lifetime of the monitor process.
It's a simple wrapper around [`ZeusMonitor`][zeus.monitor.ZeusMonitor].

```console
$ python -m zeus.monitor energy
[2023-08-22 22:44:45,106] [ZeusMonitor](energy.py:157) Monitoring GPU [0, 1, 2, 3].
[2023-08-22 22:44:46,210] [zeus.utils.framework](framework.py:38) PyTorch with CUDA support is available.
[2023-08-22 22:44:46,760] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' started.
^C[2023-08-22 22:44:50,205] [ZeusMonitor](energy.py:329) Measurement window 'zeus.monitor.energy' ended.
Total energy (J):
Measurement(time=3.4480526447296143, energy={0: 224.2969999909401, 1: 232.83799999952316, 2: 233.3100000023842, 3: 234.53700000047684})
```

The power monitor periodically prints out the GPU's power draw.
It's a simple wrapper around [`PowerMonitor`][zeus.monitor.PowerMonitor].

```console
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
