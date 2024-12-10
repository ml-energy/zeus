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

```python hl_lines="2 5-15"
from zeus.monitor import ZeusMonitor
from zeus.device.cpu import get_current_cpu_index

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
## Metric Monitoring

Zeus allows for efficient monitoring of energy and power consumption for GPUs, CPUs, and DRAM using Prometheus. It tracks key metrics such as energy usage, power draw, and cumulative consumption. Users can define measurement windows to track energy usage for specific operations, enabling granular analysis and optimization.

!!! Assumption
    A [Prometheus Push Gateway](https://prometheus.io/docs/instrumenting/pushing/) must be deployed and accessible. This ensures that metrics collected by Zeus can be pushed to Prometheus.

### Local Setup Guide

#### Step 1: Install and Start the Prometheus Push Gateway
Choose either Option 1 (Binary) or Option 2 (Docker).

##### Option 1: Download Binary
1. Visit the [Prometheus Push Gateway Download Page](https://prometheus.io/download/#pushgateway).
2. Download the appropriate binary for your operating system.
3. Extract the binary:
```sh
   tar -xvzf prometheus-pushgateway*.tar.gz
   cd prometheus-pushgateway-*
```
4. Start the Push Gateway:
```sh
./prometheus-pushgateway --web.listen-address=:9091
```
5. Verify the Push Gateway is running by visiting http://localhost:9091 in your browser.

##### Option 2: Using Docker
1. Pull the official Prometheus Push Gateway Docker image:
```sh
docker pull prom/pushgateway
```
2. Run the Push Gateway in a container:
```sh
docker run -d -p 9091:9091 prom/pushgateway
```
3. Verify it is running by visiting http://localhost:9091 in your browser.

#### Step 2: Install and Configure Prometheus
1. Visit the Prometheus [Prometheus Download Page](https://prometheus.io/download/#prometheus).
2. Download the appropriate binary for your operating system.
3. Extract the binary:
```sh
tar -xvzf prometheus*.tar.gz
cd prometheus-*
```
4. Update the Prometheus configuration file `prometheus.yml` to scrape metrics from the Push Gateway:
```sh
scrape_configs:
  - job_name: 'pushgateway'
    honor_labels: true
    static_configs:
      - targets: ['localhost:9091']  # Replace with your Push Gateway URL
```
5. Start Prometheus:
```sh
./prometheus --config.file=prometheus.yml
```
6. Visit http://localhost:9090 in your browser, or use curl http://localhost:9090/api/v1/targets
7. Verify Prometheus is running by visiting http://localhost:9090 in your browser.

### Metric Name Construction

Zeus organizes metrics using **static metric names** and **dynamic labels** for flexibility and ease of querying in Prometheus. Metric names are static and cannot be overridden, but users can customize the context of the metrics by naming the window when using `begin_window()` and `end_window()`.

#### Metric Name
- For Histogram: `energy_monitor_{component}_energy_joules`
- For Counter: `energy_monitor_{component}_energy_joules`
- For Gauge: `power_monitor_{component}_power_watts`

Note that Gauge only supports the GPU component at the moment. Tracking issue: [#128](https://github.com/ml-energy/zeus/issues/128)


component: gpu, cpu, or dram

#### Labels
- window: The user-defined window name provided during `begin_window()` and `end_window()` (e.g., `energy_histogram.begin_window(f"epoch_energy")`).
- index: The GPU index (e.g., `0` for GPU 0).

### Usage and Initialization
[`EnergyHistogram`][zeus.metric.EnergyHistogram] records energy consumption data for GPUs, CPUs, and DRAM in Prometheus Histograms. This is ideal for observing how often energy usage falls within specific ranges.

```python hl_lines="2 5-15"
from zeus.metric import EnergyHistogram

if __name__ == "__main__":
    # Initialize EnergyHistogram
    energy_histogram = EnergyHistogram(
        cpu_indices=[0], 
        gpu_indices=[0], 
        prometheus_url='http://localhost:9091', 
        job='training_energy_histogram'
    )

    for epoch in range(100):
        # Start monitoring energy for the entire epoch
        energy_histogram.begin_window("epoch_energy")
        # Perform epoch-level operations 
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        # End monitoring energy for the epoch
        energy_histogram.end_window("epoch_energy")
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1}%")

```
You can use the `begin_window` and `end_window` methods to define a measurement window, similar to other ZeusMonitor operations. Energy consumption data will be recorded for the entire duration of the window.

!!! Tip 
    You can customize the bucket ranges for GPUs, CPUs, and DRAM during initialization to tailor the granularity of energy monitoring. For example:
    ```python hl_lines="2 5-15"
    energy_histogram = EnergyHistogram(
        cpu_indices=[0], 
        gpu_indices=[0], 
        prometheus_url='http://localhost:9091', 
        job='training_energy_histogram',
        gpu_bucket_range = [10.0, 25.0, 50.0, 100.0],
        cpu_bucket_range = [5.0, 15.0, 30.0, 50.0],
        dram_bucket_range = [2.0, 8.0, 20.0, 40.0],
    )
    ```

If no custom `bucket ranges` are specified, Zeus uses these default ranges:
```
- GPU: [50.0, 100.0, 200.0, 500.0, 1000.0]
- CPU: [10.0, 20.0, 50.0, 100.0, 200.0]
- DRAM: [5.0, 10.0, 20.0, 50.0, 150.0]
```
!!! Warning
    Empty bucket ranges (e.g., []) are not allowed and will raise an error. Ensure you provide a valid range for each device or use the defaults.

 
[`EnergyCumulativeCounter`][zeus.metric.EnergyCumulativeCounter] monitors cumulative energy consumption. It tracks energy usage over time, without resetting the values, and is updated periodically.

```python hl_lines="2 5-15"

from zeus.metric import EnergyCumulativeCounter

if __name__ == "__main__":

    cumulative_counter_metric = EnergyCumulativeCounter(
        cpu_indices=[0], 
        gpu_indices=[0], 
        update_period=2,  
        prometheus_url='http://localhost:9091',
        job='energy_counter_job'
    )
    train_loader = range(10) 
    val_loader = range(5)  

    cumulative_counter_metric.begin_window("training_energy_monitoring")

    for epoch in range(100):  
        print(f"\n--- Epoch {epoch} ---")
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1:.2f}%.")

        # Simulate additional operations outside of training
        print("\nSimulating additional operations...")
        time.sleep(10)

    cumulative_counter_metric.end_window("training_energy_monitoring")
```
In this example, `cumulative_counter_metric` monitors energy usage throughout the entire training process rather than on a per-epoch basis. The `update_period` parameter defines how often the energy measurements are updated and pushed to Prometheus. 

[`PowerGauge`][zeus.metric.PowerGauge] tracks real-time power consumption using Prometheus Gauges which monitors fluctuating values such as power usage.

```python hl_lines="2 5-15"
from zeus.metric import PowerGauge

if __name__ == "__main__":

    power_gauge_metric = PowerGauge(
        gpu_indices=[0], 
        update_period=2,  
        prometheus_url='http://localhost:9091',
        job='power_gauge_job'
    )
    train_loader = range(10) 
    val_loader = range(5)  

    power_gauge_metric.begin_window("training_power_monitoring")

    for epoch in range(100):  
        print(f"\n--- Epoch {epoch} ---")
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1:.2f}%.")

        # Simulate additional operations outside of training
        print("\nSimulating additional operations...")
        time.sleep(10)

    power_gauge_metric.end_window("training_power_monitoring")
```
The `update_period` parameter defines how often the power datas are updated and pushed to Prometheus.


### How to Query Metrics in Prometheus

Energy for a specific window:
```promql
energy_monitor_gpu_energy_joules{window="epoch_energy"}
```

Sum of energy for a specific window:
```promql
sum(energy_monitor_gpu_energy_joules) by (window)
```

Sum of energy for specific GPU across all windows:
```promql
sum(energy_monitor_gpu_energy_joules{index="0"})
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

## Hardware Support
We currently support both NVIDIA (via NVML) and AMD GPUs (via AMDSMI, with ROCm 6.1 or later).

### `get_gpus`
The [`get_gpus`][zeus.device.get_gpus] function returns a [`GPUs`][zeus.device.gpu.GPUs] object, which can be either an [`NVIDIAGPUs`][zeus.device.gpu.NVIDIAGPUs] or [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object depending on the availability of `nvml` or `amdsmi`. Each [`GPUs`][zeus.device.gpu.GPUs] object contains one or more [`GPU`][zeus.device.gpu.common.GPU] instances, which are specifically [`NVIDIAGPU`][zeus.device.gpu.nvidia.NVIDIAGPU] or [`AMDGPU`][zeus.device.gpu.amd.AMDGPU] objects.

These [`GPU`][zeus.device.gpu.common.GPU] objects directly call respective `nvml` or `amdsmi` methods, providing a one-to-one mapping of methods for seamless GPU abstraction and support for multiple GPU types. For example:
- [`NVIDIAGPU.getName`][zeus.device.gpu.nvidia.NVIDIAGPU.getName] calls `pynvml.nvmlDeviceGetName`.
- [`AMDGPU.getName`][zeus.device.gpu.amd.AMDGPU.getName] calls `amdsmi.amdsmi_get_gpu_asic_info`.

### Notes on AMD GPUs

#### AMD GPUs Initialization
`amdsmi.amdsmi_get_energy_count` sometimes returns invalid values on certain GPUs or ROCm versions (e.g., MI100 on ROCm 6.2). See [ROCm issue #38](https://github.com/ROCm/amdsmi/issues/38) for more details. During the [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object initialization, we call `amdsmi.amdsmi_get_energy_count` twice for each GPU, with a 0.5-second delay between calls. This difference is compared to power measurements to determine if `amdsmi.amdsmi_get_energy_count` is stable and reliable. Initialization takes 0.5 seconds regardless of the number of AMD GPUs.

`amdsmi.amdsmi_get_power_info` provides "average_socket_power" and "current_socket_power" fields, but the "current_socket_power" field is sometimes not supported and returns "N/A." During the [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object initialization, this method is checked, and if "N/A" is returned, the [`AMDGPU.getInstantPowerUsage`][zeus.device.gpu.amd.AMDGPU.getInstantPowerUsage] method is disabled. Instead, [`AMDGPU.getAveragePowerUsage`][zeus.device.gpu.amd.AMDGPU.getAveragePowerUsage] needs to be used.

#### Supported AMD SMI Versions
Only ROCm >= 6.1 is supported, as the AMDSMI APIs for power and energy return wrong values. For more information, see [ROCm issue #22](https://github.com/ROCm/amdsmi/issues/22). Ensure your `amdsmi` and ROCm versions are up to date.
