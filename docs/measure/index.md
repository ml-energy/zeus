# Measuring Energy

!!! Important
    Please refer to the [Getting Started](../getting_started/index.md) guide to first install Zeus and set up your environment before proceeding with this section.
    
!!! Tip
    Once you've installed Zeus, you can use our [environment validation script](../getting_started/index.md#verifying-installation) to see if devices are being detected by Zeus as expected.

Zeus makes it very easy to measure time, power, and energy both programmatically in Python and also on the command line.
Measuring power and energy is also very low overhead, typically taking less than 10 ms for each call.

## Programmatic measurement

### Time and energy consumption of a chunk of code

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

### Power consumption over time

Apart from energy, you can also measure the power consumption of GPUs over time by directly using the [`PowerMonitor`][zeus.monitor.power.PowerMonitor].
It measures power in three *power domains*:  
- **GPU average power**: Windowed average power consumption of the GPU over a one-second interval.
- **GPU instantaneous power**: Instantaneous power consumption of the GPU at the time of the query.
- **GPU memory average power** (Hopper or newer): Windowed average power consumption of the GPU's memory.

!!! Important
    Not all GPUs support all power domains, and this is not really documented well. You'll have to check on your GPU by instantiating [`PowerMonitor`][zeus.monitor.power.PowerMonitor], which will automatically detect supported power domains.

When `PowerMonitor` is instantiated, it spawns separate processes that poll the device's power consumption API and collects deduplicated power samples in-memory.
Then, you can call [`get_all_power_timelines`][zeus.monitor.power.PowerMonitor.get_all_power_timelines] or [`get_power_timeline`][zeus.monitor.power.PowerMonitor.get_power_timeline] for a specific power domain to retrieve the power samples collected either for the whole lifetime of the monitor, or for a specific time window.

### Synchronizing CPU and GPU computations

Deep learning frameworks typically run actual computation on GPUs in an asynchronous fashion.
That is, the CPU (Python interpreter) asynchronously dispatches computations to run on the GPU and moves on to dispatch the next computation without waiting for the GPU to finish.
This helps GPUs achieve higher utilization with less idle time.

Due to this asynchronous nature of Deep Learning frameworks, we need to be careful when we want to take time and energy measurements of GPU execution.
We want *only and all* the computations dispatched between `begin_window` and `end_window` to be captured by our time and energy measurement.
That's what the `sync_execution_with` parameter in [`ZeusMonitor`][zeus.monitor.ZeusMonitor] and `sync_execution` paramter in [`begin_window`][zeus.monitor.ZeusMonitor.begin_window] and [`end_window`][zeus.monitor.ZeusMonitor.end_window] are for.
Depending on the Deep Learning framework you're using (currently PyTorch and JAX are supported), [`ZeusMonitor`][zeus.monitor.ZeusMonitor] will automatically synchronize CPU and GPU execution to make sure all and only the computations dispatched between the window are captured.

!!! Tip
    Zeus has one function used globally across the codebase for device synchronization: [`sync_execution`][zeus.utils.framework.sync_execution].

!!! Warning
    [`ZeusMonitor`][zeus.monitor.ZeusMonitor] covers only the common and simple case of device synchronization, when GPU indices (`gpu_indices`) correspond to one whole physical device.
    This is usually what you want, except when using more advanced device partitioning (e.g., using `--xla_force_host_platform_device_count` in JAX to partition CPUs into more pieces).
    In such cases, you probably want to opt out from using this function and handle synchronization manually at the appropriate granularity.

## Distributed power measurement and aggregation

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] is local to a single machine, but sometimes, you may want to monitor power across multiple nodes in a cluster.
In this case, you can run the Zeus daemon ([zeusd](https://crates.io/crates/zeusd)) on each machine and stream power readings over SSE (Server-Sent Events) to a central client ([`PowerStreamingClient`][zeus.monitor.power_streaming.PowerStreamingClient]) for real-time monitoring and aggregation.

Each Zeus daemon instance is described by a [`ZeusdTcpConfig`][zeus.monitor.power_streaming.ZeusdTcpConfig] (for TCP connections) or [`ZeusdUdsConfig`][zeus.monitor.power_streaming.ZeusdUdsConfig] (for Unix domain sockets), which specifies the endpoint and optionally which GPU/CPU indices to monitor.
Both `gpu_indices` and `cpu_indices` follow the same convention as [`ZeusMonitor`][zeus.monitor.ZeusMonitor]:

- `None` (default): Stream all available devices.
- A list of indices (e.g., `[0, 1]`): Stream only those devices.
- An empty list (`[]`): Skip streaming for that device type entirely.

```python
from zeus.monitor.power_streaming import (
    PowerStreamingClient,
    ZeusdTcpConfig,
    ZeusdUdsConfig,
)

# SSE connections start immediately on construction.
client = PowerStreamingClient(
    servers=[
        ZeusdTcpConfig(
            host="node1", port=4938,
            gpu_indices=[0, 1, 2, 3],
            cpu_indices=[0],        # stream only CPU package 0
        ),
        ZeusdTcpConfig(host="node2", port=4938),  # all GPUs + all CPUs
        ZeusdUdsConfig(socket_path="/var/run/zeusd.sock"),  # local UDS
    ],
)

# Snapshot: latest readings from all endpoints, keyed by "host:port".
readings = client.get_power()
for key, pr in readings.items():
    print(f"{key}: timestamp={pr.timestamp_s:.3f}s")
    print(f"  GPU power: {pr.gpu_power_w}")
    for cpu_idx, cpu_reading in pr.cpu_power_w.items():
        print(f"  CPU {cpu_idx}: {cpu_reading.cpu_w:.1f} W, DRAM: {cpu_reading.dram_w}")

# Blocking iterator: yields a snapshot each time new SSE data arrives.
# Iteration stops when stop() is called.
for readings in client:
    for key, pr in readings.items():
        print(f"{key}: GPU={pr.gpu_power_w}, CPU={pr.cpu_power_w}")

# Async iterator: same as above, without blocking the event loop.
async for readings in client:
    for key, pr in readings.items():
        print(f"{key}: GPU={pr.gpu_power_w}, CPU={pr.cpu_power_w}")

client.stop()
```

[`get_power`][zeus.monitor.power_streaming.PowerStreamingClient.get_power] returns a dict mapping each `"host:port"` key to a [`PowerReadings`][zeus.monitor.power_streaming.PowerReadings] object containing a Unix timestamp, per-GPU power in watts, and per-CPU [`CpuPowerReading`][zeus.monitor.power_streaming.CpuPowerReading] objects (with `cpu_w` and optional `dram_w` fields, both in watts).

The client spawns one background thread per device type per zeusd endpoint on construction.
Each thread holds an SSE connection and automatically reconnects on disconnection.
Unless `cpu_indices` is set to `[]`, the client probes the zeusd one-shot CPU power endpoint on init; if RAPL is not available on that server, a warning is logged and CPU streaming is skipped.
Call `stop()` when done to cleanly shut down background threads.
zeusd uses demand-driven polling -- power is only read from the hardware while at least one client is connected, so idle endpoints consume no resources.

## Hardware Support

For GPUs, we currently support both NVIDIA (via NVML) and AMD GPUs (via AMDSMI, with ROCm 6.1 or later).

CPU measurement is supported for devices that have the RAPL interface built in.
This includes the majority of Intel CPUs and most modern AMD CPUs.
DRAM energy measurement are available on some CPUs as well.

To check CPU/GPU/DRAM measurement support, refer to [Verifying installation](../getting_started/index.md#verifying-installation).

Energy measurement for Apple Silicon and Jetson Platforms is supported as well. For more information, refer to [Apple Silicon](#apple-silicon) and [Jetson Platforms](#jetson-platforms).

### [`get_gpus`][zeus.device.get_gpus] and [`get_cpus`][zeus.device.get_cpus]

The [`get_gpus`][zeus.device.get_gpus] function returns a [`GPUs`][zeus.device.gpu.GPUs] object, which can be either an [`NVIDIAGPUs`][zeus.device.gpu.NVIDIAGPUs] or [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object depending on the availability of `nvml` or `amdsmi`. Each [`GPUs`][zeus.device.gpu.GPUs] object contains one or more [`GPU`][zeus.device.gpu.common.GPU] instances, which are specifically [`NVIDIAGPU`][zeus.device.gpu.nvidia.NVIDIAGPU] or [`AMDGPU`][zeus.device.gpu.amd.AMDGPU] objects.

These [`GPU`][zeus.device.gpu.common.GPU] objects directly call respective `nvml` or `amdsmi` methods, providing a one-to-one mapping of methods for seamless GPU abstraction and support for multiple GPU types. For example:
- [`NVIDIAGPU.get_name`][zeus.device.gpu.nvidia.NVIDIAGPU.get_name] calls `pynvml.nvmlDeviceGetName`.
- [`AMDGPU.get_name`][zeus.device.gpu.amd.AMDGPU.get_name] calls `amdsmi.amdsmi_get_gpu_asic_info`.

[`get_cpus`][zeus.device.get_cpus] is similar to [`get_gpus`][zeus.device.get_gpus], but rather abstracts over CPU vendors.

### Limitations of AMD GPU support

#### AMD GPUs Initialization
`amdsmi.amdsmi_get_energy_count` sometimes returns invalid values on certain GPUs or ROCm versions (e.g., MI100 on ROCm 6.2). See [ROCm issue #38](https://github.com/ROCm/amdsmi/issues/38) for more details. During the [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object initialization, we call `amdsmi.amdsmi_get_energy_count` twice for each GPU, with a 0.5-second delay between calls. This difference is compared to power measurements to determine if `amdsmi.amdsmi_get_energy_count` is stable and reliable. Initialization takes 0.5 seconds regardless of the number of AMD GPUs.

`amdsmi.amdsmi_get_power_info` provides "average_socket_power" and "current_socket_power" fields, but the "current_socket_power" field is sometimes not supported and returns "N/A." During the [`AMDGPUs`][zeus.device.gpu.AMDGPUs] object initialization, this method is checked, and if "N/A" is returned, the [`AMDGPU.get_instant_power_usage`][zeus.device.gpu.amd.AMDGPU.get_instant_power_usage] method is disabled. Instead, [`AMDGPU.get_average_power_usage`][zeus.device.gpu.amd.AMDGPU.get_average_power_usage] needs to be used.

#### Supported AMD SMI Versions
Only ROCm >= 6.1 is supported, as the AMDSMI APIs for power and energy return wrong values. For more information, see [ROCm issue #22](https://github.com/ROCm/amdsmi/issues/22). Ensure your `amdsmi` and ROCm versions are up to date.

### Note on NUMA CPUs

If you have more than one CPU sockets, for instance, running our [environment validation script](../getting_started/index.md#verifying-installation) will show two RAPL devices.
To only measure the energy consumption of the CPU used by the current Python process, you can use the [`get_current_cpu_index`][zeus.device.cpu.get_current_cpu_index] helper function to retrieve the CPU index where the specified process ID is running and pass in only that index to the `cpu_indices` argument.

### Apple Silicon

To enable Apple Silicon energy monitoring, you must have the optional `zeus-apple-silicon` dependency installed.

If you're installing Zeus for the first time, you can have this dependency installed automatically with
`pip install 'zeus[apple]'`. You can also install this dependency manually by running `pip install zeus-apple-silicon`. This dependency is maintained in a separate codebase, and you can find more information about it [here](https://github.com/ml-energy/zeus-apple-silicon).

**Note**: if you do not have an Apple Silicon processor, are not running macOS, or do not have the above dependency installed, the Zeus monitor will skip measuring energy for Apple Silicon.

Once the dependency is installed, you can conduct measurement as normal with the Zeus monitor ([Programmatic measurement](#programmatic-measurement)), and metrics for Apple Silicon will be included in a field called `soc_energy` within the [`Measurement`][zeus.monitor.energy.Measurement] object reported by [`end_window`][zeus.monitor.ZeusMonitor.end_window]. For example:

```python
# ...
mes = monitor.end_window("epoch")
apple_energy_metrics = mes.soc_energy
```

For Apple Silicon, the `soc_energy` field will include metrics for:

- On-chip CPU (`cpu_total_mj`)
- Every efficiency core (`efficiency_cores_mj`)
- Every performance core (`performance_cores_mj`)
- Efficiency core manager (`efficiency_core_manager_mj`)
- Performance core manager (`performance_core_manager_mj`)
- DRAM (`dram_mj`)
- On-chip GPU (`gpu_mj`)
- GPU SRAM (`gpu_sram_mj`)
- Apple Neural Engine (ANE) (`ane_mj`)

Note that units are in mJ.

Some metrics may be unavailable for monitoring depending on the specific processor (e.g., DRAM is sometimes unavailable on M1 macs). If a certain subsystem's energy could not be measured, its entry in the result object will simply hold `None`.

### Jetson Platforms

Energy measurement is currently supported for NVIDIA Jetson platforms. Similarly to Apple Silicon, metrics can be retrieved as normal with the Zeus monitor ([Programmatic measurement](#programmatic-measurement)), which collects them in the `soc_energy` field of the returned [`Measurement`][zeus.monitor.energy.Measurement] object. Metrics are reported by [`end_window`][zeus.monitor.ZeusMonitor.end_window]. For example:

```python
# ...
mes = monitor.end_window("epoch")
jetson_energy_metrics = mes.soc_energy
```

**Note**: if you do not have a Jetson processor or are not running Linux, the Zeus monitor will skip measuring energy for Jetson platforms.

For Jetson, the `soc_energy` field will include energy metrics for:

- On-chip CPU (`cpu_energy_mj`)
- On-chip GPU (`gpu_energy_mj`)
- Total chip energy (`total_energy_mj`)

Note that units are in mJ.

Some metrics may be unavailable for monitoring depending on the specific Jetson device model or configuration (e.g. custom-configured boards), though this is rarely the case. For any unavailable metrics, its entry in the result object will hold `None`.

## Metric Monitoring

You can export Zeus measurements as Prometheus metrics.
Three metrics are currently supported:  

1. Energy consumption of a fixed code range (Histogram)
2. Power draw over time (Gauge)
3. Cumulative energy consumption over time (Counter)

!!! Prerequisite
    As Zeus is a library integrated to applications that are not necessarily servers, Zeus uses the **push** model for metric collection. As such, the [Prometheus Push Gateway](https://prometheus.io/docs/instrumenting/pushing/) must be deployed and accessible to the Zeus-integrated application. Example Prometheus configurations can be found in our [docker examples](https://github.com/ml-energy/zeus/tree/master/docker/prometheus).

    ```sh
    docker run -d -p 9091:9091 prom/pushgateway
    ```

### Supported Metrics and Naming

Zeus organizes metrics using **static metric names** and **dynamic labels** for flexibility and ease of querying in Prometheus. Metric names are static and cannot be overridden, but users can customize the context of the metrics by naming the window when using `begin_window()` and `end_window()`.

**Metric Name** (`component` is `gpu`, `cpu`, or `dram`)

- Energy histogram: `energy_monitor_{component}_energy_joules`
- Cumulative energy counter: `energy_monitor_{component}_energy_joules`
- Power gauge: `power_monitor_{component}_power_watts`

Note that the power gauge metric only supports the GPU component at the moment. Tracking issue: [#128](https://github.com/ml-energy/zeus/issues/128)

**Labels**

- `window`: The user-defined window name provided to `begin_window()` and `end_window()` (e.g., `energy_histogram.begin_window("epoch_energy")`).
- `index`: The index of the device (e.g., `0` for GPU 0).

### [`EnergyHistogram`][zeus.metric.EnergyHistogram]

This metric records energy consumption for GPUs, CPUs, and DRAM as Prometheus Histograms. This is ideal for observing the energy consumption distribution of a fixed and repeated code range.

```python hl_lines="1 4-9 12 15"
from zeus.metric import EnergyHistogram

if __name__ == "__main__":
    energy_histogram = EnergyHistogram(
        cpu_indices=[0], 
        gpu_indices=[0], 
        prometheus_url='http://localhost:9091', 
        job='training_energy_histogram'
    )

    for epoch in range(100):
        energy_histogram.begin_window("epoch_energy")
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        energy_histogram.end_window("epoch_energy")
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1}%")

```

!!! Tip 
    Bucket ranges for GPUs, CPUs, and DRAM can be set during initialization.

    ```python
    energy_histogram = EnergyHistogram(
        cpu_indices=[0], 
        gpu_indices=[0], 
        prometheus_url="http://localhost:9091", 
        job="training_energy_histogram",
        gpu_bucket_range=[10.0, 25.0, 50.0, 100.0],
        cpu_bucket_range=[5.0, 15.0, 30.0, 50.0],
        dram_bucket_range=[2.0, 8.0, 20.0, 40.0],
    )
    ```

### [`EnergyCumulativeCounter`][zeus.metric.EnergyCumulativeCounter]

This metric monitors cumulative energy consumption over time.

```python hl_lines="1 4-10 14 22"
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

    cumulative_counter_metric.begin_window("training_energy")

    for epoch in range(100):  
        print(f"\n--- Epoch {epoch} ---")
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1:.2f}%.")

    cumulative_counter_metric.end_window("training_energy")
```

Metric observations are pushed to Prometheus every `update_period` seconds.

### [`PowerGauge`][zeus.metric.PowerGauge]

This metric tracks real-time power consumption using Prometheus Gauges.

```python hl_lines="1 4-9 13 21"
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

    power_gauge_metric.begin_window("training_power")

    for epoch in range(100):  
        print(f"\n--- Epoch {epoch} ---")
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        acc1 = validate(val_loader, model, criterion, args)
        print(f"Epoch {epoch} completed. Validation Accuracy: {acc1:.2f}%.")

    power_gauge_metric.end_window("training_power")
```

Metric observations are pushed to Prometheus every `update_period` seconds.

### Querying Metrics in Prometheus

Once metrics are pushed to Prometheus, you can use PromQL to run simple analytics.

*Energy for a specific window*
```promql
energy_monitor_gpu_energy_joules{window="epoch_energy"}
```

*Sum of energy for a specific window*
```promql
sum(energy_monitor_gpu_energy_joules) by (window)
```

*Sum of energy for specific GPU across all windows*
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

## Troubleshooting

### Repeated execution of the main script module after using Zeus monitors

Zeus monitors (e.g., [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor], [`PowerMonitor`][zeus.monitor.power.PowerMonitor], [`TemperatureMonitor`][zeus.monitor.temperature.TemperatureMonitor], [`CarbonEmissionMonitor`][zeus.monitor.carbon.CarbonEmissionMonitor], [`EnergyCostMonitor`][zeus.monitor.price.EnergyCostMonitor]) use the `spawn` start method for helper processes. Each spawned subprocess re-imports your `__main__` module (as `__mp_main__`), so any work done at import time, such as loading a model or instantiating a monitor, runs again in every process and can exhaust GPU/CPU memory. Keep heavy initialization under `if __name__ == "__main__":` or inside functions, so subprocess imports stay lightweight.

### CPU energy measurement missing or permission denied (Intel RAPL)

Reading CPU/DRAM energy via Intel RAPL requires root because of kernel restrictions. If you cannot run as root, disable CPU measurement with `cpu_indices=[]` or follow the steps in [System privileges](../getting_started/index.md#system-privileges) to grant access.
