---
name: measuring-energy
description: Measure the energy and power consumption of code with Zeus, on NVIDIA and AMD GPUs, Intel/AMD CPUs and DRAM (RAPL), Apple silicon, and NVIDIA Jetson. Use when asked to measure, monitor, log, or profile the energy consumption, power draw, or energy efficiency of a computing workload.
license: Apache-2.0
---

# Measuring Energy and Power with Zeus

Zeus is a Python library for measuring and optimizing the energy consumption of any computation running on GPUs, CPUs, and SoCs.
Install it with `pip install zeus` (Python 3.10 or later), or `pip install 'zeus[apple]'` to enable energy measurement on Apple silicon.
Run `python -m zeus.show_env` to check which frameworks and devices Zeus detects.
Full documentation: https://ml.energy/zeus

## Choosing an API

| Goal | API |
|---|---|
| Energy and time of a block of code | `zeus.monitor.ZeusMonitor` |
| Power draw over time | `zeus.monitor.PowerMonitor` |
| Low-variance energy per iteration of a repeatable function | `zeus.profile` |
| Custom tooling on raw device counters | `zeus.device` |

For a quick measurement without writing code, run `python -m zeus.monitor energy` (prints total energy on Ctrl-C) or `python -m zeus.monitor power` (prints power draw periodically).

## Permissions

GPU energy and power measurement requires no special privileges.
CPU and DRAM energy measurement uses Intel RAPL, which requires root due to kernel restrictions.
Without root, `ZeusMonitor` reports `cpu_energy=None` when `cpu_indices` is omitted or empty, and raises `RuntimeError` when a non-empty `cpu_indices` is passed explicitly.
Alternatives are running inside Docker as root with the RAPL sysfs directory mounted, or deploying the Zeus daemon (`zeusd`) and setting `ZEUSD_SOCK_PATH` so unprivileged processes relay RAPL reads through it.

## `ZeusMonitor`: energy and time of a code block

```python
from zeus.monitor import ZeusMonitor

def run_workload() -> None: ...

if __name__ == "__main__":
    monitor = ZeusMonitor(gpu_indices=[0, 1])

    monitor.begin_window("workload")
    run_workload()
    result = monitor.end_window("workload")

    print(f"{result.time} s, {result.total_energy} J")
    print(result.gpu_energy)  # {0: ..., 1: ...} in Joules
```

- `gpu_indices=None` (default) monitors all GPUs.
  Indices follow `CUDA_VISIBLE_DEVICES` (NVIDIA) or `HIP_VISIBLE_DEVICES` (AMD), so index N is `cuda:N` in frameworks like PyTorch.
- `end_window` returns a `Measurement` with `time` (seconds), `gpu_energy` (dict of GPU index to Joules), `cpu_energy` and `dram_energy` (dict of RAPL package index to Joules, or `None` when unavailable), and `soc_energy` (Apple silicon or Jetson subsystem energies in millijoules, or `None`).
  `result.total_energy` is the sum of GPU energy only.
- Multiple windows can be open at once, and they can nest or overlap.
  Window names must be unique among open windows; use `begin_window(key, restart=True)` in notebooks where a crashed cell may have left a window open, and `end_window(key, cancel=True)` to discard a window.
- `begin_window` and `end_window` synchronize asynchronously dispatched GPU computations (`torch.cuda.synchronize` by default) so the window captures exactly the code inside it.
  Pass `sync_execution_with="jax"` or `"cupy"` to the constructor for those frameworks, or `sync_execution=False` to `begin_window`/`end_window` when the workload uses none of them (otherwise a missing framework raises `RuntimeError` on machines with GPUs).
- GPU energy counters update roughly every 100 ms, so a very short window can read zero energy.
  Constructing with `approx_instant_energy=True` replaces such zero readings with instant power draw times window duration, which is an approximation, not a measurement.
  Never enable it unprompted: explain this implication to the user and get their approval first, or lengthen the window instead.
  The flag is also unsupported on AMD GPUs that cannot report instant power; `end_window` raises `ZeusGPUNotSupportedError` there.
- `log_file="measurement.csv"` writes one row per completed window.

Zeus monitors may spawn helper processes with the `spawn` start method, which re-imports your `__main__` module.
Keep monitor construction and heavy initialization (e.g., loading models or large data) under `if __name__ == "__main__":` or inside functions, or every helper process will repeat that work.

## `PowerMonitor`: power draw over time

```python
from zeus.monitor import PowerMonitor

def run_workload() -> None: ...

if __name__ == "__main__":
    monitor = PowerMonitor(gpu_indices=[0], update_period=0.1)
    run_workload()
    print(monitor.get_power())  # {0: ...} in Watts, latest sample
    timeline = monitor.get_power_timeline("device_instant")  # {0: [(timestamp, watts), ...]}
    monitor.stop()
```

- Polling starts on construction, with one background process per power domain.
  Domains are `device_instant`, `device_average`, and `memory_average`; only domains the GPU supports are monitored, and `get_all_power_timelines` returns every monitored one.
- `update_period=None` (default) infers each GPU model's power counter update period by polling; pass a value in seconds (at least 0.05) to skip that.
- `get_energy(start_time, end_time)` integrates the power timeline into Joules per GPU, using `time.time()` timestamps.
- All monitored GPUs must be the same model.
- Call `stop()` when done to terminate the polling processes.

## `zeus.profile`: thermally stable energy profiling

Back-to-back energy measurements of the same workload drift because the GPU heats up and hotter silicon leaks more power.
This module runs trials of the form cooldown (idle), then warmup iterations, then a measured window of iterations, yielding low-variance energy per iteration.
Use it when benchmarking or comparing configurations rather than measuring a single long run end to end.

```python
from zeus.monitor import ZeusMonitor
from zeus.profile import measure, profile_parameters

def one_iteration() -> None: ...

if __name__ == "__main__":
    monitor = ZeusMonitor(gpu_indices=[0])

    # If good durations are unknown, sweep both and pick the smallest valid values.
    measurement_report, cooldown_report = profile_parameters(one_iteration, monitor)

    # With known durations, run a single trial.
    trial = measure(one_iteration, monitor, measurement_duration=5.0, cooldown_duration=10.0)
    print(f"{trial.energy_per_iter} J/iter, {trial.time_per_iter} s/iter")
```

- The target function takes no arguments and runs one iteration of the workload.
- `TrialResult` holds `energy_per_iter`, `time_per_iter`, `total_energy`, `total_time`, `iterations`, and GPU `temperature_before`/`temperature_after`.
- `profile_parameters` (and the single-parameter variants `profile_measurement_duration` and `profile_cooldown_duration`) return `SweepReport` objects whose `entries` mark each configuration `is_valid` when the energy standard deviation across trials falls below `trial_stddev_threshold`.
  Printing a report gives a one-line-per-configuration summary.
- In a distributed setting, every rank creates its own `ZeusMonitor(gpu_indices=[local_rank])` and calls the same profiling function; results are aggregated across ranks and rank 0 logs.

## `zeus.device`: low-level device APIs

When the higher-level APIs do not fit, build on the device abstraction layer directly.

```python
from zeus.device import get_gpus, get_cpus, get_soc

gpus = get_gpus()  # NVML (NVIDIA) or AMDSMI (AMD) behind one interface
print(gpus.get_name(0))
print(gpus.get_instant_power_usage(0))  # milliwatts
if gpus.supports_get_total_energy_consumption(0):
    print(gpus.get_total_energy_consumption(0))  # cumulative millijoules

cpus = get_cpus()  # Intel RAPL
print(cpus.get_total_energy_consumption(0).cpu_mj)  # cumulative millijoules

soc = get_soc()  # Apple silicon or Jetson
print(soc.get_total_energy_consumption())
```

- Each getter raises `ZeusGPUInitError`, `ZeusCPUInitError`, or `ZeusSoCInitError` when the corresponding hardware or vendor library is unavailable.
- Units at this layer are milliwatts and millijoules; `zeus.monitor` converts to Watts and Joules.
- Energy counters are cumulative, so measure a window by subtracting two readings.
  Where `supports_get_total_energy_consumption` is `False` (e.g., NVIDIA GPUs older than Volta), integrate power samples over time instead, which is what `ZeusMonitor` does internally.
- GPU measurement methods, all taking a `gpu_index`: `get_instant_power_usage`, `get_average_power_usage`, `get_average_memory_power_usage`, `get_total_energy_consumption`, and `get_gpu_temperature`.
  Some AMD GPUs do not support instant power, in which case use `get_average_power_usage`.
- `CPUs.get_total_energy_consumption(index)` returns a `CpuDramMeasurement` with `cpu_mj` and `dram_mj` (the latter `None` when DRAM energy is unsupported); RAPL counter wraparound is handled internally.
- `SoC` also offers `begin_window`/`end_window` returning a platform-specific `SoCMeasurement`.
- The GPU interface additionally exposes power limit and frequency control (`set_power_management_limit`, `set_gpu_locked_clocks`, etc.), which require the `SYS_ADMIN` capability or `zeusd`.

## Troubleshooting

- Warning about zero energy consumption: the window is shorter than the GPU energy counter update period; lengthen the window, or propose `approx_instant_energy=True` to the user (see the `ZeusMonitor` section for its implications).
- `cpu_energy` is `None`: RAPL requires root; see the Permissions section.
- Out-of-memory or repeated log lines at startup: the `__main__` module is being re-imported by helper processes; add the `if __name__ == "__main__":` guard.
- `RuntimeError: Failed to import Pytorch`: the machine has GPUs but no PyTorch; set `sync_execution_with` to the framework in use or pass `sync_execution=False` to `begin_window`/`end_window`.
