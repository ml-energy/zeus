# Zeus Power Monitor

This is a simple GPU power monitor used by Zeus.
It polls NVML and writes outputs to the designated log file path.
Find a sample of its output in [`sample.csv`](sample.csv).

```console
$ ./zeus_monitor --help
Usage: ./zeus_monitor LOGFILE DURATION SLEEP_MS [GPU_IDX]
    Set DURATION to 0 to run indefinitely.
    If SLEEP_MS is 0, the monitor won't call sleep at all.
```


## Building

The Zeus monitor is pre-built for you if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).

### Dependencies

1. CMake >= 3.22
1. CUDAToolkit, especially NVML (`libnvidia-ml.so`)

### Building the power monitor

```sh
cmake .
make
```

The resulting power monitor binary is `zeus_monitor` (`/workspace/zeus/zeus_monitor/zeus_monitor` inside the Docker container).

## Zemo (Zeus Monitor) library

The `zemo` C++ header-only library provides simple functions to spawn an NVML-polling thread that writes results to a log file.

Change the `#define`s in lines 28 to 31 in [`zemo/zemo.hpp`](zemo/zemo.hpp) to configure what information from the GPU is polled.
By default, only the momentary power draw of the GPU will be collected.
