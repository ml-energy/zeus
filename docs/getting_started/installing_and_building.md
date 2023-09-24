# Installing and Building Zeus Components

This document explains how to install the [`zeus`][zeus] Python package.

!!! Tip
    We encourage users to utilize our Docker image. Please refer to [Environment setup](./environment.md). Quick command:
    ```bash
    docker run -it --gpus all --cap-add SYS_ADMIN --ipc host mlenergy/zeus:latest bash
    ```


## `zeus` Python package

!!! Note
    This is already installed inside the container if you're using our Docker image.

### Dependencies

Install PyTorch. For instance, if you have CUDA 11.8:

```bash
pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### Install `zeus`

To install the latest release of `zeus`:

```bash
pip install zeus-ml
```

If you would like to follow the `HEAD`:

```bash
git clone https://github.com/ml-energy/zeus.git zeus
cd zeus
pip install .
```

For those would like to make changes to the source code and run them, we suggest an editable installation:

```bash
git clone https://github.com/ml-energy/zeus.git zeus
cd zeus
pip install -e .
```


## Zeus power monitor

!!! Warning
    The C++ Zeus power monitor is now deprecated as we've switched to a Python-based power monitor.
    See [`PowerMonitor`][zeus.monitor.power.PowerMonitor] or run `python -m zeus.monitor --help`.

### Dependencies

All dependencies are pre-installed if you're using our Docker image.  

1. CMake >= 3.22
1. CUDAToolkit, especially NVML (`libnvidia-ml.so`)

### Configuration

You can change the `#define`s in lines 28 to 31 in [`zemo/zemo.hpp`](https://github.com/ml-energy/zeus/tree/master/zeus_monitor/zemo/zemo.hpp) to configure what information from the GPU is polled.
By default, only the momentary power draw of the GPU will be collected.

### Building the power monitor

```bash
# Working directory is repository root
cd zeus_monitor
cmake .
make
```

The resulting power monitor binary is `zeus_monitor/zeus_monitor` (`/workspace/zeus/zeus_monitor/zeus_monitor` inside the Docker container).
