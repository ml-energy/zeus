# Getting Started

Most of the common setup steps are described in this page.
Some optimizers or examples may require some extra setup steps, which are described in the corresponding documentation.

## Installing the Python package

### From PyPI

Install the Zeus Python package simply with:

```sh
pip install zeus
```

### From source for development

You can also install Zeus from source by cloning our GitHub repository.
Specifically for development, you can do an editable installation with extra dev dependencies:

```sh
git clone https://github.com/ml-energy/zeus.git
cd zeus
pip install -e '.[dev]'
```

## Using Docker

!!! Important "Dependencies"
    You should have the following already installed on your system:

    - [Docker](https://docs.docker.com/engine/install/)
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    
Our Docker image should suit most of the use cases for Zeus.
On top of the `nvidia/cuda:11.8.0-base-ubuntu22.04` image, we add:

- Miniconda 3, PyTorch, and Torchvision
- A copy of the Zeus repo in `/workspace/zeus`

??? Quote "docker/Dockerfile"
    ```Dockerfile title="Dockerfile"
    --8<-- "docker/Dockerfile"
    ```

The default command would be:

``` { .sh .annotate }
docker run -it \
    --gpus all \              # (1)!
    --cap-add SYS_ADMIN \   # (2)!
    --ipc host \          # (3)!
    -v /sys/class/powercap/intel-rapl:/zeus_sys/class/powercap/intel-rapl \ # (4)!
    mlenergy/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container. See [Docker docs](https://docs.docker.com/engine/containers/resource_constraints/#expose-gpus-for-use) for more about the `--gpus` argument.
2. The `SYS_ADMIN` Linux security capability is needed to change the GPU's power limit or frequency. See [here](#system-privileges) for details and alternatives.
3. PyTorch DataLoader workers need enough shared memory for IPC. Without this, they may run out of shared memory and die.
4. Zeus reads Intel RAPL metrics for CPU/DRAM energy measurement through the `sysfs` interface. Docker disables this by default, so we need to mount it into the container separately (under `/zeus_sys`).

Especially, `--cap-add SYS_ADMIN` is to be able to change the GPU's power limit or frequency, and `-v /sys/class/powercap/intel-rapl:/zeus_sys/class/powercap/intel-rapl` is to be able to measure CPU/DRAM energy via Intel RAPL.
See [System privileges](#system-privileges) for details.

### Pulling from Docker Hub

Pre-built images are hosted on [Docker Hub](https://hub.docker.com/r/mlenergy/zeus){.external}.
There are three types of images available:

- `latest`: The latest versioned release.
- `v*`: Each versioned release.
- `master`: The `HEAD` commit of Zeus. Usually stable enough, and you will get all the new features.

### Building the image locally

You should specify `TARGETARCH` to be one of `amd64` or `arm64` based on your environment:

```sh
git clone https://github.com/ml-energy/zeus.git
cd zeus
docker build -t mlenergy/zeus:master --build-arg TARGETARCH=amd64 -f docker/Dockerfile .
```

## Verifying installation

After installing the Zeus package, you can run the following to see whether packages and hardware are properly detected by Zeus.

```console
$ python -m zeus.show_env
================================================================================

Python version: 3.9.19

================================================================================

[2024-09-09 16:40:14,495] [zeus.utils.framework](framework.py:25) PyTorch with CUDA support is available.
[2024-09-09 16:40:14,496] [zeus.utils.framework](framework.py:45) JAX is not available

Package availability and versions:
  Zeus: 0.10.0
  PyTorch: 2.4.1+cu121
  JAX: not available

================================================================================

[2024-09-09 16:40:14,512] [zeus.device.gpu.nvidia](nvidia.py:46) pynvml is available and initialized.

GPU availability:
  GPU 0: NVIDIA A40

================================================================================

[2024-09-09 16:40:14,519] [zeus.device.cpu.rapl](rapl.py:136) RAPL is available.
[2024-09-09 16:40:14,519] [RaplWraparoundTracker](rapl.py:82) Monitoring wrap around of /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
[2024-09-09 16:40:14,528] [RaplWraparoundTracker](rapl.py:82) Monitoring wrap around of /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
[2024-09-09 16:40:14,533] [RaplWraparoundTracker](rapl.py:82) Monitoring wrap around of /sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj
[2024-09-09 16:40:14,535] [RaplWraparoundTracker](rapl.py:82) Monitoring wrap around of /sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj

CPU availability:
  CPU 0:
    CPU measurements available (/sys/class/powercap/intel-rapl/intel-rapl:0)
    DRAM measurements available (/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0)
  CPU 1:
    CPU measurements available (/sys/class/powercap/intel-rapl/intel-rapl:1)
    DRAM measurements available (/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0)

================================================================================
```

## System privileges

### When are extra system privileges needed?

1. **CPU energy measurement**: `root` privileges are needed when measuring CPU energy through the Intel RAPL interface. This is due to a [security issue](https://www.cve.org/CVERecord?id=CVE-2020-8694). Specifically, this is needed if you want to measure CPU energy via [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] with `cpu_indices`.
2. **GPU energy optimization**: The Linux security capability `SYS_ADMIN` (`root` is fine as well as it's stronger) is required in order to change the GPU's power limit or frequency. Specifically, this is needed by the [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] and the [`PipelineFrequencyOptimizer`][zeus.optimizer.pipeline_frequency.PipelineFrequencyOptimizer].

### Option 1: Running applications in a Docker container

For CPU energy measurement, you are `root` inside a Docker container. You will just need to mount the RAPL sysfs directory into the Docker container. See [here](#using-docker) for instructions.

For GPU energy optimization, you can pass `--cap-add SYS_ADMIN` to `docker run`.
Since this significantly simplifies running Zeus, we recommend users to consider this option first.
This is also possible for Kubernetes Pods with `securityContext.capabilities.add` in container specs ([docs](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/#set-capabilities-for-a-container){.external}).

### Option 2: Deploying the Zeus daemon (`zeusd`)

Granting `SYS_ADMIN` to the entire application just to be able to change the GPU's configuration is [granting too much](https://en.wikipedia.org/wiki/Principle_of_least_privilege){.external}.
Instead, Zeus provides the [**Zeus daemon** or `zeusd`](https://github.com/ml-energy/zeus/tree/master/zeusd){.external}, which is a simple server/daemon process that is designed to run with admin privileges and exposes the minimal set of APIs wrapping NVML methods for changing the GPU's configuration.
Then, an unprivileged (i.e., run normally by any user) application can ask `zeusd` via a Unix Domain Socket to change the local node's GPU configuration on its behalf.

To deploy `zeusd`:

``` { .sh .annotate }
# Install zeusd
cargo install zeusd

# Run zeusd with admin privileges
sudo zeusd \
    --socket-path /var/run/zeusd.sock \   # (1)!
    --socket-permissions 666            # (2)!
```

1. Unix domain socket path that `zeusd` listens to.
2. Applications need *write* access to the socket to be able to talk to `zeusd`. This string is interpreted as [UNIX file permissions](https://en.wikipedia.org/wiki/File-system_permissions#Numeric_notation).

We're currently working on adding Intel RAPL support to the Zeus daemon ([tracking issue](https://github.com/ml-energy/zeus/issues/110)).
We plan to land this feature at the end of 2024.

### Option 3: Running applications with `sudo`

This is probably the worst option.
However, if none of the options above work, you can run your application with `sudo`, which is essentially `root` and automatically has `SYS_ADMIN`.

## Next Steps

- [Measuring](../measure/index.md) energy with the [`ZeusMonitor`][zeus.monitor.ZeusMonitor], programmatically or in the command line.
- [Optimizing](../optimize/index.md) energy with Zeus energy optimizers.
