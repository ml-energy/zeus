# Getting Started

Most of the common setup steps are described in this page.
Some optimizers or examples may require some extra setup steps, which are described in the corresponding documentation.

## Installing the Python package

### From PyPI

Install the Zeus Python package simply with:

```sh
pip install zeus-ml
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
    --gpus all \                 # (1)!
    --cap-add SYS_ADMIN \       # (2)!
    --ipc host \               # (3)!
    mlenergy/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container.
2. `SYS_ADMIN` capability is needed to change the GPU's power limit or frequency. See [here](#system-privileges).
3. PyTorch DataLoader workers need enough shared memory for IPC. Without this, they may run out of shared memory and die.

!!! Tip "Overriding Zeus installation"
    Inside the container, `zeus`'s installation is editable (`pip install -e`).
    So, you can mount your locally modified Zeus repository into the right path in the container (`-v /path/to/zeus:/workspace/zeus`), and your modifications will automatically be applied without you having to run `pip install` again.

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

## System privileges

!!! Important "Nevermind if you're just measuring"
    No special system-level privileges are needed if you are just measuring time and energy.
    However, when you're looking into optimizing energy and if that method requires changing the GPU's power limit or SM frequency, special system-level privileges are required.

### When are extra system privileges needed?

The Linux capability `SYS_ADMIN` is required in order to change the GPU's power limit or frequency.
Specifically, this is needed by the [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] and the [`PipelineFrequencyOptimizer`][zeus.optimizer.pipeline_frequency.PipelineFrequencyOptimizer].

### Option 1: Running applications in a Docker container

Using Docker, you can pass `--cap-add SYS_ADMIN` to `docker run`.
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
    --socket-path /var/run/zeusd.sock \  # (1)!
    --socket-permissions 666            # (2)!
```

1. Unix domain socket path that `zeusd` listens to.
2. Applications need *write* access to the socket to be able to talk to `zeusd`. This string is interpreted as [UNIX file permissions](https://en.wikipedia.org/wiki/File-system_permissions#Numeric_notation).

### Option 3: Running applications with `sudo`

This is probably the worst option.
However, if none of the options above work, you can run your application with `sudo`, which automatically has `SYS_ADMIN`.

## Next Steps

- [Measuring](../measure/index.md) energy with the [`ZeusMonitor`][zeus.monitor.ZeusMonitor], programmatically or in the command line.
- [Optimizing](../optimize/index.md) energy with Zeus energy optimizers.
