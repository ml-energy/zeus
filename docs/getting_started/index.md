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

### Obtaining privileges with Docker

Using Docker, you can pass `--cap-add SYS_ADMIN` to `docker run`.
Since this significantly simplifies running Zeus, we recommend users to consider this option first.
Also, since Zeus is running inside a container, there is less potential for damage even if things go wrong.

### Obtaining privileges with `sudo`

If you cannot use Docker, you can run your application with `sudo`.
This is not recommended due to security reasons, but it will work.

### GPU management server

It is fair to say that granting `SYS_ADMIN` to the application is itself giving too much privilege.
We just need to be able to change the GPU's power limit or frequency, instead of giving the process privileges to administer the system.
Thus, to reduce the attack surface, we are considering solutions such as a separate GPU management server process on a node ([tracking issue](https://github.com/ml-energy/zeus/issues/29)), which has `SYS_ADMIN`.
Then, an unprivileged application process can ask the GPU management server via a UDS to change the GPU's configuration on its behalf.

## Next Steps

- [Measuring](../measure/index.md) energy with the [`ZeusMonitor`][zeus.monitor.ZeusMonitor], programmatically or in the command line.
- [Optimizing](../optimize/index.md) energy with Zeus energy optimizers.
