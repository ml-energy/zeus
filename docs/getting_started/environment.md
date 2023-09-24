# Setting Up the Environment

We encourage users to do everything inside a Docker container spawned with our pre-built Docker image.

!!! Tip
    Docker may not be an option for some users. In that case,

    - Python still needs the Linux `SYS_ADMIN` capability to change the GPU's power limit. One dirty way is to run Python with `sudo`.
    - Skim through our Dockerfile (shown below) to make sure you have the stuff that's being installed.
    - Follow the instructions in [Installing and Building](installing_and_building.md).

## Zeus Docker image

We provide a pre-built Docker image in [Docker Hub](https://hub.docker.com/r/mlenergy/zeus){.external}.
On top of the `nvidia/cuda:11.8.0-devel-ubuntu22.04` image, the following are added:

1. Miniconda3 23.3.1, PyTorch 2.0.1, torchvision 0.15.2
1. A copy of the Zeus repo in `/workspace/zeus`.
1. An editable install of the `zeus` package in `/workspace/zeus/zeus`. Users can override the copy of the repo by mounting the edited repo into the container. See instructions below.

??? Quote "Dockerfile"
    ```Dockerfile title="Dockerfile"
    --8<-- "Dockerfile"
    ```

!!! Tip
    If you want to build our Docker image locally, you should specify `TARGETARCH` to be one of `amd64` or `arm64` based on your environment's architecture:
    ```sh
    docker build -t mlenergy/zeus:master --build-arg TARGETARCH=amd64 .
    ```


### Dependencies

1. [`docker`](https://docs.docker.com/engine/install/)
1. [`nvidia-docker2`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Spawn the container

The default command would be:

``` { .sh .annotate }
docker run -it \
    --gpus all \                      # (1)!
    --cap-add SYS_ADMIN \           # (2)!
    --ipc host \                  # (3)!
    mlenergy/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container. `nvidia-docker2` provides this option.
2. `SYS_ADMIN` capability is needed to manage the power configurations of the GPU via NVML.
3. PyTorch DataLoader workers need enough shared memory for IPC. Without this, they may run out of shared memory and die.

Use the `-v` option to mount outside data into the container.
For instance, if you would like your changes to `zeus/` outside the container to be immediately applied inside the container, mount the repository into the container.
You can also mount training data into the container.

``` { .sh .annotate }
# Working directory is repository root
docker run -it \
    --gpus all \                               # (1)!
    --cap-add SYS_ADMIN \                    # (2)!
    --ipc host \                           # (3)!
    -v $(pwd):/workspace/zeus \          # (4)!
    -v /data/imagenet:/data/imagenet:ro \
    mlenergy/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container. `nvidia-docker2` provides this option.
2. `SYS_ADMIN` capability is needed to manage the power configurations of the GPU via NVML.
3. PyTorch DataLoader workers need enough shared memory for IPC. Without this, they may run out of shared memory and die.
4. Mounts the repository directory into the Docker container. Since the `zeus` installation inside the container is editable, changes you made outside will apply immediately.
