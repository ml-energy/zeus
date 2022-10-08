# Setting Up the Environment

We encourage users to do everything inside a Docker container spawned with our pre-built Docker image.

## Zeus Docker image

We provide a pre-built Docker image in [Docker Hub](https://hub.docker.com/r/symbioticlab/zeus){.external}.
On top of the `nvidia/cuda:11.3.1-devel-ubuntu20.04` image, the following are provided:  

1. CMake 3.22.0
1. Miniconda3 4.12.0, PyTorch 1.10.1, torchvision 0.11.2, cudatoolkit 11.3.1
1. A copy of the Zeus repo in `/workspace/zeus`.
1. An editable install of the `zeus` package in `/workspace/zeus/zeus`. Users can override the copy of the repo by mounting the edited repo into the container. See instructions below.

??? Quote "Dockerfile"
    ```Dockerfile title="Dockerfile"
    --8<-- "Dockerfile"
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
    --shm-size 64G \              # (3)!
    symbioticlab/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container. `nvidia-docker2` provides this option.
2. `SYS_ADMIN` capability is needed to manage the power configurations of the GPU via NVML.
3. PyTorch DataLoader workers need enough shared memory for IPC. If the PyTorch training process dies with a Bus error, consider increasing this even more.

Use the `-v` option to mount outside data into the container.
For instance, if you would like your changes to `zeus/` outside the container to be immediately applied inside the container, mount the repository into the container.
You can also mount training data into the container.

``` { .sh .annotate }
# Working directory is repository root
docker run -it \
    --gpus all \                               # (1)!
    --cap-add SYS_ADMIN \                    # (2)!
    --shm-size 64G \                       # (3)!
    -v $(pwd):/workspace/zeus \          # (4)!
    -v /data/imagenet:/data/imagenet:ro \
    symbioticlab/zeus:latest \
    bash
```

1. Mounts all GPUs into the Docker container. `nvidia-docker2` provides this option.
2. `SYS_ADMIN` capability is needed to manage the power configurations of the GPU via NVML.
3. PyTorch DataLoader workers need enough shared memory for IPC. If the PyTorch training process dies with a Bus error, consider increasing this even more.
4. Mounts the repository directory into the Docker container. Since the `zeus` installation inside the container is editable, changes you made outside will apply immediately.
