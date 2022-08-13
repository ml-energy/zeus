# Setting Up the Environment

We encourage users to do everything inside a Docker container spawned with our pre-built Docker image.

## Zeus Docker image

We provide a pre-built Docker image in Docker Hub: https://hub.docker.com/r/symbioticlab/zeus.
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

```sh
docker run -it \
    --gpus 1                    `# Mount one GPU` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --shm-size 64G              `# PyTorch DataLoader workers need enough shm` \
    symbioticlab/zeus:latest \
    bash
```

If you would like your changes to `zeus/` outside the container to be immediately applied inside the container, mount the repository into the container:

```sh
# Working directory is repository root
docker run -it \
    --gpus 1                    `# Mount one GPU` \
    --cap-add SYS_ADMIN         `# Needed to change the power limit of the GPU` \
    --shm-size 64G              `# PyTorch DataLoader workers need enough shm` \
    -v $(pwd):/workspace/zeus   `# Mount the repo into the container` \
    symbioticlab/zeus:latest \
    bash
```
