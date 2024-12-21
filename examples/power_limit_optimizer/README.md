# Integrating the power limit optimizer with ImageNet training

This example will demonstrate how to integrate Zeus with `torchvision` and the ImageNet dataset.

[`train_single.py`](train_single.py) and [`train_dp.py`](train_dp.py) were adapted and simplified from [PyTorch's example training code for ImageNet](https://github.com/pytorch/examples/blob/main/imagenet/main.py). [`train_fsdp.py`](train_fsdp.py) was adapted from [Getting Started with Fully Sharded Data Parallel(FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

[`train_single.py`](train_single.py) is for simple single GPU training, [`train_dp.py`](train_dp.py) is for data parallel training with PyTorch DDP, and [`train_fsdp.py`](train_fsdp.py) is for Fully Sharded Data Parallel training. 

## Dependencies

All packages (including torchvision) are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/#using-docker).
You just need to download and extract the ImageNet data and mount it to the Docker container with the `-v` option (first step below).

1. Download the ILSVRC2012 dataset from [the ImageNet homepage](http://www.image-net.org/).
    Then, extract archives using [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) provided by PyTorch.
1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    pip install torchvision==0.15.2
    ```

## `ZeusMonitor` and `GlobalPowerLimitOptimizer`

- [`ZeusMonitor`](http://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor): Measures the GPU time and energy consumption of arbitrary code blocks.
- [`GlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.GlobalPowerLimitOptimizer): Online-profiles each power limit with `ZeusMonitor` and finds the cost-optimal power limit.

## Multi-GPU Distributed Training (Pytorch DDP and FSDP)

When using `ZeusMonitor` and/or `GlobalPowerLimitOptimizer` in a multi-GPU Distributed context, construct one instance of `ZeusMonitor` and/or `GlobalPowerLimitOptimizer` per local rank (per GPU on each node), and pass in the local rank to `ZeusMonitor` as shown below:

```python
monitor = ZeusMonitor(gpu_indices=[local_rank]) # pass in local rank to gpu_indices.
plo = GlobalPowerLimitOptimizer(monitor)
```

Ensure that only one GPU is monitored per `ZeusMonitor`. Internally, `GlobalPowerLimitOptimizer` performs an [AllReduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) to aggregate time and energy measurements across all GPUs before making a power limit decision.

## Example command

You can specify the maximum training time slowdown factor (1.0 means no slowdown) by setting `ZEUS_MAX_SLOWDOWN`. The default is set to 1.1 in this example script, meaning the lowest power limit that keeps training time inflation within 10% will be automatically found.
`GlobalPowerLimitOptimizer` supports other optimal power limit selection strategies. See [here](https://ml.energy/zeus/reference/optimizer/power_limit).

```bash
# Single-GPU
python train_single.py \
    [DATA_DIR] \
    --gpu 0                 `# Specify the GPU id to use`

# Multi-GPU Distributed Data Parallel
torchrun \
    --nnodes 1 \
    --nproc_per_node gpu    `# Number of processes per node, should be equal to the number of GPUs.` \
                            `# When set to 'gpu', it means use all the GPUs available.` \
    train_dp.py \
    [DATA_DIR]

# Multi-GPU Fully Sharded Data Parallel
torchrun \
    --nnodes 1 \
    --nproc_per_node=gpu    `# Number of processes per node, should be equal to the number of GPUs.` \
    train_fsdp.py \
    [DATA_DIR]
```

