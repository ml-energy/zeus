# Integrating Zeus with torchvision and ImageNet

This example will demonstrate how to integrate Zeus with `torchvision` and the ImageNet dataset.

[`train_single.py`](train_single.py) and [`train_dp.py`](train_dp.py) were adapted and simplified from [PyTorch's example training code for ImageNet](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
The former script is for simple single GPU training, whereas the latter is for data parallel training with PyTorch DDP and [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html).

## Dependencies

All packages (including torchvision) are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).
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
- [`GlobalPowerLimitOptimizer`](/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.GlobalPowerLimitOptimizer): Online-profiles each power limit with `ZeusMonitor` and finds the cost-optimal power limit.

## Example command


Only `ZEUS_TARGET_METRIC` is required; other environment variables have default values as shown below.

```bash
# Single-GPU
python train_single.py \
    [DATA_DIR] \
    --gpu 0                 `# Specify the GPU id to use`

# Multi-GPU Data Parallel
torchrun \
    --nnodes 1 \
    --nproc_per_node gpu    `# Number of processes per node, should be equal to the number of GPUs.` \
                            `# When set to 'gpu', it means use all the GPUs available.` \
    train.py \
    [DATA_DIR]
```
