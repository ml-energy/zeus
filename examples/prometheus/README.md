# Integrating the power limit optimizer with ImageNet training

This example will demonstrate how to integrate Zeus with `torchvision` and the ImageNet dataset.

[`train_single.py`](train_single.py) and [`train_dp.py`](train_dp.py) were adapted and simplified from [PyTorch's example training code for ImageNet](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
The former script is for simple single GPU training, whereas the latter is for data parallel training with PyTorch DDP and [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html).

## Dependencies

All packages (including torchvision and prometheus_client) are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).
You just need to download and extract the ImageNet data and mount it to the Docker container with the `-v` option (first step below).

1. Download the ILSVRC2012 dataset from [the ImageNet homepage](http://www.image-net.org/).
    Then, extract archives using [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) provided by PyTorch.
1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    pip install torchvision==0.15.2
    ```
1. Install `prometheus_client`:
    ```sh
    pip install zeus-ml[prometheus]
    ```

## EnergyHistogram, PowerGauge, and EnergyCumulativeCounter
- [`EnergyHistogram`](https://ml.energy/zeus/reference/metric/#zeus.metric.EnergyHistogram): Records energy consumption data for GPUs, CPUs, and DRAM and pushes the data to Prometheus as histogram metrics. This is useful for tracking energy usage distribution over time.
- [`PowerGauge`](https://ml.energy/zeus/reference/metric/#zeus.metric.PowerGauge): Monitors real-time GPU power usage and pushes the data to Prometheus as gauge metrics, which are updated at regular intervals.
- [`EnergyCumulativeCounter`](https://ml.energy/zeus/reference/metric/#zeus.metric.EnergyCumulativeCounter): Tracks cumulative energy consumption over time for CPUs and GPUs and pushes the results to Prometheus as counter metrics.

## `ZeusMonitor` and `GlobalPowerLimitOptimizer`

- [`ZeusMonitor`](http://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor): Measures the GPU time and energy consumption of arbitrary code blocks.
- [`GlobalPowerLimitOptimizer`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.GlobalPowerLimitOptimizer): Online-profiles each power limit with `ZeusMonitor` and finds the cost-optimal power limit.

## Example command

You can specify the maximum training time slowdown factor (1.0 means no slowdown) by setting `ZEUS_MAX_SLOWDOWN`. The default is set to 1.1 in this example script, meaning the lowest power limit that keeps training time inflation within 10% will be automatically found.
`GlobalPowerLimitOptimizer` supports other optimal power limit selection strategies. See [here](https://ml.energy/zeus/reference/optimizer/power_limit).

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
    train_dp.py \
    [DATA_DIR]
```


