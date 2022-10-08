# Integrating Zeus with torchvision and ImageNet

This example will demonstrate how to integrate Zeus with `torchvision` and the ImageNet dataset.
Also, this example will show how to enable Zeus distributed data parallel training mode with Multi-GPU on a single node. 

[`train.py`](train.py) is adapted from [PyTorch's example training code for ImageNet](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
Please make sure to take a careful look at the docstring of [`train.py`](train.py), especially the `Important notes` and `Simlified example code`, before starting producing your own data parallel training code.
You can search for `# ZEUS` in [`train.py`](train.py) for noteworthy places that require modification from conventional training scripts.
Parts related to data parallel is marked with `# DATA PARALLEL`.

**Usages**

- Zeus
    - [Running Zeus for a single job](#running-zeus-for-a-single-job)
    - [Running Zeus over multiple recurrences](#running-zeus-over-multiple-recurrences)
- Extra
    - [Just training a vision model on ImageNet](#just-training-a-vision-model-on-imagenet)

## Running Zeus for a single job

While our paper is about optimizing the batch size and power limit over multiple recurrences of the job, it is also possible to use just [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) to JIT-profile and optimize the power limit.

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).
You just need to download the ImageNet data and mount it to the Docker container with the `-v` option.

1. Download the ILSVRC2012 dataset from [the ImageNet homepage](http://www.image-net.org/).
    Then, extract archives using [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) provided by PyTorch.
1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

[`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) interfaces with the outside world via environment variables.
Check out its [class reference](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) for details.

Only `ZEUS_TARGET_METRIC` is required; other environment variables have default values as shown below.

```bash
export ZEUS_TARGET_METRIC="0.50"               # Stop training when target val metric is reached
export ZEUS_LOG_DIR="zeus_log"                 # Directory to store profiling logs
export ZEUS_JOB_ID="zeus"                      # Used to distinguish recurrences, so not important
export ZEUS_COST_THRESH="inf"                  # Kill training when cost (Equation 2) exceeds this
export ZEUS_ETA_KNOB="0.5"                     # Knob to tradeoff energy and time (Equation 2)
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor" # Path to power monitor
export ZEUS_PROFILE_PARAMS="20,80"             # warmup_iters,profile_iters for each power limit
export ZEUS_USE_OPTIMAL_PL="True"              # Whether to acutally use the optimal PL found

# Single-GPU
python train.py \
    [DATA_DIR] \
    --gpu 0                 `# Specify the GPU id to use` \
    --zeus

# Multi-GPU Data Parallel
# Please check out train.py for more launching methods.
torchrun \
    --nnodes 1 \
    --nproc_per_node gpu    `# Number of processes per node, should be equal to the number of GPUs.` \
                            `# When setting to 'gpu', it means use all the GPUs available, i.e.` \
    train.py [DATA_DIR] \
    --zeus \
    --torchrun
```

## Running Zeus over multiple recurrences

This example shows how to integrate [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) and drive batch size and power optimizations with [`ZeusMaster`](https://ml.energy/zeus/reference/run/master/#zeus.run.master.ZeusMaster).

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).
You just need to download the ImageNet data and mount it to the Docker container with the `-v` option.

1. Download the ILSVRC2012 dataset from [the ImageNet homepage](http://www.image-net.org/).
    Then, extract archives using [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) provided by PyTorch.
1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

```sh
# All arguments shown below are default values.
# Multi-GPU Data Parallel
python run_zeus.py \
    [DATA_DIR]             `# Specify the location of ImageNet dataset` \
    --seed 1 \
    --b_0 256 \
    --lr_0 0.1 \
    --b_min 8 \
    --b_max 1024 \
    --num_recurrence 100 \
    --eta_knob 0.5 \
    --beta_knob 2.0 \
    --target_metric 0.50 \
    --max_epochs 100
```

## Just training a vision model on ImageNet

[`train.py`](train.py) can also be used as a simple training script, without having to do anything with Zeus.

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).
You just need to download the ImageNet data and mount it to the Docker container with the `-v` option.

1. Download the ILSVRC2012 dataset from [the ImageNet homepage](http://www.image-net.org/).
    Then, extract archives using [this script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) provided by PyTorch.
2. Install PyTorch, `torchvision`, and `cudatoolkit`:
    ```sh
    conda install -c pytorch pytorch==1.10.1 torchvision==0.11.2 cudatoolkit==11.3.1
    ```

### Example command

```sh
# Single-GPU
python train.py \
    [DATA_DIR] \
    --epochs 100 \
    --batch_size 1024 \
    --gpu 0                 `# Specify the GPU id to use` \

# Multi-GPU Data Parallel
torchrun \
    --nnodes 1 \
    --nproc_per_node gpu    `# Number of processes per node, should be equal to the number of GPUs.` \
                            `# When setting to 'gpu', it means use all the GPUs available, i.e.` \
    train.py [DATA_DIR] \
    --epochs 100 \
    --batch_size 1024 \
    --torchrun
```
