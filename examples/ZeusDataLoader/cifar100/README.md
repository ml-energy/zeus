# Integrating Zeus with torchvision and CIFAR100

This example will demonstrate how to integrate Zeus with `torchvision` and the CIFAR100 dataset provided by it.
We believe that it would be straightforward to extend this example to support other image classification datasets such as CIFAR10 or ImageNet.

You can search for `# ZEUS` in [`train.py`](train.py) for noteworthy places that require modification from conventional training scripts.


**Usages**

- Zeus
    - [Running Zeus for a single job](#running-zeus-for-a-single-job)
    - [Running Zeus over multiple recurrences](#running-zeus-over-multiple-recurrences)
- Extra
    - [Just training a vision model on CIFAR100](#just-training-a-vision-model-on-cifar100)


## Running Zeus for a single job

While our paper is about optimizing the batch size and power limit over multiple recurrences of the job, it is also possible to use just [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) to JIT-profile and optimize the power limit.

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).

1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

[`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) interfaces with the outside world via environment variables.
Check out its [class reference](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) for details.

Only `ZEUS_TARGET_METRIC` is required; other environment variables below show their default values when omitted.

```bash
export ZEUS_TARGET_METRIC="0.50"               # Stop training when target val metric is reached
export ZEUS_LOG_DIR="zeus_log"                 # Directory to store profiling logs
export ZEUS_JOB_ID="zeus"                      # Used to distinguish recurrences, so not important
export ZEUS_COST_THRESH="inf"                  # Kill training when cost (Equation 2) exceeds this
export ZEUS_ETA_KNOB="0.5"                     # Knob to tradeoff energy and time (Equation 2)
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor" # Path to power monitor
export ZEUS_PROFILE_PARAMS="10,40"              # warmup_iters,profile_iters for each power limit
export ZEUS_USE_OPTIMAL_PL="True"              # Whether to acutally use the optimal PL found

python train.py \
    --zeus \
    --arch shufflenetv2 \
    --epochs 100 \
    --batch_size 128
```


## Running Zeus over multiple recurrences

This example shows how to integrate [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) and drive batch size and power optimizations with [`ZeusMaster`](https://ml.energy/zeus/reference/run/master/#zeus.run.master.ZeusMaster).

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).

1. Install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install `torchvision`:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

```sh
# All arguments shown below are default values.
python run_zeus.py \
    --seed 1 \
    --b_0 1024 \
    --b_min 8 \
    --b_max 4096 \
    --num_recurrence 100 \
    --eta_knob 0.5 \
    --beta_knob 2.0 \
    --target_metric 0.50 \
    --max_epochs 100
```


## Just training a vision model on CIFAR100

[`train.py`](train.py) can also be used as a simple training script, without having to do anything with Zeus.

### Dependencies

All packages are pre-installed if you're using our [Docker image](https://ml.energy/zeus/getting_started/environment/).

1. Install `torchvision`:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

```sh
python train.py \
    --arch shufflenetv2 \
    --epochs 100 \
    --batch_size 1024
```
