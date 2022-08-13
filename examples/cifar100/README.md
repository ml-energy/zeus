# Integrating Zeus with torchvision and CIFAR100

This example will demonstrate how to integrate Zeus with `torchvision` and the CIFAR100 dataset provided by it.
We believe that it would be straightforward to extend this example to support other image classification datasets such as CIFAR10 or ImageNet.

You can search for `# ZEUS` in [`train.py`](train.py) for noteworthy places that require modification from conventional training scripts.


**Usages**

- Zeus
    - [Running Zeus for a single job](#running-zeus-for-a-single-job)
    - [Running Zeus over multiple recurrences](#running-zeus-over-multiple-recurrences)
- Extra
    - [Profiling power and time](#profiling-power-and-time)
    - [Just training a vision model on CIFAR100](#just-training-a-vision-model-on-cifar100)


## Running Zeus for a single job

While our paper is about optimizing the batch size and power limit over multiple recurrences of the job, it is also possible to use just `ZeusDataLoader` ([source](/zeus/run/dataloader.py)) to JIT-profile and optimize the power limit.

### Dependencies

1. Install `zeus` and build the power monitor, following [`docs/install_and_build.md`](../../docs/install_and_build.md).
1. Install python dependencies for this example:
    ```sh
    pip install -r requirements.txt
    ```

### Example command

`ZeusDataLoader` interfaces with the outside world via environment variables.
Check out its [class docstring](/zeus/run/dataloader.py) for details.

Only `ZEUS_TARGET_METRIC` is required; other environment variables below show their default values when omitted.

```bash
export ZEUS_TARGET_METRIC="0.50"         # Stop training when target val metric is reached
export ZEUS_LOG_DIR="zeus_log"           # Directory to store profiling logs
export ZEUS_JOB_ID="zeus"                # Used to distinguish recurrences, so not important
export ZEUS_COST_THRESH="inf"            # Kill training when cost (Equation 2) exceeds this
export ZEUS_ETA_KNOB="0.5"               # Knob to tradeoff energy and time (Equation 2)
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor" # Path to power monitor
export ZEUS_PROFILE_PARAMS="1.0,4.0"     # warmup_secs,profile_secs for each power limit
export ZEUS_USE_OPTIMAL_PL="True"        # Whether to acutally use the optimal PL found

python train.py \
    --zeus \
    --arch shufflenetv2 \
    --epochs 100 \
    --batch_size 128
```


## Running Zeus over multiple recurrences

This example shows how to integrate `ZeusDataLoader` ([source](/zeus/run/dataloader.py)) and drive batch size and power optimizations with `ZeusMaster` ([source](/zeus/run/master.py)).

### Dependencies

1. Install `zeus` and build the power monitor, following [`docs/install_and_build.md`](../../docs/install_and_build.md).
2. Only for those not using our Docker image, install `torchvision` separately:
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


## Profiling power and time

You can use Zeus's `ProfileDataLoader` ([source](/zeus/profile/torch.py)) to profile the power and time consumption of training.

### Dependencies

1. Install `zeus` and build the power monitor, following [`docs/install_and_build.md`](../../docs/install_and_build.md).
2. Only for those not using our Docker image, install `torchvision` separately:
    ```sh
    conda install -c pytorch torchvision==0.11.2
    ```

### Example command

`ProfileDataLoader` interfaces with the outside world via environment variables.
Check out its [class docstring](/zeus/profile/torch.py) for details.

Only `ZEUS_LOG_PREFIX` is required; other environment variables below show their default values when omitted.

```bash
export ZEUS_LOG_PREFIX="cifar100+shufflenetv2"  # Filename prefix for power and time log files
export ZEUS_MONITOR_SLEEP_MS="100"              # Milliseconds to sleep after sampling power
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor"  # Path to power monitor

python train.py \
    --profile \
    --arch shufflenetv2 \
    --epochs 2 \
    --batch_size 1024
```

A CSV file of timestamped momentary power draw of the first GPU (index 0) will be written to `cifar100+shufflenetv2+gpu0.power.csv` (the `+gpu0.power.csv` part was added by `ProfileDataLoader`).
At the same time, a CSV file with headers epoch number, split (`train` or `eval`), and time consumption in seconds will be written to `cifar100+shufflenetv2.time.csv` (the `.time.csv` part was added by `ProfileDataLoader`).


## Just training a vision model on CIFAR100

`train.py` can also be used as a simple training script, without having to do anything with Zeus.

### Dependencies

Only for those not using our Docker image, install PyTorch, `torchvision`, and `cudatoolkit` separately:
```sh
conda install -c pytorch pytorch==1.10.1 torchvision==0.11.2 cudatoolkit==11.3.1
```

### Example command

```sh
python train.py \
    --arch shufflenetv2 \
    --epochs 100 \
    --batch_size 1024
```
