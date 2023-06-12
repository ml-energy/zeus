# Integrating Zeus with Huggingface and Capriccio

This example will demonstrate how to integrate Zeus with [Capriccio](../../capriccio), a drifting sentiment analysis dataset.

You can search for `# ZEUS` in [`train.py`](train.py) for noteworthy places that require modification from conventional training scripts.
Parts relevant to using Capriccio are also marked with `# CAPRICCIO`.

**Usages**

- Zeus
    - [Running Zeus for a single job](#running-zeus-for-a-single-job)
    - [Running Zeus over multiple recurrences](#running-zeus-over-multiple-recurrences)
- Extra
    - [Fine-tuning a Huggingface language model on one slice](#fine-tuning-a-huggingface-language-model-on-one-slice)

## Running Zeus for a single job

While our paper is about optimizing the batch size and power limit over multiple recurrences of the job, it is also possible to use just [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) to JIT-profile and optimize the power limit.

### Dependencies

1. Generate Capriccio, following the instructions in [Capriccio's README.md](../../capriccio/).
1. If you're not using our [Docker image](https://ml.energy/zeus/getting_started/environment/), install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install python dependencies for this example:
    ```sh
    pip install -r requirements.txt
    ```

### Example command

[`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) interfaces with the outside world via environment variables.
Check out the [class reference](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) for details.

Only `ZEUS_TARGET_METRIC` is required; other environment variables below show their default values when omitted.

```bash
export ZEUS_TARGET_METRIC="0.84"               # Stop training when target val metric is reached
export ZEUS_LOG_DIR="zeus_log"                 # Directory to store profiling logs
export ZEUS_JOB_ID="zeus"                      # Used to distinguish recurrences, so not important
export ZEUS_COST_THRESH="inf"                  # Kill training when cost (Equation 2) exceeds this
export ZEUS_ETA_KNOB="0.5"                     # Knob to tradeoff energy and time (Equation 2)
export ZEUS_MONITOR_PATH="/workspace/zeus/zeus_monitor/zeus_monitor" # Path to power monitor
export ZEUS_PROFILE_PARAMS="10,40"              # warmup_iters,profile_iters for each power limit
export ZEUS_USE_OPTIMAL_PL="True"              # Whether to acutally use the optimal PL found

python train.py \
    --zeus \
    --data_dir data \
    --slice_number 9 \
    --model_name_or_path bert-base-uncased \
    --batch_size 128
```


## Running Zeus over multiple recurrences

This example shows how to integrate [`ZeusDataLoader`](https://ml.energy/zeus/reference/run/dataloader/#zeus.run.dataloader.ZeusDataLoader) and drive batch size and power optimizations with [`ZeusMaster`](https://ml.energy/zeus/reference/run/master/#zeus.run.master.ZeusMaster).

### Dependencies

1. Generate Capriccio, following the instructions in [Capriccio's README.md](../../capriccio/).
1. If you're not using our [Docker image](https://ml.energy/zeus/getting_started/environment/), install `zeus` and build the power monitor, following [Installing and Building](https://ml.energy/zeus/getting_started/installing_and_building/).
1. Install python dependencies for this example:
    ```sh
    pip install -r requirements.txt
    ```

### Example command

```sh
# All arguments shown below are default values.
python run_zeus.py \
    --seed 123 \
    --b_0 128 \
    --lr_0 4.00e-7 \
    --b_min 8 \
    --b_max 128 \
    --num_recurrence 38 \
    --eta_knob 0.5 \
    --beta_knob 2.0 \
    --target_metric 0.84 \
    --max_epochs 10 \
    --window_size 10
```


## Fine-tuning a Huggingface language model on one slice

`train.py` can also be used to fine-tune a pretrained language model on one slice of Capriccio, without having to do anything with Zeus.

### Dependencies

1. Generate Capriccio, following the instructions in [Capriccio's README.md](../../capriccio/).
1. Only for those not using our [Docker image](https://ml.energy/zeus/getting_started/environment/), install PyTorch separately:
    ```sh
    conda install -c pytorch pytorch==1.10.1
    ```
1. Install python dependencies for this example:
    ```sh
    pip install -r requirements.txt
    ```

### Example command

```sh
python train.py \
    --data_dir data \
    --slice_number 9 \
    --model_name_or_path bert-base-uncased \
    --batch_size 128
```
