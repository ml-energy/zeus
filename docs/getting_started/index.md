# Getting Started

Zeus is an energy measurement and optimization toolbox for deep learning.

## How it works

Zeus in action, integrated with Stable Diffusion fine-tuning:
<iframe width="560" height="315" src="https://www.youtube.com/embed/MzlF5XNRSJY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


## Just measuring GPU time and energy

### Prerequisites

If your NVIDIA GPU's architecture is Volta or newer, simply do the following in your Python environment
```sh
pip install zeus-ml
```
and get going with [`ZeusMonitor`][zeus.monitor.ZeusMonitor].

Otherwise, we recommend using our Docker container:

1. [Set up your environment](environment.md).
2. [Install Zeus](installing.md).

### `ZeusMonitor`

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] makes it very simple to measure the GPU time and energy consumption of arbitrary Python code blocks.

```python hl_lines="4 11-13"
from zeus.monitor import ZeusMonitor

# All GPUs are measured simultaneously if `gpu_indices` is not given.
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

for epoch in range(100):
    monitor.begin_window("epoch")

    measurements = []
    for x, y in train_loader:
        monitor.begin_window("step")
        train_one_step(x, y)
        result = monitor.end_window("step")
        measurements.append(result)

    result = monitor.end_window("epoch")
    print(f"Epoch {epoch} consumed {result.time} s and {result.total_energy} J.")

    avg_time = sum(map(lambda m: m.time, measurements)) / len(measurements)
    avg_energy = sum(map(lambda m: m.total_energy, measurements)) / len(measurements)
    print(f"One step took {avg_time} s and {avg_energy} J on average.")
```


## Optimizing a single training job's energy consumption

All GPU power limits can be profiled quickly *during training* and used to optimize the energy consumption of the training job.

### Prerequisites

In order to change the GPU's power limit, the process requires the Linux `SYS_ADMIN` security capability, and the easiest way to do this is to spin up a container and give it `--cap-add SYS_ADMIN`.
We provide ready-to-go [Docker images](environment.md).


### `GlobalPowerLimitOptimizer`

After going through the prerequisites, [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] into your training script.

Refer to
[our integration example with ImageNet](https://github.com/ml-energy/zeus/tree/master/examples/imagenet/)
for complete running examples for single-GPU and multi-GPU data parallel training.

```python hl_lines="10"
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import GlobalPowerLimitOptimizer

# Data parallel training with four GPUs.
# Omitting `gpu_indices` will use all GPUs, while respecting
# `CUDA_VISIBLE_DEVICES`.
monitor = ZeusMonitor(gpu_indices=[0,1,2,3])
# The power limit optimizer profiles power limits during training
# using the `ZeusMonitor` instance.
plo = GlobalPowerLimitOptimizer(monitor)

for epoch in range(100):
    plo.on_epoch_begin()

    for x, y in train_dataloader:
        plo.on_step_begin()
        # Learn from x and y!
        plo.on_step_end()

    plo.on_epoch_end()

    # Validate the model if needed, but `plo` won't care.
```

!!! Important
    What is the *optimal* power limit?
    The [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] supports multiple [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector]s that chooses one power limit among all the profiled power limits.
    Selectors that are current implemented are [`Energy`][zeus.optimizer.power_limit.Energy], [`Time`][zeus.optimizer.power_limit.Time], [`ZeusCost`][zeus.optimizer.power_limit.ZeusCost] and [`MaxSlowdownConstraint`][zeus.optimizer.power_limit.MaxSlowdownConstraint].

### `HFGlobalPowerLimitOptimizer`
For easy use with [HuggingFace 🤗 Transformers](https://huggingface.co/docs/transformers/en/index), [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer] is a drop-in compatible [HuggingFace 🤗 Trainer Callback](https://huggingface.co/docs/transformers/en/main_classes/callback). When initializing a [HuggingFace 🤗 Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) or a [TFL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer), initialize and pass in [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer] as shown below:

```python hl_lines="10"
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer

monitor = ZeusMonitor()
optimizer = HFGlobalPowerLimitOptimizer(monitor)

# Also works with SFTTrainer.
trainer = Trainer(
    ...,
    callbacks=[optimizer], # Add the `HFGlobalPowerLimitOptimizer` callback
)
```
Refer to our [HuggingFace 🤗 example integration](https://github.com/ml-energy/zeus/tree/master/examples/huggingface/) for:

- Transformers [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) integration for **causal langauge modeling** (i.e., pre-training)
- TRL [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer) integration for **Gemma 7b supervised fine-tuning with QLoRA**

## Large model training jobs

We created [Perseus](../perseus/index.md), which can optimize the energy consumption of large model training with practically no slowdown!

## Recurring jobs

In production, it's likely that a DNN is trained and re-trained repetitively to keep it up to date.
For these kinds of recurring jobs, we can take those recurrences as exploration opportunities to find the cost-optimal training batch size.
This is done with a Multi-Armed Bandit algorithm.
See [`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer].

Two full examples are given for the batch size optimizer:

- [MNIST](https://github.com/ml-energy/zeus/tree/master/examples/batch_size_optimizer/mnist/): Single-GPU and data parallel training, with integration examples with Kubeflow
- [Sentiment Analysis](https://github.com/ml-energy/zeus/tree/master/examples/batch_size_optimizer/capriccio/): Full training example with HuggingFace transformers using the Capriccio dataset, a sentiment analysis dataset with data drift.
