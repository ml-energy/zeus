# Getting Started

Zeus automatically tunes the **batch size** and **GPU power limit** of a recurring DNN training job.

!!! Important
    Zeus can optimize the batch size of **recurring** jobs, i.e. training jobs that re-run multiple times over time. However, Zeus can still optimize the GPU power limit even if your jobs does not recur.

!!! Info
    Zeus currently supports **single GPU training** and **single node data parallel training**. Support for distributed data parallel training will be added soon.

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
2. [Install and build Zeus components](installing_and_building.md).

### `ZeusMonitor`

[`ZeusMonitor`][zeus.monitor.ZeusMonitor] makes it very simple to measure the GPU time and energy consumption of arbitrary Python code blocks.

```python
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

    eres = monitor.end_window("epoch")
    print(f"Epoch {epoch} consumed {eres.time} s and {eres.total_energy} J.")

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

```python
from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer

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

## Recurring jobs

!!! Info
    We plan to integrate [`ZeusMaster`][zeus.run.ZeusMaster] with an MLOps platform like [KubeFlow](https://www.kubeflow.org/).
    Let us know about your preferences, use cases, and expectations by [posting an issue](https://github.com/ml-energy/zeus/issues/new?assignees=&labels=&template=feature_request.md&title=Regarding%20Integration%20with%20MLOps%20Platroms)!

The cost-optimal batch size is located *across* multiple job runs using a Multi-Armed Bandit algorithm.
First, go through the steps for non-recurring jobs. 
[`ZeusDataLoader`][zeus.run.ZeusDataLoader] will transparently optimize the GPU power limit for any given batch size.
Then, you can use [`ZeusMaster`][zeus.run.ZeusMaster] to drive recurring jobs and batch size optimization.

This example will come in handy:

- [Running trace-driven simulation on single recurring jobs and the Alibaba GPU cluster trace](https://github.com/ml-energy/zeus/tree/master/examples/trace_driven){.external}
