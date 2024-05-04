# Power Limit Optimizer

The power limit optimizer ([`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]) finds the optimal GPU power limit for DNN training.
Users can customize the power limit optimizer to choose the optimal power limit based on their own criteria using the [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector] interface.

## Usage

Use cases currently supported are single GPU training and data parallel training.
For data parallel training, the power limit of all GPUs involved are changed together, since all GPUs have the same computation load.

!!! Tip "Upcoming"
    Distributed data parallel training support is planned ([tracking issue](https://github.com/ml-energy/zeus/issues/43){.external}).

!!! Important "Extra system privileges needed"
    In order to optimize the GPU power limit, the power limit optimizer should be able to change the power limit.
    This requires extra system privileges. See [here](../getting_started/index.md#system-privileges) for details.


## `GlobalPowerLimitOptimizer`

You can use the power limit optimizer by integrating [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] into your training loop.
In order to inform the optimizer of epoch and training step boundaries, a couple methods need to be called inside the training loop (highlighted):

```python hl_lines="9 12 14 16"
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import GlobalPowerLimitOptimizer

# Data parallel training with four GPUs.
monitor = ZeusMonitor(gpu_indices=[0,1,2,3])
plo = GlobalPowerLimitOptimizer(monitor)

for epoch in range(100):
    plo.on_epoch_begin()

    for x, y in train_dataloader:
        plo.on_step_begin()
        # Learn from x and y
        plo.on_step_end()

    plo.on_epoch_end()
```

We provide [integration examples](https://github.com/ml-energy/zeus/tree/master/examples/power_limit_optimizer/){.external} for Torchvision & ImageNet single-GPU and data parallel training.

!!! Tip "What is the *optimal* power limit?"
    [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] accepts an optional [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector] in its constructor, which defines how to choose one power limit among all the profiled power limits.
    Built-in optimum selectors are [`Energy`][zeus.optimizer.power_limit.Energy], [`Time`][zeus.optimizer.power_limit.Time], [`ZeusCost`][zeus.optimizer.power_limit.ZeusCost] and [`MaxSlowdownConstraint`][zeus.optimizer.power_limit.MaxSlowdownConstraint].
    Users can inherit from [`OptimumSelector`][zeus.optimizer.power_limit.OptimumSelector] to implement their custom optimum selector.

## `HFGlobalPowerLimitOptimizer`

For easy use with HuggingFace Transformers, [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer] is implemented as a [HuggingFace Trainer Callback](https://huggingface.co/docs/transformers/en/main_classes/callback){.external} by inheriting from [`TrainerCallback`][transformers.TrainerCallback].
When initializing a [HuggingFace Trainer][transformers.Trainer] or a [TFL SFTTrainer][trl.SFTTrainer], initialize and pass in [`HFGlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.HFGlobalPowerLimitOptimizer] as shown below:

```python hl_lines="11"
from transformers import Trainer
from zeus.monitor import ZeusMonitor
from zeus.optimizer.power_limit import HFGlobalPowerLimitOptimizer

monitor = ZeusMonitor()
plo = HFGlobalPowerLimitOptimizer(monitor)

# Also works with trl.SFTTrainer.
trainer = Trainer(
    ...,
    callbacks=[plo],
)
```

Refer to our [HuggingFace integration examples](https://github.com/ml-energy/zeus/tree/master/examples/huggingface/){.external} for:

- Transformers `Trainer` integration for **causal language modeling** (i.e., pre-training)
- TRL `SFTTrainer` integration for **Gemma 7B supervised fine-tuning with QLoRA**
