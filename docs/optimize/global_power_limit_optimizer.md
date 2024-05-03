### `GlobalPowerLimitOptimizer`

Integrate [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer] into your training script.

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

!!! Important "What is the *optimal* power limit?"
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
