# Getting Started

Zeus automatically tunes the **batch size** and **GPU power limit** of a recurring DNN training job.

!!! Important
    Zeus can optimize the batch size of **recurring** jobs, i.e. training jobs that re-run multiple times over time. However, Zeus can still optimize the GPU power limit even if your jobs does not recur.

!!! Info
    Zeus currently supports **single GPU training** and **single node data parallel training**. Support for distributed data parallel training will be added soon.

## Prerequisites

1. [Set up your environment](environment.md).
2. [Install and build Zeus components](installing_and_building.md).


## Non-recurring jobs

The GPU power limit can be profiled and optimized quickly for any training job.
After going through the prerequisites, integrate [`ZeusDataLoader`][zeus.run.ZeusDataLoader] into your training script.

Integration example:

### Single-GPU

```python
from zeus.run import ZeusDataLoader

# The one instantiated with max_epochs becomes the train dataloader
train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100)
eval_loader = ZeusDataLoader(eval_set, batch_size=256)

for epoch_number in train_loader.epochs():
    for batch in train_loader:
        # Learn from batch
    for batch in eval_loader:
        # Evaluate on batch

    train_loader.report_metric(validation_metric)
```

### Data parallel with multi-GPU on a single-node
Zeus supports only one process per GPU profiling. In data parallel training,
each process has its `local_rank` within the node and will run the
following code.
We also specify the important steps for a better comprehension.
Please refer to [the integration example with ImageNet](https://github.com/SymbioticLab/Zeus/tree/master/examples/imagenet/train.py)
for the complete example.

```python
import torch
import torchvision

from zeus.run import ZeusDataLoader

# Step 1: Initialize the default process group.
dist.init_process_group(
    backend=args.dist_backend,
    init_method=args.dist_url,
)

# Step 2: Create a model and wrap it with `DistributedDataParallel`.
model = torchvision.models.resnet18()
torch.cuda.set_device(local_rank)
model.cuda(local_rank)
# Zeus only supports one process per GPU profiling. If you are doing data
# parallel training, please use `DistributedDataParallel` for model replication
# and specify the `device_ids` and `output_device` correctly.
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
)

# Step 3: Create instances of `DistributedSampler` to partition the dataset
# across the GPUs.
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set)

# Step 4: Create instances of `ZeusDataLoader`.
# Pass "dp" to `distributed` and samplers in the previous step to
# `sampler`.
# The one instantiated with `max_epochs` becomes the train dataloader.
train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100, 
                              sampler=train_sampler, distributed="dp")
eval_loader = ZeusDataLoader(eval_set, batch_size=256, sampler=eval_sampler,
                             distributed="dp")

# Step 5: Put your training code here.
for epoch_number in train_loader.epochs():
    for batch in train_loader:
        # Learn from batch
    for batch in eval_loader:
        # Evaluate on batch

    # If doing data parallel training, please make sure to call 
    # `torch.distributed.all_reduce()` to reduce the validation metric 
    # across all GPUs before calling `train_loader.report_metric()`.
    train_loader.report_metric(validation_metric)
```

The following examples will help:

- Integrating Zeus with computer vision
  - [Integrating Zeus with CIFAR100 dataset](https://github.com/SymbioticLab/Zeus/tree/master/examples/cifar100){.external}
  - [Integrating Zeus with ImageNet dataset](https://github.com/SymbioticLab/Zeus/tree/master/examples/imagenet){.external}
- [Integrating Zeus with NLP](https://github.com/SymbioticLab/Zeus/tree/master/examples/capriccio){.external}


## Recurring jobs

The optimal batch size is explored *across* multiple job runs using a Multi-Armed Bandit algorithm.
First, go through the steps for non-recurring jobs. 
[`ZeusDataLoader`][zeus.run.ZeusDataLoader] will transparently optimize the GPU power limit for any given batch size.
Then, you can use [`ZeusMaster`][zeus.run.ZeusMaster] to drive recurring jobs and batch size optimization.

This example will come in handy:

- [Running trace-driven simulation on single recurring jobs and the Alibaba GPU cluster trace](https://github.com/SymbioticLab/Zeus/tree/master/examples/trace_driven){.external}

!!! Info
    We plan to integrate [`ZeusMaster`][zeus.run.ZeusMaster] with an MLOps platform like [KubeFlow](https://www.kubeflow.org/).
    Feel free to let us know about your preferences, use cases, and expectations.
