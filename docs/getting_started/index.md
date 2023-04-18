# Getting Started

Zeus automatically tunes the **batch size** and **GPU power limit** of a recurring DNN training job.

!!! Important
    Zeus can optimize the batch size of **recurring** jobs, i.e. training jobs that re-run multiple times over time. However, Zeus can still optimize the GPU power limit even if your jobs does not recur.

!!! Info
    Zeus currently supports **single GPU training** and **single node data parallel training**. Support for distributed data parallel training will be added soon.

## How it works

Zeus in action, integrated with Stable Diffusion fine-tuning:
<iframe width="560" height="315" src="https://www.youtube.com/embed/MzlF5XNRSJY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

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

!!! Important
    Zeus assumes that exactly one process manages one GPU, and hence
    one instance of [`ZeusDataLoader`][zeus.run.ZeusDataLoader] exists
    for each GPU.

Users can integrate Zeus into existing data parallel training scripts
with five specific steps, which are noted below in the comments.

Please refer to
[our integration example with ImageNet](https://github.com/SymbioticLab/Zeus/tree/master/examples/imagenet/train.py)
for a complete example.

```python
import torch
import torch.distributed as dist
import torchvision

from zeus.run import ZeusDataLoader

# Step 1: Initialize the default process group.
# This should be done before instantiating `ZeusDataLoader`.
dist.init_process_group(
    backend=args.dist_backend,
    init_method=args.dist_url,
)

# Step 2: Create a model and wrap it with `DistributedDataParallel`.
model = torchvision.models.resnet18()
torch.cuda.set_device(local_rank)
model.cuda(local_rank)
# Zeus assumes that exactly one process manages one GPU. If you are doing data
# parallel training, please use `DistributedDataParallel` for model replication
# and specify the `device_ids` and `output_device` as below:
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
)

# Step 3: Create instances of `DistributedSampler` to partition the dataset
# across the GPUs.
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set)

# Step 4: Instantiate `ZeusDataLoader`.
# `distributed="dp"` tells `ZeusDataLoader` to operate in data parallel mode.
# The one instantiated with `max_epochs` becomes the train dataloader.
train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100, 
                              sampler=train_sampler, distributed="dp")
eval_loader = ZeusDataLoader(eval_set, batch_size=256, sampler=eval_sampler,
                             distributed="dp")

# Step 5: Training loop.
# Use the train dataloader's `epochs` generator to allow Zeus to early-stop
# based on the training cost. Use `report_metric` to let Zeus know the current
# validation metric.
for epoch_number in train_loader.epochs():
    for batch in train_loader:
        # Learn from batch
    for batch in eval_loader:
        # Evaluate on batch

    # Make sure you all-reduce the validation metric across all GPUs,
    # since Zeus expects the final validation metric.
    val_metric_tensor = torch.tensor([validation_metric], device="cuda")
    dist.all_reduce(val_metric_tensor, async_op=False)
    train_loader.report_metric(val_metric_tensor.item())
```

The following examples will help:

- Integrating Zeus with computer vision
    - [Integrating Zeus with CIFAR100 dataset](https://github.com/SymbioticLab/Zeus/tree/master/examples/cifar100){.external}
    - [Integrating Zeus with ImageNet dataset](https://github.com/SymbioticLab/Zeus/tree/master/examples/imagenet){.external}
- [Integrating Zeus with NLP](https://github.com/SymbioticLab/Zeus/tree/master/examples/capriccio){.external}


## Recurring jobs

!!! Info
    We plan to integrate [`ZeusMaster`][zeus.run.ZeusMaster] with an MLOps platform like [KubeFlow](https://www.kubeflow.org/).
    Let us know about your preferences, use cases, and expectations by [posting an issue](https://github.com/SymbioticLab/Zeus/issues/new?assignees=&labels=&template=feature_request.md&title=Regarding%20Integration%20with%20MLOps%20Platroms)!

The cost-optimal batch size is located *across* multiple job runs using a Multi-Armed Bandit algorithm.
First, go through the steps for non-recurring jobs. 
[`ZeusDataLoader`][zeus.run.ZeusDataLoader] will transparently optimize the GPU power limit for any given batch size.
Then, you can use [`ZeusMaster`][zeus.run.ZeusMaster] to drive recurring jobs and batch size optimization.

This example will come in handy:

- [Running trace-driven simulation on single recurring jobs and the Alibaba GPU cluster trace](https://github.com/SymbioticLab/Zeus/tree/master/examples/trace_driven){.external}
