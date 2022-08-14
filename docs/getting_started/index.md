# Getting Started

Zeus automatically tunes the **batch size** and **GPU power limit** of a recurring DNN training job.

!!! Important
    Zeus can optimize the batch size of **recurring** jobs, i.e. training jobs that re-run multiple times over time. However, Zeus can still optimize the GPU power limit even if your jobs does not recur.

!!! Info
    Zeus currently only supports **single GPU training**. Support for single-node data parallel training and distributed data parallel training will be added soon.

## Prerequisites

1. [Set up your environment](environment.md).
2. [Install and build Zeus components](installing_and_building.md).


## Non-recurring jobs

The GPU power limit can be profiled and optimized quickly for any training job.
After going through the prerequisites, integrate [`ZeusDataLoader`][zeus.run.ZeusDataLoader] into your training script.

The following examples will help:

- [Integrating Zeus with computer vision](https://github.com/SymbioticLab/Zeus/tree/master/examples/cifar100){.external}
- [Integrating Zeus with NLP](https://github.com/SymbioticLab/Zeus/tree/master/examples/capriccio){.external}


## Recurring jobs

The optimal batch size is explored *across* multiple job runs using a Multi-Armed Bandit algorithm.
First, go through the steps for non-recurring jobs. 
[`ZeusDataLoader`][zeus.run.ZeusDataLoader] will transparently optimize the GPU power limit for any given batch size.
Then, you can use [`ZeusMaster`][zeus.run.ZeusMaster] to drive recurring jobs and batch size optimization.

This example will come in handy:

- [Running trace-driven simulation on single recurring jobs and the Alibaba GPU cluster trace](https://github.com/SymbioticLab/Zeus/tree/master/examples/trace_driven){.external}

We plan to integrate [`ZeusMaster`][zeus.run.ZeusMaster] with an MLOps platform like [KubeFlow](https://www.kubeflow.org/).
Feel free to let us know about your preferences, use cases, and expectations.
