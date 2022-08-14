# Getting Started

Zeus automatically tunes the **batch size** and **GPU power limit** of a recurring DNN training job.

!!! Important
    Zeus optimizes the batch size of **recurring** jobs, i.e. training jobs that re-run multiple times over time. However, Zeus can still optimize the GPU power limit even if your jobs does not recur.

!!! Info
    Zeus currently only supports **single GPU training**. Support for single-node data parallel training and distributed data parallel training will be added in the future.

## Non-recurring jobs

The GPU power limit can be profiled and optimized quickly for any training job.
You just need to [set up your environment](environment.md), [install and build Zeus components](installing_and_building.md), and [integrate Zeus into your training script](integrating.md).


## Recurring jobs

The optimal batch size is explored *across* multiple job runs using a Multi-Armed Bandit algorithm, so your job should be recurring.
