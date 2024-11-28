# Optimizing Energy

!!! Important
    You will need to set up your environment to make energy measurement and optimization work. Please refer to the [Getting Started](../getting_started/index.md) guide.

Zeus provides multiple optimizers that tune different knobs either in the Deep Learning workload-side or the GPU-side.

## [Power limit optimizer](power_limit_optimizer.md)

Finds the optimal GPU power limit for DNN training.
Users can control what *optimal* means, including minimum energy, minimum energy *given* maximum training slowdown, and minimum *cost* (linear combination of time and energy).

## [Batch size optimizer](batch_size_optimizer.md)

Finds the optimal DNN training batch size for training jobs that recur over time.
This would be especially useful for production training jobs where the underlying dataset is constantly updated, and the model is periodically re-trained to keep it up-to-date.

## [Pipeline frequency optimizer](pipeline_frequency_optimizer.md)

In large model training (e.g., pre-training Large Language Models), pipeline parallelism is almost essential today.
The pipeline frequency optimizer plans the GPU SM frequency across time for an iteration of pipeline parallel training.
It generates a set of frequency plans, including a plan that reduces energy the most, another that reduces energy with *negligible* slowdown, and plans in the middle.
