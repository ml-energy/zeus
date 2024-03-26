# Batch Size Optimizer in Zeus

## What is it?

Batch size optimizer(BSO) can choose the best batch size that minimizes the cost, where cost is defined as $cost = \eta \times \text{Energy consumption to accuracy} + (1-\eta) \times \text{Max power}\times \text{Time to accuracy}$.

## How does it work?

Core of BSO is a multi-arm-bandit based on **recurrent** trainings. After each training, we feed the result cost to MAB and after certain number of trainings, MAB can converge to the best batch size. In addition to MAB, we employed early-stopping and pruning to handle stragglers. For more detail, refer to [paper](https://www.usenix.org/conference/nsdi23/presentation/you).

## Should I use this?

The key of BSO is recurrent training. If you are training your model periodically or repeatedly, BSO can be a great choice to reduce energy or time consumption.

## Limitations

We currently doesn't support heterogeneous GPUs or different configurations. Number of GPUs, gpu models, and other configurations in JobSpec should be identical in recurrent trainings. If you are running your training in a various environment each time, then it might not desirable to use BSO.
