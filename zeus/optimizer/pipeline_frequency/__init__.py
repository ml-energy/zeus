"""Optimize the energy consumption of large model training with Perseus.

A a high-level, this optimizer assigns each forward and backward computation
in a pipeline parallel training iteration with a GPU frequency that leads to
a Pareto-optimal training iteration time and energy consumption.

Currently, this optimizer depends on PyTorch.
"""

from zeus.optimizer.pipeline_frequency.optimizer import PipelineFrequencyOptimizer
