"""The server guides the `PipelineFrequencyOptimizer` with frequency plans.

The server is agnostic to the training framework the
[`PipelineFrequencyOptimizer`][zeus.optimizer.pipeline_frequency.optimizer.PipelineFrequencyOptimizer]
is integrated with. A server is useful because large model training is
typically distributed, and we still need one place to coordinate the
frequency plans. Later, the server will be extended to support complete
online profiling and optimization.
"""
