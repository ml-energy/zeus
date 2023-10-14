# Integrating Perseus with Training Frameworks

!!! Warning
    Perseus is under active development, and breaking changes may happen.
    Currently, we have all the low-level APIs in place, but it's not a turnkey solution yet.
    This document always reflects the master `HEAD`.

This page aims to walk you through the process of integrating Perseus with arbitrary training frameworks.
We also have a reference integration with [Merak](https://github.com/ml-energy/merak-zeus).
Especially take a look at `Merak.runtime.pipe_engine`.

## Assumptions

We assume that there are concrete regions of the framework's code where the forward pass and the backward pass exclusively happens.
For instance, in DeepSpeed, `PipelineEngine` has [`_exec_forward_pass`](https://github.com/microsoft/DeepSpeed/blob/4fc181b01077521ba42379013ce91a1c294e5d8e/deepspeed/runtime/pipe/engine.py#L626) and [`_exec_backward_pass`](https://github.com/microsoft/DeepSpeed/blob/4fc181b01077521ba42379013ce91a1c294e5d8e/deepspeed/runtime/pipe/engine.py#L703).
As another example, in Megatron-LM, users can pass in their custom `forward_step_func` to `pretrain`, and [`forward_step`](https://github.com/NVIDIA/Megatron-LM/blob/79a9feef261352ac1ee80b36f2cf73c20f864965/megatron/core/pipeline_parallel/schedules.py#L149) in the codebase calls it. The backward pass is done (roughly) in the [`backward_step`](https://github.com/NVIDIA/Megatron-LM/blob/79a9feef261352ac1ee80b36f2cf73c20f864965/megatron/core/pipeline_parallel/schedules.py#L216) function.

## Integrate `PerseusOptimizer`

1. Instantiate the [`PerseusOptimizer`][zeus.optimizer.perseus.optimizer.PerseusOptimizer] somewhere before actual training runs. Let's call the object `opt`.
1. Surround one training step with `opt.on_step_begin()` and `opt.on_step_end()`.
1. Wrap the forward pass region with `opt.on_instruction_begin("forward")` and `opt.on_instruction_end("forward")`.
1. Wrap the backward pass region with `opt.on_instruction_begin("backward")` and `opt.on_instruction_end("backward")`.

That's it.

## Profiling Instructions

It's important to optimize on top of accurate measurements of forward and backward instructions.
For now, we're taking an offline approach, where we run each instruction under a given GPU frequency N times and average time and energy consumption.
See [Merak's `profile` function](https://github.com/ml-energy/merak-zeus/blob/40eb07f80b3b3c2905bde303b02a6f707193f083/Merak/merak_trainer.py#L620).

We're on the process of implementing an online approach that is directly integrated into `PerseusOptimizer` so that you don't need to implement a separate profiler inside your framework.
