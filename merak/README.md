<!---
Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.

Maintainer: TXacs (txacs1993@gmail.com), Swli (lucasleesw9@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

This is a fork of Merak that has been integrated with the Perseus client.
Find below the original README:

# Merak: fast and easy to use 3D parallelism DNN training framework

[Merak](https://arxiv.org/abs/2206.04959) is a distributed deep learning training framework with automated 3D parallelism. It can automatically slice, allocate and train a DNN model, making the development of giant model fast and easy. The current version of Merak is adapted to PyTorch.

## Motivation of Merak

With the rapidly growing size of DNN models, exquisite distributed training solutions for giant models are required. However, using the SOTA technology of giant model pretraining: 3D parallelism (data parallelism, tensor model parallelism, pipeline model parallelism) needs much experiences and model rewriting.

The motivation of Merak is to simplify the usage of 3D parallelism and ensure that users only need to add as little code as the popular training tool [Huggingface transformers trainer](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#trainer) to achieve complicated 3D parallelism.




## Installation

To install Merak:

```bash
# ensure PyTorch >= 1.10 installed since it requires extra index url
# (check https://pytorch.org/get-started/locally/)
# ensure pybind11 installed
git clone http://hpdl-group/Merak.git
cd Merak
pip install .
```


## How to use

To use Merak, make the following modifications to your programs:

1. Import Merak before import transformers and torch
2. Set degrees of the data parallel, tensor model parallel and pipeline model parallel; and run `Merak.init(dp, tp, pp)` to initialize Merak.
3. Set training arguments `MerakArguments`. Replacement of [transformers trainer arguments](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#transformers.TrainingArguments)
4. Set `MerakTrainer`. Replacement of [transformers trainer](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#trainer).

Example usage (see the Merak [examples](https://github.com/HPDL-Group/Merak/tree/main/examples) directory for full training examples):

```Python
import Merak
from Merak import MerakArguments, MerakTrainer

# Init Merak with degrees of 3D parallelism.
dp = 2
tp = 1
pp = 2
Merak.init(dp, tp, pp)

# Set training args MerakArgument.
training_args = MerakArguments(
	output_dir='./path/to/save'
)

# Set our Trainer
trainer = MerakTrainer(
     do_train=...,
     model=...,
     args=training_args,
     train_data=...,
     eval_data=...,
)

# Do train
trainer.train()
```

For more details you could refer to our api [document](https://github.com/HPDL-Group/Merak/blob/main/docs/api_doc.md).
For more detail usage, please check [transformers](https://github.com/huggingface/transformers/tree/v4.15.0/) tutorials and its trainer [examples](https://github.com/huggingface/transformers/tree/v4.15.0/examples/pytorch).


## Merak Features


### Automatic 3D parallel training
In pipeline model parallelism of Merak, we uses `torch.fx` and `transformers.utils.fx` to trace a model into `GraphModule`. Then we come up with a graph sharding algorithm to split traced graph evenly into a sequence of `GraphModules`. For example, in the GPT model, each attention block and mlp block will be an individual module. Next a high performance runtime engine would allocate the module sequence and execute the training procedures.

As for tensor model parallelism, we use a feature dict to map the parameters into `ColumnParallelLinear` and `RowParallelLinear` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/mpu/layers.py). We hold default feature dicts for common models in `transformers` package. In addition, users could define a feature dict through our API easily to achieve the tensor model parallelism.

-   Using as easy as single GPU training.

For giant models in `transformers`: our implementation is based on `transformers.trainer` class. With a few lines of code setting of parallel degrees, training model with 3D parallelism could be as easy as single GPU training.
For models not in  `transformers`: as long as a model is traceable by `torch.fx` and trainable by `transformers.trainer`, the model could trained by Merak as well.

-   Sharding a giant model in a single worker.

Training, even only loading, a DNN model on a single GPU device could easily exceed the device's memory capacity nowadays. Before the model being initialized in memory, we create proxy layers for `torch.nn.Linear` layers. Proxy layers do not own parameters but could participate in model tracing and graph sharding normally. This make it possible that a single worker could store a whole giant DNN model and execute the graph sharding swiftly.

-   Auto dataloader for 3D parallelism.

When we train a model with pipeline parallelism, different stages require different data, some stages even do not load data. So we try to make the different stages only get their needed data, without loading the full dataset.


###   High-performance training

To further boost the training performance, our efficient3D parallel runtime engine proposes some novel technologies to achieve better integration of training resources.
- Shifted critical path schedule.

We introduce a shifted critical path pipeline schedule for reducing pipeline bubbles. Critical path is an operation sequence that determines the pipeline latency. Our schedule shortens the critical path by dropping redundant recomputation and adjusting orders and start time of operations.

- Stage-aware recomputation.

In addition, we observe that a more efficient memory utilization can be obtained by adopting the activation recomputation in a fine-grained way. Hence we develop a stage-aware recomputation method to exploit the usage of worker memory, which employs idle memory for less recomputation according to pipeline stage rank and pipeline depth, and thereby speedup training.

- Sub-pipelined TMP.

Furthermore, we improve the concurrency of the communication and computation in TMP with our sub-pipelined TMP approach, which applies microbatch splitting for individual sub-microbatches, and thereby pipelines sub-microbatches to overlap the communication and computation in TMP.

Please refer to our [paper](https://arxiv.org/abs/2206.04959) for more technical details and experiment results.



## References

The Merak source code was based off the  [transformers trainer](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#trainer), [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repository.

## Cite Us
```
@article{lai2022merak,
  title={Merak: An Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models},
  author = {Lai, Zhiquan and Li, Shengwei and Tang, Xudong and Ge, Keshi and Liu, Weijie and Duan, Yabo and Qiao, Linbo and Li, Dongsheng},
  journal={arXiv preprint arxiv:2206.04959},
  year={2022}
}
```
