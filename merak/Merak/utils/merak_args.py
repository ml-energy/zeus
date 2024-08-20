# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import torch
from transformers import TrainingArguments
from transformers.file_utils import (
    cached_property,
    torch_required,
)


_GLOBAL_ARGS = None

def manual_set_args(args):
    global _GLOBAL_ARGS
    # some args for megatron
    if hasattr(args,'fp16') and args.fp16:
        args.params_dtype = torch.half
    else:
        args.params_dtype = torch.float32
    args.use_cpu_initialization = False
    _GLOBAL_ARGS = args


def get_args():
    """Return arguments."""
    assert _GLOBAL_ARGS is not None, '{} is not initialized.'.format('args')
    return _GLOBAL_ARGS


args_dict = {
    "seq_length": ['seq_length', 'max_position_embeddings', 'n_positions'],
    "num_heads": ['num_attention_heads', 'n_head', 'num_heads'],
    "hidden_size": ['hidden_size', 'dim', 'n_embd', 'd_model'],
    "num_layers": ['num_hidden_layers', 'n_layers'],
}


def mergeargs(training_args, model_config):
    training_args.DDP_impl = 'local'
    if training_args.input_names == []:
        training_args.input_names = None

    if training_args.seq_length is None:
        for seq in args_dict['seq_length']:
            if hasattr(model_config, seq):
                training_args.seq_length = getattr(model_config, seq)
                
    if training_args.num_layers is None:
        for layers in args_dict['num_layers']:
            if hasattr(model_config, layers):
                training_args.num_layers = getattr(model_config, layers)
    assert training_args.num_layers is not None, 'num_layers should be set.'

    if training_args.shard_count is None:
        training_args.shard_count = training_args.num_layers*2 + 4

    if training_args.train_schedule == 'shifted_critical_path':
        training_args.train_schedule = 'full_critical_path_1f1b'

    if training_args.activation_checkpointing == False:
        training_args.checkpoint_num_layers = 0

    if torch.distributed.get_rank()==0 and training_args.wall_clock_breakdown:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(training_args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(training_args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)
    
    

@dataclass
class MerakArguments(TrainingArguments):
    """
    MerakArguments inherits from transformers.TrainingArguments (https://huggingface.co/docs/transformers/master/en/main_classes/trainer#transformers.TrainingArguments) 
    extending the arguments we need in 3D parallelism.
    Using [`HfArgumentParser`] we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) 
    arguments that can be specified on the command line.
    Parameters:
    -   train_schedule (str, Optional,  defaults to '1f1b') -- Some possible choices are the pipe schedules as strings: '1f1b', 'ds_default', 'early_recompute_1f1b', 'last_no_recompute_1f1b', 'shifted_critical_path', 'instruction_profile', 'envpipe'.
    -   partition_method (str, Optional, defaults to 'uniform_transformer') -- Possible choices are the pipeline layer partion strategy as strings: 
        'uniform','uniform_floor','parameters','uniform_transformer','custom:','autopipe'.
    -   init_method_std (float, defaults to 0.02) -- Standard deviation of the zero mean normal distribution used for tp weight initialization in Megatron
    -   activation_checkpointing (bool, defaults to True) -- Whether to use activation checkpointing. 
    -   checkpoint_num_layers (int, defaults to 1) -- Chunk size (number of layers) for checkpointing.
    -   input_names (List[str], Optional, defaults to None) -- The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead. 
        Example: ['input_ids', 'attention_mask', 'token_type_ids']
    -   num_layers (int, Optional, defaults to None) -- Number of hidden layers in the Transformer, will try to get this in model config.
    -   seq_length (int, Optional, defaults to None) -- The maximum sequence length that this model might ever be used with, will try to get this in model config.
    -   wall_clock_breakdown (bool, defaults to False) -- Whether to log detail time spend on each rank.
    -   shard_count (int, Optional, defaults to None) -- Number of shards that model needs to be break, will be training_args.num_layers*2 if not set.
    -   prescale_gradients (bool, defaults to False) -- Whether to enable gradient prescaling.
    -   gradient_predivide_factor (float, defaults to 1.0) -- Gradient predivide factor in gradient prescaling.
    -   save (bool, defaults to False) -- Whether to save checkpoint.
    -   finetune (bool, defaults to False) -- Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
    -   no_save_rng (bool, defaults to False) -- Do not save current rng state.
    -   no_save_optim (bool, defaults to False) -- Do not save current optimizer.
    -   no_load_rng (bool, defaults to False) -- Do not load current optimizer.
    -   no_load_optim (bool, defaults to False) -- Do not load current optimizer.
    -   split_inputs (bool, defaults to False) -- Whether to split input data.
    -   activation_checkpoint_ratio (float, Optional, defaults to None) -- activation checkpoint ratio of first stage, in range(0,1). Default to None.
    -   tp_overlapping_level (float, Optional, defaults to 0) -- "Possible tensor parallelism communication overlapping level from 0 to 3."
                                                                 "0 refers to no overlapping; 1 refers to only overlap within linear function;"
                                                                 "2 refers to overlap within transformer blocks, requires rewrite transformer blocks;"
                                                                 "3 refers to overlap between transformer blocks, requires rewrite transformer model.",
                                                                 "choices": [0,1,2,3]
    -   loss_scale (float, defaults to 0) -- 'loss_scale is a fp16 parameter representing the loss scaling value for FP16 training.'
                                             'The default value of 0.0 results in dynamic loss scaling, '
                                             'otherwise the value will be used for static fixed loss scaling.'
    -   initial_scale_power (int, defaults to 32) -- 'initial_scale_power is a fp16 parameter representing the power of the initial dynamic loss scale value.'
                                                     'The actual loss scale is computed as 2^initial_scale_power.'
    -   loss_scale_window (int, defaults to 1000) -- 'loss_scale_window is a fp16 parameter representing the window over which to raise/lower the dynamic loss scale value.'
    -   hysteresis (int, defaults to 2) -- 'hysteresis is a fp16 parameter representing the delay shift in dynamic loss scaling.'
    -   min_loss_scale (int, defaults to 1) -- 'min_loss_scale is a fp16 parameter representing the minimum dynamic loss scale value.'

    -   export_pytorch_trace (bool, defaults to False) -- 'Whether to export PyTorch profiler results to json trace format.'
    -   export_timing_csv (bool, defaults to False) -- 'Whether to export the start and end timings of forward and backward instructions.'
    -   energy_breakdown (bool, defaults to False) -- 'Whether to record energy breakdown.'
    -   perseus_url (str, defaults to 'http://127.0.0.1:7787') -- 'Perseus server URL.'
    -   model_name (str, defaults to '') -- 'Name of the DNN being trained.'
    -   num_swaps (int, defaults to 0) -- 'How many swaps to do in the ManualSwapTrainSchedule'
    -   num_warmup_steps (int, defaults to 5) -- 'How many warmup steps to run in profiling mode'
    -   num_prof_steps (int, defaults to 30) -- 'How many steps to measure'
    -   num_transformers (int, defaults to 0) -- 'The number of transformer encoder/decoder layers in the model'
    -   num_initial_embeddings (int, default to 0) -- 'The number of initial embedding layers before transformer layers'
    -   profile (int, defaults to False) -- 'Whether the engine should run in instruction profiling mode'
    -   full_profiling (int, defaults to False) -- 'Whether profiling mode should profile all frequencies'
    -   envpipe_reschedule_cnt (list[int], defaults to []) -- 'The number of forward computation reschedules for each stage in the pipeline'
    """

    train_schedule: str = field(
        default="1f1b",
        metadata={
            "help": "Possible choices are the pipe schedules as strings: `1f1b`, `ds_default`, `early_recompute_1f1b`, "
                    "`ds_default`,  `last_no_recompute_1f1b`, `full_critical_path_1f1b`, `shifted_critical_path`, `instruction_profiler`, `envpipe`"
                    " Defaults to `1f1b`.",
        },
    )
    partition_method: str = field(
        default="unspecified",
        metadata={
            "help": "Possible choices are the pipeline layer partion strategy as strings: 'uniform','uniform_floor',"
                    "'parameters','autopipe','uniform_transformer'. Defaults to 'uniform_transformer'.",
        },
    )
    init_method_std: float = field(
        default=0.02, 
        metadata={"help": 'Standard deviation of the zero mean normal distribution used for TP weight initialization in Megatron. Defaults to 0.02'}
    )
    activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use activation checkpointing. Defaults to True"},
    )
    checkpoint_num_layers: int = field(
        default=1, 
        metadata={"help": "chunk size (number of layers) for checkpointing. 0 means disable activation checkpoint. Defaults to 1"}
    )
    input_names: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead."
                        "Example: ['input_ids', 'attention_mask', 'token_type_ids']"}
    )
    num_layers: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of hidden layers in the Transformer, will try to get or eval this in model config. Defaults to None."}
    )
    seq_length: Optional[int] = field(
        default=None, 
        metadata={"help": "The maximum sequence length that this model might ever be used with, will try to get this in model config. Defaults to None."}
    )
    wall_clock_breakdown: bool = field(
        default=True,
        metadata={"help": "Whether to record detail time spend on each rank. Defaults to True"},
    )
    print_wall_clock_breakdown: bool = field(
        default=False,
        metadata={"help": "Whether to print out detail time spend on each rank. Defaults to False"},
    )
    shard_count: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of shards that model needs to be break. Defaults to None,  will be training_args.num_layers*2 if not set."}
    )
    prescale_gradients: bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient prescaling. Defaults to False"},
    )
    gradient_predivide_factor: float = field(
        default=1.0, 
        metadata={"help": 'Gradient predivide factor in gradient prescaling. Defaults to 1'}
    )
    cache_sharding: bool = field(
        default=True,
        metadata={"help": "Whether to cache the partitioned graphs of model with microbatch size. Defaults to False"},
    )
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Set the cache name of partitioned graphs, must be setted when cache_sharding is True."}
    )
    return_logits: bool = field(
        default=False,
        metadata={"help": "Whether to return logits and labels in evaluation. Defaults to False"},
    )

    # checkpoint arguments
    save: bool = field(
        default=False,
        metadata={"help": "Whether to save checkpoints."}
    )
    finetune: bool = field(
        default=False,
        metadata={"help": "Load model for finetuning. Do not load optimizer "
                          "or rng state from checkpoint and set iteration to 0."}
    )
    no_save_rng: bool = field(
        default=False,
        metadata={"help": "Do not save current rng state"}
    )
    no_save_optim: bool = field(
        default=False,
        metadata={"help": "Do not save current optimizer"}
    )
    no_load_optim: bool = field(
        default=False,
        metadata={"help": "Do not load optimizer when loading checkpoint."}
    )
    no_load_rng: bool = field(
        default=False,
        metadata={"help": "Do not load rng state when loading checkpoint."}
    )

    # split input
    split_inputs: bool = field(
        default=False,
        metadata={"help": "Whether to split input data"}
    )
    parallel_vocab: bool = field(
        default=False,
        metadata={"help": "Whether to parallel vocabulary when TMP > 1"}
    )

    activation_checkpoint_ratio: Optional[List[str]] = field(
        default=None, 
        metadata={"help": 'activation checkpoint ratio of first stage, in range(0,1) for each pipeline stage. Default to None'}
    )
    tp_overlapping_level: int = field(
        default=0,
        metadata={
            "help": "Possible tensor parallelism communication overlapping level from 0 to 3."
                    "0 refers to no overlapping; 1 refers to only overlap within linear function;"
                    "2 refers to overlap within transformer blocks, requires rewrite transformer blocks;"
                    "3 refers to overlap between transformer blocks, requires rewrite transformer model.",
            "choices": [0,1,2,3],
        },
    )
    loss_scale: float = field(
        default=0.,
        metadata={"help": 'loss_scale is a fp16 parameter representing the loss scaling value for FP16 training.'
                          'The default value of 0.0 results in dynamic loss scaling, '
                          'otherwise the value will be used for static fixed loss scaling.'
        }
    )
    initial_scale_power: int = field(
        default=32,
        metadata={"help": 'initial_scale_power is a fp16 parameter representing the power of the initial dynamic loss scale value.'
                          'The actual loss scale is computed as 2^initial_scale_power.'
        }
    )
    loss_scale_window: int = field(
        default=1000,
        metadata={"help": 'loss_scale_window is a fp16 parameter representing the window over which to raise/lower the dynamic loss scale value.'}
    )
    hysteresis: int = field(
        default=2,
        metadata={"help": 'hysteresis is a fp16 parameter representing the delay shift in dynamic loss scaling.'}
    )
    min_loss_scale: int = field(
        default=1,
        metadata={"help": 'min_loss_scale is a fp16 parameter representing the minimum dynamic loss scale value.'}
    )
    export_pytorch_trace: bool = field(
        default=False,
        metadata={"help": 'Whether to export PyTorch profiler results to json trace format.'}
    )
    export_timing_csv: bool = field(
        default=False,
        metadata={"help": 'Whether to export the start and end timings of forward and backward instructions.'}
    )
    energy_breakdown: bool = field(
        default=False,
        metadata={"help": 'Whether to record energy breakdown.'}
    )
    perseus_url: str = field(
        default="http://localhost:7787",
        metadata={"help": 'Perseus server URL'}
    )
    model_name: str = field(
        default="",
        metadata={"help": "Name of the DNN being trained."}
    )
    num_swaps: int = field(
        default=0,
        metadata={"help": "How many swaps to do in the ManualSwapTrainSchedule"}
    )
    num_warmup_steps: int = field(
        default=5,
        metadata={"help": "How many warmup steps to run in profiling mode"}
    )
    num_prof_steps: int = field(
        default=30,
        metadata={"help": "How many steps to measure."}
    )
    num_transformers: int = field(
        default=0,
        metadata={"help": "The number of transformer encoder/decoder layers in the model"}
    )
    num_initial_embeddings: int = field(
        default=0,
        metadata={"help": "The number of initial embedding layers before transformer layers"}
    )
    profile: bool = field(
        default=False,
        metadata={"help": "Whether the engine should run in instruction profiling mode"}
    )
    full_profiling: bool = field(
        default=False,
        metadata={"help": "Whether profiling mode should profile all frequencies"}
    )
    envpipe_reschedule_cnt: Optional[list[int]] = field(
        default=None,
        metadata={"help": "The number of forward computation reschedules for each stage in the pipeline"}
    )


    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        
        assert not self.no_cuda or self.local_rank == -1, 'only support cuda device and distributed training for now'
        # Here, we'll use torch.distributed.
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", self.local_rank)
        self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(self.local_rank)

        return device

