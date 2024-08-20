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

import datetime

import transformers
import torch
import torch.distributed as dist
import copy

def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


from .modules.layer_proxy import Conv1DProxy, LinearProxy

func_bak = (transformers.modeling_utils.Conv1D, torch.nn.Linear)

# monkey patch for proxy layers
transformers.modeling_utils.Conv1D = Conv1DProxy
torch.nn.Linear = LinearProxy

def get_patched_func():
    return func_bak

# # monkey patch for proxy layers
# transformers.modeling_utils.Conv1D = Conv1DProxy
# torch.nn.Linear = LinearProxy

topo = None
communication_grid = None

def init(pp, tp, dp, backend='nccl'):
    """
    Initialized the distributed communication groups, include data parallel, 
    tensor model parallel and pipeline model parallel. Each parallel degree 
    has it own communication group, we can ge the rank or size through mpu API.

    Parameters:
    -   dp (int) -- Parallel degree of data parallelism.
    -   tp (int) -- Parallel degree of tensor model parallelism.
    -   pp (int) -- Parallel degree of pipeline model parallelism.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend)
    # we init topology and communication grid here
    from .mpu.topology import PipeModelDataParallelTopology, PipelineParallelGrid
    global topo
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=tp, num_dp=dp)
    global communication_grid
    communication_grid = PipelineParallelGrid(topo, dist.new_group(ranks=range(dist.get_world_size())))
    

    # set mpu for transformers model
    from .mpu.initialize import set_data_parallel_group, set_model_parallel_group, set_pipe_parallel_group
    set_data_parallel_group(communication_grid.get_data_parallel_group())
    set_model_parallel_group(communication_grid.get_slice_parallel_group())
    set_pipe_parallel_group(communication_grid.get_pipe_parallel_group())

    print_rank_0(f'Pipeline Model Parallel Size: {pp} \nTensor Model Parallel Size: {tp} \nData Parallel Size: {dp} \n')


def get_topo():
    global topo
    return topo

def get_grid():
    global communication_grid
    return communication_grid



from .merak_trainer import MerakTrainer
from .utils.merak_args import MerakArguments
from .modules import set_tp_layer_lists
