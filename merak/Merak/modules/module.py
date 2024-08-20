# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/module.py

import collections
import os

import re as regex

from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist

from .. import print_rank_0
from ..utils import logger, get_args
from . import utils as module_utils
from ..runtime.checkpointing import checkpoint as checkpoint_func
from ..autoshard.convert import add_inputs_to_shards
from ..mpu.layers import VocabParallelEmbedding

class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class PipelineModule(nn.Module):
    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 communicaiton_grid=None,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpoint_func,
                 activation_checkpoint_ratio=None,
                 tie_dims=set(),
                 input_to_shard_dic=None):
        """Modules to be parallelized with pipeline parallelism.

        The key constraint that enables pipeline parallelism is the
        representation of the forward pass as a sequence of layers
        and the enforcement of a simple interface between them. The
        forward pass is implicitly defined by the module ``layers``. The key
        assumption is that the output of each layer can be directly fed as
        input to the next, like a ``torch.nn.Sequence``. The forward pass is
        implicitly:

        .. code-block:: python

            def forward(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return x

        Args:
            layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
            num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
            topology (``mpu.topology.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
            loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
            base_seed (int, optional): [description]. Defaults to 1234.
            communicaiton_grid  (``mpu.topology.PipelineParallelGrid``, optional): Defines the communicators of every parallelism axes for training.
            partition_method (str, optional): [description]. Defaults to 'parameters'.
            activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
            activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``runtime.checkpointing.checkpoint``.
            tie_dims (set(str)): torch shape string of parameters that needs to be tied.
            input_to_shard_dic (dict): input name to shard id mapping from autoshard.covert func.
        """

        super().__init__()

        if num_stages is None and topology is None:
            raise RuntimeError('must provide num_stages or topology')

        self.micro_offset = 0

        self.loss_fn = loss_fn

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank != None
        self._topo = topology
        self._grid = communicaiton_grid

        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        self.num_stages = self._topo.get_dim('pipe')

        # Initialize partition information
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method, input_to_shard_dic=input_to_shard_dic)

        self.forward_funcs = []
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func
        self.activation_checkpoint_ratio = activation_checkpoint_ratio

        if self.activation_checkpoint_ratio is not None:
            if len(self.activation_checkpoint_ratio) == 1:
                first_ratio = 1-float(self.activation_checkpoint_ratio[0])
                self.activation_checkpoint_ratio = [1-(first_ratio*(self.num_stages-1)/(self.num_stages-s)) for s in range(1,self.num_stages)] + [0]
                print_rank_0(f'activation checkpoint ratio list: {self.activation_checkpoint_ratio}')
                if self.activation_checkpoint_ratio[self.stage_id] <= 0:
                    self.activation_checkpoint_interval=0
            elif len(self.activation_checkpoint_ratio) < self.num_stages:
                last_ratio = self.activation_checkpoint_ratio[-1]
                self.activation_checkpoint_ratio += [last_ratio] * (self.num_stages-len(self.activation_checkpoint_ratio))
            
        # Offset the random seed by the stage ID.
        #newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        #module_utils.set_random_seed(newseed)

        #with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        
        self._build(tie_dims, input_to_shard_dic)
        self.tied_comms = {}

            
        del self._layer_specs, layers
        self.to(torch.cuda.current_device())

    def _build(self, tie_dims, input_to_shard_dic):
        specs = self._layer_specs
        self.tied_modules_keys = set(str(s).replace(".", "_") for s in tie_dims)
        self.tied_stage = collections.defaultdict(set)

        self.input_to_stage_dic = collections.defaultdict(list)
        for input_name, shard_id in input_to_shard_dic.items():
            self.input_to_stage_dic[self.stage_owner(shard_id)].append(input_name)

        for layer_idx, layer in enumerate(specs):
            for m in layer.modules():
                if hasattr(m, 'weight'):
                    if m.weight.shape in tie_dims:
                        self.tied_stage[str(m.weight.shape).replace(".", "_")].add(self.stage_owner(layer_idx))


        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    module_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            elif isinstance(layer, nn.Module):
                name = str(layer_idx)

                inputs_of_this_stage = []
                for input in self.input_to_stage_dic[self.stage_id]:
                    if layer_idx < input_to_shard_dic[input]:
                        inputs_of_this_stage.append(input)
                # print(layer_idx, inputs_of_this_stage)
                if inputs_of_this_stage:
                    layer = add_inputs_to_shards(layer, inputs_of_this_stage)
                    # print(layer.code)
                self.forward_funcs.append(layer)
                self.add_module(name, layer)

            else:
                self.forward_funcs.append(layer)
                
        num_layers = len(self.forward_funcs)
        if self.activation_checkpoint_ratio is not None and float(self.activation_checkpoint_ratio[self.stage_id]) != 1.0:
            self.checkpointable_num_layers = 0
            self.checkpointable_idx = []
            # self.no_checkpointable_idx = []
            prev_checkpointable = None
            for start_idx in range(0, num_layers):
                end_idx = start_idx + 1
                funcs = self.forward_funcs[start_idx:end_idx]
                if self._is_checkpointable(funcs):
                    if prev_checkpointable:
                        self.checkpointable_idx[-1][1] = end_idx
                    else:
                        if prev_checkpointable is None:
                            self.frist_checkpointable = True
                        self.checkpointable_idx.append([start_idx, end_idx])
                    self.checkpointable_num_layers += 1
                    prev_checkpointable = True
                else:
                    if prev_checkpointable is None or prev_checkpointable == True:
                        if prev_checkpointable is None:
                            self.frist_checkpointable = False
                        self.checkpointable_idx.append([start_idx, end_idx])
                    else:
                        self.checkpointable_idx[-1][1] = end_idx
                    prev_checkpointable = False
        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.model_parallel = True

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(
                f"Partitioning '{layername}' found no valid layers to partition.")
        return idxs

    def forward(self, forward_input):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        self.micro_offset += 1

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:# and not isinstance(inputs[0], tuple):
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed *
                                    local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            module_utils.set_random_seed(new_seed)
                    if isinstance(inputs, tuple):
                        inputs = layer(*inputs)
                    else:
                        inputs = layer(inputs)
                return inputs

            return exec_func


        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        elif self.activation_checkpoint_ratio is not None and float(self.activation_checkpoint_ratio[self.stage_id]) != 1.0:
            # if self.stage_id == 0:
            #     self.activation_checkpoint_ratio = 0.6
            ac_num_layers = int(len(self.forward_funcs) * float(self.activation_checkpoint_ratio[self.stage_id]))
            
            ### a naive implement
            # non_ac_layers = len(self.forward_funcs) - ac_num_layers
            # x = forward_input
            # if not isinstance(x, tuple):
            #     x = (x, )
            # x = exec_range_func(0, non_ac_layers)(*x)
            # if not isinstance(x, tuple):
            #     x = (x, )
            # x = self.activation_checkpoint_func(
            #         exec_range_func(non_ac_layers, len(self.forward_funcs)),
            #         *x)

            next_checkpointable = self.frist_checkpointable
            x = forward_input
            for start_idx, end_idx in self.checkpointable_idx:
                if next_checkpointable:
                    if not isinstance(x, tuple):
                        x = (x, )
                    if ac_num_layers <= 0:
                        x = exec_range_func(start_idx, end_idx)(*x)
                    else:
                        layer_num = end_idx - start_idx
                        if ac_num_layers >= layer_num:
                            x = self.activation_checkpoint_func(
                                        exec_range_func(start_idx, end_idx),
                                        *x)
                        else:
                            x = self.activation_checkpoint_func(
                                        exec_range_func(start_idx, start_idx+ac_num_layers),
                                        *x)
                            if not isinstance(x, tuple):
                                x = (x, )
                            x = exec_range_func(start_idx+ac_num_layers, end_idx)(*x)
                        ac_num_layers -=layer_num
                else:
                    if not isinstance(x, tuple):
                        x = (x, )
                    x = exec_range_func(start_idx, end_idx)(*x)
                next_checkpointable = not next_checkpointable
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval,
                              num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )
                if self._is_checkpointable(funcs) and (self.stage_id < self.num_stages-1 or end_idx !=num_layers):
                    # print('checkpoint', self.stage_id,self.forward_funcs[start_idx:end_idx])
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx,
                                        end_idx),
                        *x)
                else:
                    # print('no checkpoint',self.stage_id,self.forward_funcs[start_idx:end_idx])
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_layers(self, method='uniform', input_to_shard_dic=None):
        # NOTE: Here each 'layer' is a `torch.fx.GraphModule` from `Merak.autoshoard`.
        #       Basically the entire DNN's computation graph is traced with `torch.fx`
        #       and grouped into 'layer's that roughly have the same number of parameters.
        #       Then, we're allocating a sequence of such layers to each pipeline stage.
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()
        
        # Convert Alpa to partitioning.
        if method == 'alpa':
            method = 'custom:0,5,9,17'
        elif method == 'alpa2':
            method = 'custom:0,5,9,13,17'

        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = module_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=num_stages, 
                                                    use_ceil=True)
        elif method == 'uniform_floor':
            num_layers = len(self._layer_specs)
            self.parts = module_utils.partition_uniform(num_items=num_layers,
                                                    num_parts=num_stages, 
                                                    use_ceil=False)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            # print(param_counts)
            self.parts = module_utils.partition_balanced(weights=param_counts,
                                                     num_parts=num_stages)
        elif method == "uniform_transformer":
            args = get_args()
            num_layers = len(self._layer_specs)
            num_transformers = args.num_transformers
            num_initial_embeddings = args.num_initial_embeddings

            # First, produces layer indices ignoring any embeddings before transformers.

            # When we split T5 with 100 shards, one encoder goes into one shard and one decoder
            # gets split into two shards. So, in order to uniformly distribute transformer layers
            # into each stage, we need to split the number of encoders and decoders separately.
            if args.model_name.startswith("t5"):
                print_rank_0("T5 partitioning")
                assert num_stages % 2 == 0, "HACK: We force #stages to be even for T5"
                num_decoders = num_encoders = num_transformers // 2
                num_encoder_gms = num_encoders
                num_decoder_gms = 2 * num_decoders
                assert num_encoder_gms % (num_stages // 2) == 0
                assert num_decoder_gms % (num_stages // 2) == 0
                num_encoder_gms_per_stage = num_encoder_gms // (num_stages // 2)
                num_decoder_gms_per_stage = num_decoder_gms // (num_stages // 2)
                parts = [0]
                for i in range(1, num_stages + 1):
                    # First half; add encoder
                    if i <= num_stages // 2:
                        parts.append(i * num_encoder_gms_per_stage)
                    # Later half; add decoder
                    else:
                        parts.append(num_encoder_gms + (i - num_stages // 2) * num_decoder_gms_per_stage)
                self.parts = parts
            # Other transformer-based models that are sharded neatly (i.e. one `GraphModule`
            # holds exactly one whole transformer encoder or decoder).
            else:
                assert num_layers == args.shard_count, f"{num_layers=} != {args.shard_count=}"
                parts = module_utils.partition_uniform(
                    num_items=num_transformers,
                    num_parts=num_stages,
                    use_ceil=True,
                )

            # All embedding layers go to the first stage. Offset every index by
            # the number of embedding layers in the beginning, except for the first.
            self.parts = list(map(lambda p: p + num_initial_embeddings, parts))
            self.parts[0] = 0

            # There are potentially other non-transformer layers in the end, e.g.
            # LM heads for classifiers. Setting the last element to `num_layers` will
            # automatically cover these.
            self.parts[-1] = num_layers

            print_rank_0(f"uniform_transformer ({num_layers=}, {num_initial_embeddings=}, {num_transformers=}): {self.parts}")

        elif method.startswith('custom:'):
            # Hardcode `parts` from a command line argument.
            # For example, `--partition_method 'custom:0,7,14,21,28'` will set
            # `parts` to `0, 7, 14, 21, 28]`. This means that there are four stages,
            # and each begins with the 0th, 7th, 14th, and 21st layer among 28 total layers.
            layer_indices = method.split(':')[1]
            self.parts = [int(x) for x in layer_indices.split(',')]
            assert num_stages == len(self.parts) - 1, "For p stages, provide p + 1 integers"
            # We can very well make the last integer implicit, but we just want to make sure
            # the user knows what they're doing by making sure they know how many layers (GraphModules)
            # there are to prevent mistakes.
            assert (
                self.parts[-1] == len(self._layer_specs)
            ), f"The last integer should equal the total number of layers ({len(self._layer_specs)})"

        elif method == 'autopipe':

            # experimental partition method
            from ..utils.profiler import FlopsProfiler
            mflops_list = []
            input = self._layer_specs[0].dummy_inputs
            if not isinstance(input, dict):
                input = [i.cpu() for i in input]
            self.dummy_input = input 
            extra_inputs = {}
            for k, v in input_to_shard_dic.items():
                if v != 0:
                    if v not in extra_inputs:
                        extra_inputs[v] = []
                    extra_inputs[v].append(input.pop(k))
            for idx, layer in enumerate(self._layer_specs):
                prof = FlopsProfiler(layer)
                prof.start_profile()
                if idx == 0 and isinstance(input, dict):
                    input = layer(**input)
                else:
                    if idx in extra_inputs:
                        input.extend(extra_inputs[idx])
                    input = layer(*input)
                flops = prof.get_total_flops()
                # if dist.get_rank() == 0:
                #     prof.print_model_profile()
                # print_rank_0(flops)
                mflops_list.append(round(flops / 10.0**6))
                prof.end_profile()
            if self.global_rank == 0:
                logger.info(f'Using experimental autopipe partition, mflops list: {mflops_list}')
            from ..autopipe import pipeline
            #  input is forward_time, backward_time, pipeline_stage
            self.parts = pipeline(mflops_list, [i*3 for i in mflops_list], num_stages)
            self.parts.append(len(self._layer_specs))
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'],
                        comm['weight_attr']),
                src=min(comm['ranks']),
                group=comm['group'],
            )

    def tie_modules(self):
        ''' Build communication structures for tied modules. '''
        # for layer_idx, layer in enumerate(specs):

        def get_name(name):
            return str(name).replace(".", "_")

        for m in self.modules():
            if hasattr(m, 'weight'):
                if m.weight is not None:
                    if get_name(m.weight.shape) in self.tied_modules_keys:
                        if get_name(m.weight.shape) in self.tied_modules:
                            # print(f'module {m} ties to exsiting {self.tied_modules[get_name(m.weight.shape)]}')
                            m.weight = self.tied_modules[get_name(m.weight.shape)].weight
                        else:
                            # print(f'module {m} needs tied to key {get_name(m.weight.shape)}')
                            self.tied_modules[get_name(m.weight.shape)] = m
                            self.tied_weight_attrs[get_name(m.weight.shape)] = 'weight'


        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return
        for key in self.tied_modules_keys:
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(self.tied_stage[key]):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp,
                                                           model=mp))
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp))
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
        self.tied_comms = tied_comms
        # print(self.tied_comms)
        self._synchronize_tied_weights()


    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def topology(self):
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim('pipe')

    def _is_checkpointable(self, funcs):
        for f in funcs:
            if isinstance(f, torch.fx.GraphModule):
                for n, m in f.named_modules():
                    if isinstance(m, (torch.nn.Embedding, VocabParallelEmbedding)):
                        return False
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
