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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/utils/fx.py

from multiprocessing import dummy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .. import print_rank_0
import copy
import functools
import inspect
import random
import os
import transformers
from packaging import version

def money_pathch_is_torch_fx_available():
    return True
transformers.file_utils.is_torch_fx_available = money_pathch_is_torch_fx_available

from transformers.utils.fx import (HFTracer, 
                        _generate_random_int, 
                        transform_to_dynamic_input_, 
                        _generate_supported_model_classes,
                        _SUPPORTED_MODELS, 
                        _SUPPORTED_MODELS_FOR_DYNAMIC_AXES,
                        _wrap_method_for_model_tracing,
                        _reset_tensor_methods)

from .graph_shard import shard_model_transformers
from ..modules.layer_proxy import LinearProxy, Conv1DProxy
from ..modules.transformer_blocks import PipedGPT2Block

import gc


from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
    PretrainedConfig,
    PreTrainedModel,
    logging,
)
from transformers.models.auto import get_values

def hf_fx_compatibility(model):
    added_model = tuple(_generate_supported_model_classes('vit'))
    transformers_fx_models = tuple(_SUPPORTED_MODELS+_SUPPORTED_MODELS_FOR_DYNAMIC_AXES+added_model)
    if isinstance(model, PreTrainedModel) and isinstance(model, transformers_fx_models):
        return True
    else:
        return False
    
default_leaf_modules = (LinearProxy, Conv1DProxy, PipedGPT2Block)

def convert_to_sequential(model, args, extra_leaf_modules=(), trace_batch=None):
    added_model = tuple(_generate_supported_model_classes('vit'))
    transformers_fx_models = tuple(_SUPPORTED_MODELS+_SUPPORTED_MODELS_FOR_DYNAMIC_AXES+added_model)
    # print_rank_0(transformers_fx_models)
    if not args.fp16:
        model.cpu()

    if args.cache_sharding:
        assert args.cache_dir is not None
        if os.path.isfile(f'{args.cache_dir}_graph0_cache.pt'):
            cache_dir = os.path.split(f'{args.cache_dir}_graph0_cache.pt')[0]
            result_len = len([n for n in os.listdir(cache_dir) if args.cache_dir.split("/")[-1] + '_graph' in n])
            result = []
            for i in range(result_len):
                graph_slice = torch.load(f'{args.cache_dir}_graph{i}_cache.pt')
                result.append(graph_slice)
                del graph_slice
            input_to_shard = torch.load(f'{args.cache_dir}_input_cache.pt')

            print_rank_0(f"Retrieved cached graphs from {args.cache_dir}")

            return model, result, input_to_shard

    if isinstance(model, transformers_fx_models) or "bloom" in str(type(model)).lower():
        traced, dummy_inputs = symbolic_trace(
            model,
            input_names = args.input_names,
            batch_size = args.per_device_train_batch_size,
            sequence_length = args.seq_length,
            extra_leaf_modules = extra_leaf_modules,
            trace_batch=trace_batch,
        )
    else:
        if isinstance(extra_leaf_modules, list):
            extra_leaf_modules = tuple(extra_leaf_modules)
        elif isinstance(extra_leaf_modules, nn.Module):
            extra_leaf_modules = tuple([extra_leaf_modules])
        else:
            assert isinstance(extra_leaf_modules, tuple), 'leaf_modules should be tuple'
        

        tracer = LayerProxyTracer(leaf_modules=default_leaf_modules+extra_leaf_modules)
        traced_graph = tracer.trace(model)
        traced = torch.fx.GraphModule(model, traced_graph)
        dummy_inputs = None

    ## test code
    # print_rank_0(traced.graph)
    # if torch.distributed.get_rank() == 0:
    #   traced.graph.print_tabular()
    # print_rank_0(traced.code)
    # print_rank_0(traced)
    # print_rank_0(model)

    ## a experience users number threshold, a node has more user than this threshold    
    ## indicate the node is needed in multiple stages and could be transmitted between stages
    output_node_threshold = 5
    output_nodes_count = {}
    for node in traced.graph.nodes:
        if len(list(node.users)) > output_node_threshold:
            output_nodes_count[node.name] = len(list(node.users))



    result, input_to_shard = shard_model_transformers(
        traced, model, args.shard_count, output_nodes_count
    )
    del traced
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result[0].dummy_inputs = dummy_inputs
    if args.half_precision_backend != "apex":
        torch.cuda.synchronize()

    if args.cache_sharding:# and isinstance(model, transformers_fx_models):
        if args.local_rank == 0:
        # if dist.get_rank() == 0:
            file_path = os.path.abspath(os.path.dirname(args.cache_dir))
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            for idx, graph in enumerate(result):
                # graph.to_folder(f'{args.cache_dir}', f'module_{idx}')
                torch.save(graph, f'{args.cache_dir}_graph{idx}_cache.pt')
            torch.save(input_to_shard, f'{args.cache_dir}_input_cache.pt')
            dist.barrier()
        else:
            dist.barrier()

    # # test code
    # for i in result:
    #     if torch.distributed.get_rank() == 0:
    #         i.graph.print_tabular()
    #     print_rank_0(i.code)

    return model, result, input_to_shard

class LayerProxyTracer(torch.fx.Tracer):
    """Tracer with an extended set of leaf nn.Modules."""

    def __init__(self, leaf_modules):
        super().__init__()
        self.leaf_modules = leaf_modules
    
    def is_manual_leaf_module(self, m):
        for i in self.leaf_modules:
            if isinstance(m, i):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        return super().is_leaf_module(m, model_qualified_name) or self.is_manual_leaf_module(m)


class MpTracer(HFTracer):
    def __init__(self, leaf_modules=(), manual_input_shape = None, trace_batch=None, batch_size=1, sequence_length=[128, 128], num_choices=-1):
        super().__init__(batch_size, sequence_length, num_choices)
        self.leaf_modules = leaf_modules
        if manual_input_shape is not None:
            self.encoder_shape = manual_input_shape

        self.trace_batch = trace_batch

    def is_manual_leaf_module(self, m):
        for i in self.leaf_modules:
            if isinstance(m, i):
                return True
        return False
        
    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        return super().is_leaf_module(m, model_qualified_name) or self.is_manual_leaf_module(m)
    
    def _generate_dummy_input(self, model, input_name):
        """Generates dummy input for model inference recording."""
        model_class = model.__class__
        device = model.device
        # device = 'cpu'
        inputs_dict = dict()
        if self.trace_batch is not None:
            return self.trace_batch

        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = self.encoder_shape[0]
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.ones(batch_size, dtype=torch.long, device=device)
            elif model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                GPT2DoubleHeadsModel,
            ]:
                inputs_dict["labels"] = torch.zeros(self.decoder_shape, dtype=torch.long, device=device)
            elif model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(self.encoder_shape, dtype=torch.long, device=device)
            else:
                raise NotImplementedError(f"{model_class} not supported yet.")

        elif "mask" in input_name or "ids" in input_name:
            shape = self.encoder_shape if "decoder" not in input_name else self.decoder_shape
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.long, device=device)
        elif "pixel_values" in input_name:
            shape = [self.encoder_shape[0], model.config.num_channels, 
                    model.config.image_size, model.config.image_size]
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.float, device=device)
        else:
            shape = self.encoder_shape if "decoder" not in input_name else self.decoder_shape
            shape += [model.config.hidden_size]
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.float, device=device)
        return inputs_dict

    def trace(self, root: PreTrainedModel, concrete_args = None, method_names=None):
        if concrete_args is None:
            concrete_args = {}

        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() - concrete_args.keys()

        self.record(root, input_names, method_names=method_names)

        for method_name, cache_name in self.recorded_methods.items():
            _wrap_method_for_model_tracing(root, method_name, cache_name)

        graph = torch.fx.Tracer.trace(self, root, concrete_args=concrete_args)

        _reset_tensor_methods(self.original_methods)

        torch_version = version.parse(torch.__version__)
        if torch_version.minor <= 11:
            # torch version compatibility
            # https://github.com/huggingface/transformers/pull/17129
            # https://github.com/pytorch/pytorch/pull/59569
            for node in graph.nodes:
                if node.op == "placeholder":
                    # Removing default values for inputs as the forward pass will fail with them.
                    if node.target in input_names:
                        node.args = ()
                    # It is a concrete arg so it is not used and should be removed.
                    else:
                        graph.erase_node(node)
        return graph


def symbolic_trace(
    model,
    input_names = None,
    batch_size = 1,
    sequence_length = (128, 128),
    num_choices = -1,
    extra_leaf_modules=(),
    trace_batch=None,
):

    """
    Performs symbolic tracing on the model.

    Args:
        model (:obj:`PretrainedModel`):
            The model to trace.
        input_names (:obj:`List[str]`, `optional`):
            The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The batch size of the traced model inputs.
        sequence_length (:obj:`int` or :obj:`List[int]]`):
            The sequence length of the traced model inputs. For sequence-to-sequence models with different sequence
            lengths between the encoder and the decoder inputs, this must be :obj:`[encoder_sequence_length,
            decoder_sequence_length]`.
        num_choices (:obj:`int`, `optional`, defaults to -1):
            The number of possible choices for a multiple choice task.

    Returns:
        :obj:`torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example::

        from transformers.utils.fx import symbolic_trace
        traced_model = symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            batch_size=1,
            sequence_length=128,
        )
    """
    if input_names is None or input_names == []:
        input_names = model.dummy_inputs.keys()

    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    # print(concrete_args)
    # Preparing HFTracer batch_size and sequence_lenght values for potential dynamic axes.
    use_dynamic_batch_size = batch_size <= 0
    if isinstance(sequence_length, (list, tuple)):
        use_dynamic_sequence_length = sequence_length[0] <= 0 or sequence_length[1] <= 0
    elif isinstance(sequence_length, int):
        use_dynamic_sequence_length = sequence_length <= 0
    else:
        use_dynamic_sequence_length = False

    if use_dynamic_batch_size or use_dynamic_sequence_length:
        forbidden_values = [
            model.config.num_attention_heads,
            model.config.hidden_size,
            model.config.hidden_size // model.config.num_attention_heads,
        ]
        if use_dynamic_batch_size:
            batch_size = _generate_random_int(forbidden_values=forbidden_values)
        forbidden_values.append(batch_size)
        if use_dynamic_sequence_length:
            encoder_sequence_length = _generate_random_int(forbidden_values=forbidden_values)
            forbidden_values.append(encoder_sequence_length)
            decoder_sequence_length = _generate_random_int(forbidden_values=forbidden_values)
            sequence_length = [encoder_sequence_length, decoder_sequence_length]


    if isinstance(extra_leaf_modules, list):
        extra_leaf_modules = tuple(extra_leaf_modules)
    elif isinstance(extra_leaf_modules, nn.Module):
        extra_leaf_modules = tuple([extra_leaf_modules])
    else:
        assert isinstance(extra_leaf_modules, tuple), 'leaf_modules should be tuple'
    # Tracing.
    tracer = MpTracer(leaf_modules=default_leaf_modules+extra_leaf_modules,
            trace_batch=trace_batch,
            batch_size=batch_size, 
            sequence_length=sequence_length, 
            num_choices=num_choices)
    with torch.no_grad():
        traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)
    dummy_inputs = {}

    for name in input_names:
        dummy_inputs.update(tracer._generate_dummy_input(model, name))

    del traced_graph, tracer

    traced.config = copy.deepcopy(model.config)
    traced.num_choices = num_choices

    traced.use_dynamic_batch_size = use_dynamic_batch_size
    traced.use_dynamic_sequence_length = use_dynamic_sequence_length
    traced.static_batch_size = batch_size
    traced.static_sequence_length = sequence_length

    transform_to_dynamic_input_(traced)

    return traced, dummy_inputs


def add_inputs_to_shards(gm, inputs):
    # for input_name in inputs:
    add_outputs = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and node.next.op != 'placeholder' and add_outputs == []:
            # print(node)
            with gm.graph.inserting_after(node):
                for input_name in reversed(inputs):
                    pl_node = gm.graph.create_node("placeholder", input_name)
                    add_outputs.append(pl_node)
        elif node.op == "output":
            with gm.graph.inserting_after(node):
                # if node.args
                added_output = node.args[0] + tuple(reversed(add_outputs))
                gm.graph.create_node(op='output', target='output', args=(added_output,))
                # gm.graph.output(added_output)
            gm.graph.erase_node(node)
            break
    gm.recompile()
    return gm
                    
