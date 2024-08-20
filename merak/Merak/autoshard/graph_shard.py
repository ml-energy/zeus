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

# Parts of the code here are adapted from https://github.com/facebookresearch/fairscale/blob/86c62cc9c989178e629823c2fd4a3cc11723e92b/fairscale/experimental/nn/auto_shard.py

import collections
from copy import deepcopy
import logging
from typing import Dict, List, Set

import torch
import torch.fx
from torch.fx.node import Node
from .. import print_rank_0

SHARD_METHOD = 'exclude_emb'
# SHARD_METHOD = 'param_uniform'

def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)

def _get_count(param_count: Dict, node_name: str) -> int:
    """Identify different mutations of a given node name."""

    if node_name in param_count:
        return param_count[node_name]
    elif node_name.replace(".", "_") in param_count:
        return param_count[node_name.replace(".", "_")]
    else:
        raise RuntimeError(f"Unable to find match between param {param_count} and node {node_name}")


def _create_shard_to_param_count(param_count: Dict, node_name_to_shard_id: Dict) -> Dict:
    """Utility to create a map from shard id to param count using existing state."""

    shard_to_param_count: Dict[int, int] = {}
    for node_name in node_name_to_shard_id.keys():
        try:
            count = _get_count(param_count, node_name)
        except RuntimeError:
            # print_rank_0(f"Unable to find match node {node_name}")
            continue
        if node_name_to_shard_id[node_name] in shard_to_param_count:
            shard_to_param_count[node_name_to_shard_id[node_name]] += count
        else:
            shard_to_param_count[node_name_to_shard_id[node_name]] = count
    return shard_to_param_count


def _split_nodes(traced_graph_module, shard_count, node_user_count=None) -> Dict:
    """Utility used to trace a graph and identify shard cutpoints."""

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id = 0
    nodes_so_far = []
    param_count: Dict[str, int] = {}
    shard_to_param_count = {}
    excluded_count = 0
    init_node_user_count = deepcopy(node_user_count)
    
    # Find the total number of params in the model and
    # the number of params per shard we are aiming for.
    for name, module in traced_graph_module.named_modules():
        name = _snake_case(name).replace(".", "_")
        if SHARD_METHOD == 'exclude_emb':
            if isinstance(module, torch.nn.Embedding):
                param_count[name] = 0
                excluded_count += sum([x.numel() for x in module.parameters()])
                continue
            ## test only
            # if isinstance(module, torch.nn.Embedding):
            #     param_count[name] = 1024*1024
            #     continue
            # if isinstance(module, torch.nn.Linear):
            #     param_count[name] = 1024*1024
            #     continue
        param_count[name] = sum([x.numel() for x in module.parameters()])

    for name, p in traced_graph_module.named_parameters():
        name = _snake_case(name).replace(".", "_")
        param_count[name] = p.numel()

    # print_rank_0(param_count)
    print_rank_0(f"Total number of params are {param_count['']}")
    per_shard_param = (param_count[""]-excluded_count)// shard_count
    print_rank_0(f"Per shard param count {per_shard_param}")
    print_rank_0(f"Node count {len(traced_graph_module.graph.nodes)}")


    if SHARD_METHOD == 'exclude_emb':
        for name, module in traced_graph_module.named_modules():
            name = _snake_case(name).replace(".", "_")
            if isinstance(module, torch.nn.Embedding):
                # HACK: T5 has a relative attention bias layer as an nn.Embedding only in the first
                #       transformer encoder self-attention layer. If we set the number of parameters to
                #       `per_shard_param + 1` for that one, we end up splitting a transformer encoder
                #       in the middle, and end up with a `GraphModule` that has half a transformer layer,
                #       which makes all later `GraphModule`s chimeras of two transformer layers.
                if "relative_attention_bias" not in name:
                    param_count[name] = per_shard_param + 1


    func_inputs = {}
    shard_output = {-1: None, 0: None}
    extra_output = {}

    for node in traced_graph_module.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            func_inputs[node.name] = 0
        elif node.op in ["get_attr", "call_function", "call_method", "call_module"]:

            min_shard_id = shard_id
            min_node_name = ""
            # For each of the args of a given node, find the arg that is not the
            # last node we traversed. This is to help us find skip connections
            # across shards.
            # 如果args里面有非上一个node输出的输入, 找到args里面shard_id最小的, 记录shard_id和名字
            # print(node.name, node.op, node.args)
            for arg in node.args:
                # If the node has args that are inputs to the forward function, they
                # may not have explicit names.

                if not hasattr(arg, "name"): 
                    continue

                # 如果args有需记录user的node, 将user数减一
                if arg.name in node_user_count:
                    node_user_count[arg.name] -= 1

                # if a node is inplace OP, it should stay with its input
                if node.op is 'call_module':
                    try:
                        if hasattr(traced_graph_module, node.target):
                            mod = getattr(traced_graph_module, node.target)
                        else:
                            submod = traced_graph_module
                            prefix = node.target.split('.')
                            for item in prefix:
                                mod = getattr(submod, item, None)
                                submod = mod
                        if hasattr(mod, 'inplace'):
                            min_node_name = arg.name
                            min_shard_id = node_name_to_shard_id[min_node_name]
                            continue
                    except:
                        pass

                
                # 如果args里面有某一个shrad的输出
                if arg.name in shard_output.values():
                    # 如果args里面有上一个shrad输出, 跳过
                    if arg.name == shard_output[shard_id-1]:
                        continue
                    # 不是上一个shard的输出改node最小shard_id为此shard+1
                    # print_rank_0([node.name, 'has ', arg.name, node_name_to_shard_id[arg.name]])
                    if node_name_to_shard_id[arg.name] + 1 < min_shard_id:
                        min_shard_id = node_name_to_shard_id[arg.name] + 1
                        min_node_name = arg.name
                    continue

                # 记录inputs的使用情况
                if arg.name in func_inputs:
                    if func_inputs[arg.name] == 0:
                        # the first node to use this input
                        func_inputs[arg.name] = node.name
                        continue
                    else:
                        input_arg_id = node_name_to_shard_id[func_inputs[arg.name]]
                        if input_arg_id < min_shard_id:
                            min_shard_id = input_arg_id
                            min_node_name = func_inputs[arg.name]
                        continue

                if arg.name in node_name_to_shard_id and arg.name != nodes_so_far[-1]:
                    if node_name_to_shard_id[arg.name] < min_shard_id and arg.name not in node_user_count:
                        min_shard_id = node_name_to_shard_id[arg.name]
                        # print_rank_0(['because of ', arg.name, node_name_to_shard_id[arg.name]])
                        min_node_name = arg.name

            # If there is an input that is not from the previous shard,
            # we collapse all the shards in between to be part of 1 shard.
            # and update the param count per shard accordingly.
            # 从args内shard_id最小的输入到当前node, 需要划分到同一个shard内, 
            # 当前shard_id也改为此shard_id

            if min_shard_id < shard_id:
                for node_name in reversed(nodes_so_far):
                    if node_name_to_shard_id[node_name] > min_shard_id:
                        # print_rank_0(['reset', node_name, node_name_to_shard_id[node_name], min_shard_id])
                        node_name_to_shard_id[node_name] = min_shard_id
                    if node_name == min_node_name:
                        break
                shard_id = min_shard_id

                shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)                
            # Update state that is tracking node -> shard id and shard id -> param count.
            node_name_to_shard_id[node.name] = shard_id
            # print_rank_0([node.name, node_name_to_shard_id[node.name]])
            nodes_so_far.append(node.name)

            shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)
            # print_rank_0([node.name, shard_to_param_count])
            # If we have gone over the number of params per shard count that we want to
            # achieve, we should add a new shard.
            # The shard_id may not have been updated in the map if we are at a node that does not
            # have params.
            if shard_id in shard_to_param_count and shard_to_param_count[shard_id] > per_shard_param:
                shard_output[shard_id] = node.name
                reset_keys = [ i for i in shard_output.keys() if i > shard_id]
                for key in reset_keys:
                    shard_output.pop(key)

                # 如果需记录user的node中, node已产生且还有user未使用, 则需要成为这个shard的output
                extra_output[shard_id] = [k for k in node_user_count \
                    if node_user_count[k]>0 and k in node_name_to_shard_id]
    
                reset_keys = [ i for i in extra_output.keys() if i > shard_id]
                for key in reset_keys:
                    extra_output.pop(key)
                shard_id += 1
                # print_rank_0(['output', shard_output])
                # print_rank_0([extra_output, node_user_count])
        elif node.op == "output":
            break
        # print_rank_0([node.name, len(nodes_so_far)])    
    # print_rank_0(func_inputs)
    # print_rank_0([ node_name_to_shard_id[name] for name in shard_output.values() if name is not None])
    for k, v in func_inputs.items():
        func_inputs[k] = node_name_to_shard_id[v]
    # print_rank_0(func_inputs)
    return node_name_to_shard_id, shard_output, func_inputs, extra_output



def shard_model_transformers(traced_graph_module, model, shard_count=3, output_nodes_count=None):
    """Utility used to shard a model using torch.fx.

    This function traces the model twice in an attempt to identify the
    right cutpoints and then shard the model. In the first pass we calculate
    the number of parameters as we are tracing the graph and mark nodes at
    which we might want to create a new module. In the second pass we
    modify the graph by inserting placeholders and output nodes to essentially
    shard the graph.

    We don't support skip connections between shards. This means that all
    input and output is self contained within a given shard. A node from
    shard 1 cannot be an input to a node from shard 3. We expect all inputs
    to a given shard to be coming from the last node in the previous shard.
    This means that we may not be able to shard models by the specified
    `shard_count` mentioned by the user.

    Args:
        model (nn.Module): Model to be sharded as specified by the device count.

        shard_count (int): Number of shards that we want to split the model into.

    """
    module_list: List[torch.fx.GraphModule] = []
    num_graphs = 0
    new_graph = torch.fx.Graph()  # type: ignore
    env: Dict[str, Node] = {}
    new_input_node = None

    # traced_graph_module = _trace(model)

    # This is the first pass where we attempt to get a map of where
    # we need to insert placeholder and output nodes.
    node_name_to_shard_id, shard_output, func_inputs, extra_output = _split_nodes(traced_graph_module, shard_count, output_nodes_count)

    # dummy value which indicates that this is the first node.
    prev_shard_id = 1000
    prev_node = None
    for node in traced_graph_module.graph.nodes:
        # If the current node is in the next shard, we insert an output node.
        # A new graph is created and a placeholder is added for the next shard.
        if node.name in node_name_to_shard_id and prev_shard_id < node_name_to_shard_id[node.name]:
            assert prev_node, "prev_node cannot be None"

            with new_graph.inserting_after(prev_node):
                # print_rank_0(f'making output of graph {num_graphs}')
                if num_graphs in extra_output:
                    extra_output_name = extra_output[num_graphs]
                    if isinstance(env[prev_node.name], tuple):
                        added_output = env[prev_node.name]+ tuple([env[i] for i in extra_output_name])
                    else:
                        added_output = tuple([env[prev_node.name]] + [env[i] for i in extra_output_name])
                    new_graph.create_node(op='output', target='output', args=(added_output,))
                else:
                    new_graph.output(env[prev_node.name])

            num_graphs += 1
            module_list.append(torch.fx.GraphModule(model, new_graph))
            new_graph = torch.fx.Graph()
            # node_name = shard_output[num_graphs-1]
            node_name = "placeholder" + str(num_graphs)
            pl_node = new_graph.create_node("placeholder", node_name)
            env[node_name] = pl_node
            # print('********', node.args)
            env[shard_output[num_graphs-1]] = pl_node
            new_input_node = pl_node

            if num_graphs-1 in extra_output:
                extra_output_name = extra_output[num_graphs-1]
                for i in extra_output[num_graphs-1]:
                    pl_node = new_graph.create_node("placeholder", i)
                    env[i] = pl_node

            for key, v in func_inputs.items():
                if v == num_graphs:
                    pl_node = new_graph.create_node("placeholder", key)
                    env[key] = pl_node

        if new_input_node is not None:
            # Account for a placeholder in the new graph.
            if len(node.args) > 1:
                new_args = []
                for arg in node.args:
                    if hasattr(arg, 'name'):
                        new_args.append(new_input_node)
                    else:
                        new_args.append(arg)
                assert len(node.args) == len(new_args), f'args of node {node.name} length not match'
                node.args = tuple(new_args)
            else:
                node.args = (new_input_node,)
            new_input_node = None

        if node.op in ["placeholder", "get_attr", "call_function", "call_method", "call_module"]:
            # Copy the nodes from the existing graph to the new graph.
            if node.op == 'placeholder':
                # node_name_using_this_input = func_inputs[node.name]
                if func_inputs[node.name] == 0:
                    new_node = new_graph.node_copy(node, lambda x: env[x.name])
                    env[node.name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
            # new_node = new_graph.node_copy(node, lambda x: env[x.name])
            # env[node.name] = new_node
        elif node.op == "output":
            # If this is the last node, we should add an output
            # node and add the last graph to the list.
            assert prev_node, "prev_node cannot be None"

            with new_graph.inserting_after(prev_node):
                # new_graph.output(env[prev_node.name])
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
            new_graph.lint()
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break
        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list, func_inputs
