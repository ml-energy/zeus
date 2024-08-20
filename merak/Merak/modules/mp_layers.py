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

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

from ..utils import get_args
from .. import mpu, print_rank_0


## bias 在使用PipedParallelLinear 时候不能 skip_bias_add
## 在使用AsyncRowParallelLinear 必须skip_bias_add
## AsyncColumnParallelLinear 不能skip_bias_add

class ColPara(torch.nn.Module):
    def __init__(self, in_feature, out_feature, init_method, bias=True, gather_output=False, need_permute=False):
        super(ColPara, self).__init__()
        self.need_permute = need_permute

        args = get_args()
        if args.tp_overlapping_level > 1:
            self.col_linear = AsyncColumnParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                gather_output=gather_output,
                init_method=init_method,
                skip_bias_add=False)
        elif args.tp_overlapping_level == 1:
            self.col_linear = PipedColumnParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                gather_output=gather_output,
                init_method=init_method,
                skip_bias_add=False)
        else:
            self.col_linear = mpu.ColumnParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                gather_output=gather_output,
                init_method=init_method,
                skip_bias_add=False)

    @property
    def weight(self):
        return self.col_linear.weight

    def forward(self, x):
        # input is (batch_size, seq_length, hidden_size)
        # (batch_size, seq_length, hidden_size) -> (seq_length, batch_size, hidden_size) 
        if self.need_permute: x = x.permute(1, 0, 2)
        
        # (seq_length, batch_size, hidden_size) * (4h/mp or 3h/mp, hidden_size) ->
        # (seq_length, batch_size, 4h/mp or 3h/mp)
        intermediate_parallel, _ = self.col_linear(x)
        if self.need_permute:
            # return (batch_size, seq_length, 4h/mp or 3h/mp)
            return intermediate_parallel.permute(1, 0, 2)
        else:
            return intermediate_parallel



class RowPara(torch.nn.Module):
    def __init__(self, in_feature, out_feature, output_layer_init_method, bias=True, input_is_parallel=True, need_permute=False):
        super(RowPara, self).__init__()
        self.need_permute = need_permute
        args = get_args()
        self.return_bias = (args.tp_overlapping_level > 1)
        if args.tp_overlapping_level > 1:
            self.row_linear = AsyncRowParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                input_is_parallel=input_is_parallel,
                init_method=output_layer_init_method,
                skip_bias_add=True)
        elif args.tp_overlapping_level == 1:
            self.row_linear = PipedRowParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                input_is_parallel=input_is_parallel,
                init_method=output_layer_init_method,
                skip_bias_add=False)
        else:
            self.row_linear = mpu.RowParallelLinear(
                in_feature,
                out_feature,
                bias=bias,
                input_is_parallel=input_is_parallel,
                init_method=output_layer_init_method,
                skip_bias_add=False)
    def forward(self, x):
        # input is (batch_size, seq_length, h or 4h/mp) 
        if self.need_permute: x = x.permute(1, 0, 2)
        # input is (seq_length, batch_size, 4*hidden_size/mp)

        # (seq_length, batch_size, h or 4h/mp) * (hidden_size, h or 4h/mp) ->
        # (seq_length, batch_size, hidden_size)
        output, bias = self.row_linear(x)
        
        if self.need_permute:
            # return (batch_size, seq_length, hidden_size)
            return output.permute(1, 0, 2)
        else:
            if self.return_bias:
                return output, bias
            return output



def get_async_op_hook(grad):
    if mpu.mappings.ASYNC_OP != []:
        op = mpu.mappings.ASYNC_OP.pop(0)
        op.wait()
    return grad


class AsyncColumnParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(AsyncColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.output_size_per_partition = mpu.divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        mpu.layers._initialize_affine_weight_gpu(self.weight, init_method,
                                        partition_dim=0, stride=stride)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = mpu.mappings.async_copy_to_model_parallel_region(input_)
        # input_parallel.register_hook(get_async_op_hook)
        # Matrix multiply.
        # print_rank_0([input_parallel.shape, input_parallel.grad_fn])
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = mpu.mappings.gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel 
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size, self.output_size_per_partition, self.bias is not None
        )


class AsyncRowParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(AsyncRowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.input_size_per_partition = mpu.divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()

        self.weight = nn.Parameter(torch.empty(
            self.output_size, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        mpu.layers._initialize_affine_weight_gpu(self.weight, init_method,
                                        partition_dim=1, stride=stride)
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size, device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = mpu.mappings.scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = mpu.mappings.async_reduce_from_model_parallel_region(output_parallel)
        # output = output_ + self.bias if self.bias is not None else output_
        output_bias = self.bias if self.skip_bias_add else None
        return output_, output_bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size_per_partition, self.output_size, self.bias is not None
        )


class PipedColumnParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(PipedColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.output_size_per_partition = mpu.divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        mpu.layers._initialize_affine_weight_gpu(self.weight, init_method,
                                        partition_dim=0, stride=stride)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_1, input_2 = input_.chunk(2, dim=0)


        input_parallel_1 = mpu.mappings.copy_to_model_parallel_region(input_1)
        if input_parallel_1.requires_grad:
            input_parallel_1.register_hook(get_async_op_hook)
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output_parallel_1 = F.linear(input_parallel_1, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output_1 = mpu.mappings.gather_from_model_parallel_region(output_parallel_1)
        else:
            output_1 = output_parallel_1 

        input_parallel_2 = mpu.mappings.async_copy_to_model_parallel_region(input_2)
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output_parallel_2 = F.linear(input_parallel_2, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output_2 = mpu.mappings.gather_from_model_parallel_region(output_parallel_2)
        else:
            output_2 = output_parallel_2


        output_bias = self.bias if self.skip_bias_add else None
        output = torch.cat((output_1, output_2), dim=0)
        return output, output_bias

    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size, self.output_size_per_partition, self.bias is not None
        )


class PipedRowParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(PipedRowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.input_size_per_partition = mpu.divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        assert not skip_bias_add
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()

        self.weight = nn.Parameter(torch.empty(
            self.output_size, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=args.params_dtype))
        mpu.layers._initialize_affine_weight_gpu(self.weight, init_method,
                                        partition_dim=1, stride=stride)
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size, device=torch.cuda.current_device(),
                dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = mpu.mappings.scatter_to_model_parallel_region(input_)
        
        
        input_1, input_2 = input_parallel.chunk(2, dim=0)
        # Matrix multiply.
        output_parallel_1 = F.linear(input_1, self.weight)
        # All-reduce across all the partitions.
        output_1_ = mpu.mappings.async_reduce_from_model_parallel_region(output_parallel_1)
        
        output_parallel_2 = F.linear(input_2, self.weight)

        async_op = mpu.mappings.ASYNC_OP.pop(0)
        async_op.wait()
        output_2_ = mpu.mappings.reduce_from_model_parallel_region(output_parallel_2)
        

        output_1 = output_1_ + self.bias if self.bias is not None else output_1_
        output_2 = output_2_ + self.bias if self.bias is not None else output_2_

        output = torch.cat((output_1, output_2), dim=0)
        return output, None

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size_per_partition, self.output_size, self.bias is not None
        )

