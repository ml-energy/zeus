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

import torch
import torch.nn as nn
import torch.distributed as dist
from .. import print_rank_0
from ..mpu import get_model_parallel_world_size
import transformers

from .utils import init_method_normal, scaled_init_method_normal
from .mp_layers import ColPara, RowPara
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math

SHOULD_PRINT_CONV = True
SHOULD_PRINT_LINEAR = True


class NumelParameter(nn.Parameter):
    def numel(self):
        return self.num_element()


class LinearProxy(nn.Module):
    def __init__(self, *module_args, **module_kwargs):
        self.module_args = module_args
        if len(module_kwargs) != 0:
            for k in module_kwargs:
                self.module_args += (module_kwargs[k],)
        super(LinearProxy, self).__init__()
        global SHOULD_PRINT_LINEAR
        if SHOULD_PRINT_LINEAR:
            print_rank_0('using linear proxy')
            SHOULD_PRINT_LINEAR = False
        self.in_features = self.module_args[0]
        self.out_features = self.module_args[1]
        if len(self.module_args) >= 3:
            self._bias = self.module_args[2]
        else:
            self._bias = False

        self.mp_attr = ' '

        self.__flops__ = 0

        w = torch.empty(1, 1)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.out_features*self.in_features

        if self._bias:
            self.bias = NumelParameter(torch.zeros(1))
            self.bias.num_element = lambda: self.out_features
        else:
            self.register_parameter('bias', None)


    def build(self, init_args, fp16):
        if self.mp_attr == ' ':
            if fp16:
                return torch.nn.Linear(*self.module_args).cuda().half()
            else:
                return torch.nn.Linear(*self.module_args).cuda()
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias)
            else:
                return RowPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias, need_permute=False)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias)
            else:
                return ColPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias, need_permute=False)
        
        raise NotImplementedError(f"Not supported model/tensor parallelism layers {self.mp_attr}")
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def forward(self, x):
        # shape = x.shape
        # shape[-1] = self.out_features
        # if len(x.shape) == 2:
        #     return x[:, :1].expand(-1, self.out_features).contiguous()
        self.__flops__ = 2 * x.numel() * self.out_features

        try:
            return x[:, :, :1].expand(-1,-1, self.out_features).contiguous()#, device=x.device)
        except IndexError:
            return x[:, :1].expand(-1, self.out_features).contiguous()


class Conv1DProxy(nn.Module):
    def __init__(self, out_features, in_features):
        self.module_args = (out_features, in_features)
        super(Conv1DProxy, self).__init__()
        global SHOULD_PRINT_CONV
        if SHOULD_PRINT_CONV:
            print_rank_0('using conv1d proxy')
            SHOULD_PRINT_CONV = False
        self.in_features = in_features
        self.out_features = out_features

        self.mp_attr = ' '

        w = torch.empty(1, 1)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.out_features*self.in_features
        self.bias = NumelParameter(torch.zeros(1))
        self.bias.num_element = lambda: self.out_features
    
    def build(self, init_args, fp16):
        if self.mp_attr == ' ':
            if fp16:
                return transformers.modeling_utils.Conv1D(*self.module_args).cuda().half()
            else:
                return transformers.modeling_utils.Conv1D(*self.module_args).cuda()
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[1], self.module_args[0], init_method)
            else:
                return RowPara(self.module_args[1], self.module_args[0], init_method)#, need_permute=True)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[1], self.module_args[0], init_method)
            else:
                return ColPara(self.module_args[1], self.module_args[0], init_method)#, need_permute=True)
        
        raise NotImplementedError(f"Not supported model/tensor parallelism layers {self.mp_attr}")

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def forward(self, x):
        self.__flops__ = 2 * x.numel() * self.out_features
        return x[:, :, :1].expand(-1, -1, self.out_features)
