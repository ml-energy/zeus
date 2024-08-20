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

from transformers.utils.fx import _generate_supported_model_classes

SUPPORTED_MODEL_NAMES = [
    "bert",
    "gpt2",
    "t5",
    "vit"
]

MP_MODEL_MAPPING = {
    'gpt2': {
        'input_output_mapping':[(1, 3, 'col'), (1, 1,'row'), (1, 4, 'col'), (4, 1, 'row')],
        'tp_attr_list':['num_heads', 'split_size']         
    },
    't5': {
        'col_para_list':['Attention.q', 'Attention.k', 'Attention.v', 'DenseReluDense.wi'],
        'row_para_list':['Attention.o', 'DenseReluDense.wo'],
        'weight_change_list':[('relative_attention_bias', 1)],
        'tp_attr_list':['n_heads', 'inner_dim']
    },
    'bert':{
        'col_para_list':['query', 'key', 'value', 'intermediate.dense'],
        'row_para_list':['output.dense'],
        'tp_attr_list':['num_attention_heads','all_head_size']
    },
    'vit':{
        'col_para_list':['query', 'key', 'value', 'intermediate.dense'],
        'row_para_list':['output.dense'],
        'tp_attr_list':['num_attention_heads','all_head_size']
    },
}

def get_mp_layer_lists(model_class):
    for model_name in SUPPORTED_MODEL_NAMES:
        if model_class in _generate_supported_model_classes(model_name):
            return MP_MODEL_MAPPING[model_name]
    return None