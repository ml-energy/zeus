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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/gpt2/modeling_gpt2.py

import math
import os
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2Model, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils.fx import _generate_supported_model_classes
from .layer_proxy import Conv1DProxy, LinearProxy
from .. import mpu, print_rank_0
from ..utils import get_args


def tp_overlapping_available(model_class):
    SUPPORTED_MODEL_NAMES = ['gpt2']
    for model_name in SUPPORTED_MODEL_NAMES:
        if model_class in _generate_supported_model_classes(model_name):
            return True
    return False



class DropoutResidual(nn.Module):
    def __init__(self, drop_out):
        super(DropoutResidual, self).__init__()
        self.hidden_dropout = drop_out
    
    def forward(self, hidden, residual):
        out = torch.nn.functional.dropout(hidden, p=self.hidden_dropout)
        layernorm_input = residual + out
        return layernorm_input


class PipedGPT2Attention(GPT2Attention):
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)


        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        if isinstance(self.c_proj, (Conv1DProxy,LinearProxy)):
            attn_output = self.c_proj(attn_output)
            attn_bias = attn_output[0]
        else:
            attn_output, attn_bias = self.c_proj(attn_output)


        return attn_output, attn_bias


class PipedMlp(GPT2MLP):
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        if isinstance(self.c_proj, (Conv1DProxy,LinearProxy)):
            hidden_states = self.c_proj(hidden_states)
            hidden_states_bias = hidden_states[0]
        else:
            hidden_states, hidden_states_bias = self.c_proj(hidden_states)
        return hidden_states, hidden_states_bias


class PipedGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()

        assert not config.add_cross_attention, 'only support self attention for now'
        args = get_args()
        self.overlap_level = args.tp_overlapping_level
        assert self.overlap_level > 1, 'overlap_level should great than 1'

        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.is_last_layer = (layer_idx==config.num_hidden_layers-1)
        self.is_first_layer = (layer_idx==0)

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.self_attention = PipedGPT2Attention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.drop_residual = DropoutResidual(config.resid_pdrop)

        self.mlp = PipedMlp(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        assert layer_past == None
        # assert attention_mask is not None, 'only support data with attention_mask'
        if self.overlap_level == 2:
            hidden_states_chunks = hidden_states.chunk(2, dim=0)
        else:
            if self.is_first_layer:
                hidden_states_chunks = hidden_states.chunk(2, dim=0)
            else:
                hidden_states_chunks = hidden_states.chunk(4, dim=0)
        
        if attention_mask is not None:
            attention_mask_chunks = attention_mask.chunk(2, dim=0)
        else:
            attention_mask_chunks = (None, None)

        if self.overlap_level == 2:
            hidden_states_chunks = PipedTPFunction.apply(self, attention_mask_chunks, \
                head_mask, *hidden_states_chunks)
        else:
            hidden_states_chunks = CrossLayerPipedTPFunction.apply(self, attention_mask_chunks, \
                head_mask, *hidden_states_chunks)

        hidden_states = torch.cat(hidden_states_chunks, dim=0)

        return hidden_states

class PipedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([PipedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert encoder_hidden_states == None
        assert use_cache==False
        assert output_attentions == False
        assert output_hidden_states == False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)


        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



def sync_comm():
    if mpu.mappings.ASYNC_OP != []:
        async_op = mpu.mappings.ASYNC_OP.pop(0)
        async_op.wait()


"""
We break the fp and bp of transformer block into several phases:

|<-         fp1&2        ->|<-                fp2&3                 ->|<-    fp5&6      ->|
layernorm -> self attention -> dropout & residual -> layernorm -> mlp -> dropout & residual
|<-bp5&6->|<-                      bp2&3                     ->|<-       bp1&2          ->|

"""


class PipedTPFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_functions, attention_mask, head_mask, input_1, input_2):
        
        with torch.enable_grad():
            ## fp 1
            # input_1_clone = input_1
            input_1_clone = input_1.detach()
            input_1_clone.requires_grad = True

            layernorm_output_1 = run_functions.input_layernorm(input_1_clone)
            
            # ln1_out = layernorm_output_1
            ln1_out = layernorm_output_1.detach()
            ln1_out.requires_grad = True

            attention_output_1, attention_output_1_bias = \
                run_functions.self_attention(ln1_out, attention_mask=attention_mask[0], head_mask=head_mask)

            ## fp 2
            # input_2_clone = input_2
            input_2_clone = input_2.detach()
            input_2_clone.requires_grad = True

            layernorm_output_2 = run_functions.input_layernorm(input_2_clone)

            # ln2_out = layernorm_output_2
            ln2_out = layernorm_output_2.detach()
            ln2_out.requires_grad = True

            attention_output_2, attention_output_2_bias = \
                run_functions.self_attention(ln2_out, attention_mask=attention_mask[1], head_mask=head_mask)

            ## fp 3
            sync_comm()
            layernorm_input_1 = run_functions.drop_residual(attention_output_1+attention_output_1_bias, input_1_clone)
            post_attn_ln_1 = run_functions.post_attention_layernorm(layernorm_input_1)

            # ln_1_in = post_attn_ln_1
            post_ln_1_in = layernorm_input_1.detach()
            post_ln_1_in.requires_grad = True
            post_ln_1_out = post_attn_ln_1.detach()
            post_ln_1_out.requires_grad = True

            mlp_output_1, mlp_bias_1 = run_functions.mlp(post_ln_1_out)

            ## fp 4
            sync_comm()
            layernorm_input_2 = run_functions.drop_residual(attention_output_2+attention_output_2_bias, input_2_clone)
            post_attn_ln_2 = run_functions.post_attention_layernorm(layernorm_input_2)

            # ln_2_in = post_attn_ln_2
            post_ln_2_in = layernorm_input_1.detach()
            post_ln_2_in.requires_grad = True
            post_ln_2_out = post_attn_ln_2.detach()
            post_ln_2_out.requires_grad = True
            
            mlp_output_2, mlp_bias_2 = run_functions.mlp(post_ln_2_out)

            
            ## fp 5
            sync_comm()
            output_1 = run_functions.drop_residual(mlp_output_1+mlp_bias_1, post_ln_1_in)


            ## fp 6
            sync_comm()
            output_2 = run_functions.drop_residual(mlp_output_2+mlp_bias_2, post_ln_2_in)



        ctx.save_for_backward(input_1_clone, input_2_clone, \
                layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, output_1, \
                post_ln_2_in, post_ln_2_out, output_2)
        return output_1.detach(), output_2.detach()


    @staticmethod
    def backward(ctx, *grads):

        input_1, input_2, layernorm_output_1, layernorm_output_2, \
            ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
            layernorm_input_2, post_attn_ln_2, \
            post_ln_1_in, post_ln_1_out, output_1, \
            post_ln_2_in, post_ln_2_out, output_2 = ctx.saved_tensors

        output_1_grad, output_2_grad = grads

        ## bp1
        torch.autograd.backward(output_2, output_2_grad)

        ## bp2
        torch.autograd.backward(output_1, output_1_grad)

        ## bp3
        sync_comm()
        torch.autograd.backward((layernorm_input_2, post_attn_ln_2), (post_ln_2_in.grad, post_ln_2_out.grad))

        ## bp4
        sync_comm()
        torch.autograd.backward((layernorm_input_1, post_attn_ln_1), (post_ln_1_in.grad, post_ln_1_out.grad))

        ## bp5
        sync_comm()
        torch.autograd.backward(layernorm_output_2, ln2_out.grad)

        ## bp6
        sync_comm()
        mpu.mappings.ASYNC_OP = []
        torch.autograd.backward(layernorm_output_1, ln1_out.grad)

        return (None, None, None, input_1.grad, input_2.grad)




class CrossLayerPipedTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_functions, attention_mask, head_mask, input_1, input_2, *inputs):
        ctx.layer_states = (run_functions.is_first_layer, run_functions.is_last_layer)
        
        with torch.enable_grad():

            if run_functions.is_first_layer:
                ## fp 1
                # input_1_clone = input_1
                input_1_clone = input_1.detach()
                input_1_clone.requires_grad = True

                layernorm_output_1 = run_functions.input_layernorm(input_1_clone)
                
                # ln1_out = layernorm_output_1
                ln1_out = layernorm_output_1.detach()
                ln1_out.requires_grad = True

                attention_output_1, attention_output_1_bias = \
                    run_functions.self_attention(ln1_out, attention_mask=attention_mask[0], head_mask=head_mask)
                
                
                ## fp 2
                # input_2_clone = input_2
                input_2_clone = input_2.detach()
                input_2_clone.requires_grad = True

                layernorm_output_2 = run_functions.input_layernorm(input_2_clone)

                ln2_out = layernorm_output_2.detach()
                ln2_out.requires_grad = True

                attention_output_2, attention_output_2_bias = \
                    run_functions.self_attention(ln2_out, attention_mask=attention_mask[1], head_mask=head_mask)

            else:
                post_ln_1_in, post_ln_2_in = inputs
                ## last layer fp5 + fp1
                sync_comm()
                
                mlp_output_1 = input_1
                mlp_output_1_detached = mlp_output_1.detach()
                mlp_output_1_detached.requires_grad = True
                post_ln_1_in_detached = post_ln_1_in.detach()
                post_ln_1_in_detached.requires_grad = True

                input_1 = run_functions.drop_residual(mlp_output_1_detached, post_ln_1_in_detached)
                input_1_clone = input_1.detach()
                input_1_clone.requires_grad = True

                layernorm_output_1 = run_functions.input_layernorm(input_1_clone)
                
            
                ln1_out = layernorm_output_1.detach()
                ln1_out.requires_grad = True

                attention_output_1, attention_output_1_bias = \
                    run_functions.self_attention(ln1_out)


                ## last layer fp6 + fp2
                sync_comm()

                mlp_output_2 = input_2
                mlp_output_2_detached = mlp_output_2.detach()
                mlp_output_2_detached.requires_grad = True
                post_ln_2_in_detached = post_ln_2_in.detach()
                post_ln_2_in_detached.requires_grad = True

                input_2 = run_functions.drop_residual(mlp_output_2_detached, post_ln_2_in_detached)

                input_2_clone = input_2.detach()
                input_2_clone.requires_grad = True

                layernorm_output_2 = run_functions.input_layernorm(input_2_clone)

                ln2_out = layernorm_output_2.detach()
                ln2_out.requires_grad = True

                attention_output_2, attention_output_2_bias = \
                    run_functions.self_attention(ln2_out)


            ## fp 3
            sync_comm()
            layernorm_input_1 = run_functions.drop_residual(attention_output_1+attention_output_1_bias, input_1_clone)
            post_attn_ln_1 = run_functions.post_attention_layernorm(layernorm_input_1)

            # ln_1_in = post_attn_ln_1
            post_ln_1_in = layernorm_input_1.detach()
            post_ln_1_in.requires_grad = True
            post_ln_1_out = post_attn_ln_1.detach()
            post_ln_1_out.requires_grad = True

            mlp_output_1, mlp_bias_1 = run_functions.mlp(post_ln_1_out)

            ## fp 4
            sync_comm()
            layernorm_input_2 = run_functions.drop_residual(attention_output_2+attention_output_2_bias, input_2_clone)
            post_attn_ln_2 = run_functions.post_attention_layernorm(layernorm_input_2)

            # ln_2_in = post_attn_ln_2
            post_ln_2_in = layernorm_input_1.detach()
            post_ln_2_in.requires_grad = True
            post_ln_2_out = post_attn_ln_2.detach()
            post_ln_2_out.requires_grad = True
            
            mlp_output_2, mlp_bias_2 = run_functions.mlp(post_ln_2_out)
            if run_functions.is_last_layer:
                ## fp 5, only last layer
                sync_comm()
                output_1 = run_functions.drop_residual(mlp_output_1, post_ln_1_in)


                ## fp 6, only last layer
                sync_comm()
                output_2 = run_functions.drop_residual(mlp_output_2, post_ln_2_in)


        if run_functions.is_last_layer:
            ctx.save_for_backward(mlp_output_2_detached, post_ln_2_in_detached, \
                mlp_output_1_detached, post_ln_1_in_detached,\
                input_1_clone, input_2_clone, \
                layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, output_1, \
                post_ln_2_in, post_ln_2_out, output_2)
            return output_1.detach(), output_2.detach()
        elif run_functions.is_first_layer:
            ctx.save_for_backward(
                input_1_clone, input_2_clone, \
                layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, mlp_output_1, \
                post_ln_2_in, post_ln_2_out, mlp_output_2)
            return mlp_output_1.detach(), mlp_output_2.detach(), post_ln_1_in.detach(), post_ln_2_in.detach()
        
        else:
            ctx.save_for_backward(mlp_output_2_detached, post_ln_2_in_detached, \
                mlp_output_1_detached, post_ln_1_in_detached, \
                input_1_clone, input_2_clone, \
                layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, mlp_output_1, \
                post_ln_2_in, post_ln_2_out, mlp_output_2)
            return mlp_output_1.detach(), mlp_output_2.detach(), post_ln_1_in.detach(), post_ln_2_in.detach() #  output_1, output_2

    @staticmethod
    def backward(ctx, *grads):

        is_first_layer, is_last_layer = ctx.layer_states

        if is_first_layer:
            input_1, input_2, \
                layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, mlp_output_1, \
                post_ln_2_in, post_ln_2_out, mlp_output_2 = ctx.saved_tensors
        elif is_last_layer:
            mlp_output_2_detached, post_ln_2_in_detached, \
                mlp_output_1_detached, post_ln_1_in_detached,\
                input_1, input_2, layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, output_1, \
                post_ln_2_in, post_ln_2_out, output_2 = ctx.saved_tensors
        else:
            mlp_output_2_detached, post_ln_2_in_detached, \
                mlp_output_1_detached, post_ln_1_in_detached, \
                input_1, input_2, layernorm_output_1, layernorm_output_2, \
                ln1_out, ln2_out, layernorm_input_1, post_attn_ln_1,\
                layernorm_input_2, post_attn_ln_2, \
                post_ln_1_in, post_ln_1_out, mlp_output_1, \
                post_ln_2_in, post_ln_2_out, mlp_output_2 = ctx.saved_tensors

        if is_last_layer:

            output_1_grad, output_2_grad = grads
            ## bp1
            torch.autograd.backward(output_2, output_2_grad)

            ## bp2
            torch.autograd.backward(output_1, output_1_grad)

            ## bp3
            sync_comm()
            torch.autograd.backward((layernorm_input_2, post_attn_ln_2), (post_ln_2_in.grad, post_ln_2_out.grad))

            ## bp4
            sync_comm()
            torch.autograd.backward((layernorm_input_1, post_attn_ln_1), (post_ln_1_in.grad, post_ln_1_out.grad))

        else:

            mlp_output_1_grad, mlp_output_2_grad, post_ln_1_in_grad, post_ln_2_in_grad = grads
           
            ## bp1 - DR
            torch.autograd.backward(mlp_output_2, mlp_output_2_grad)


            ## bp2 - DR
            torch.autograd.backward(mlp_output_1, mlp_output_1_grad)

            ## bp3
            sync_comm()
            torch.autograd.backward((layernorm_input_2, post_attn_ln_2), (post_ln_2_in_grad, post_ln_2_out.grad))


            ## bp4
            sync_comm()
            torch.autograd.backward((layernorm_input_1, post_attn_ln_1), (post_ln_1_in_grad, post_ln_1_out.grad))


        if is_first_layer:
            ## bp5, only first layer
            sync_comm()
            torch.autograd.backward(layernorm_output_2, ln2_out.grad)

            ## bp6, only first layer
            sync_comm()
            mpu.mappings.ASYNC_OP = []
            torch.autograd.backward(layernorm_output_1, ln1_out.grad)
            return (None, None, None, input_1.grad, input_2.grad)

        else:
            ## bp5 + DR
            sync_comm()
            torch.autograd.backward(layernorm_output_2, ln2_out.grad)
            torch.autograd.backward(input_2, input_2.grad)

            ## bp6 + DR
            sync_comm()
            mpu.mappings.ASYNC_OP = []
            torch.autograd.backward(layernorm_output_1, ln1_out.grad)
            torch.autograd.backward(input_1, input_1.grad)

            return (None, None, None, \
                    mlp_output_1_detached.grad, mlp_output_2_detached.grad, \
                    post_ln_1_in_detached.grad, post_ln_2_in_detached.grad)


