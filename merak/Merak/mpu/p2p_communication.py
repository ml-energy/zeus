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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/p2p_communication.py

from functools import reduce
import operator
import torch
from .. import mpu


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 use_ring_exchange=False, recv_next_buffer=None,
                 recv_prev_buffer=None):
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.)
    """
    tensor_recv_prev = None
    tensor_recv_next = None
    if recv_prev:
        assert recv_prev_buffer is not None, 'recv_prev_buffer is None'
        tensor_recv_prev = recv_prev_buffer
    if recv_next:
        assert recv_next_buffer is not None, 'recv_next_buffer is None'
        tensor_recv_next = recv_next_buffer

    # Send tensors in both the forward and backward directions as appropriate.
    if use_ring_exchange:
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                        tensor_recv_prev=tensor_recv_prev,
                                        tensor_send_next=tensor_send_next,
                                        tensor_recv_next=tensor_recv_next,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(send_next_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()




def recv_forward(buffer):
    """Receive tensor from previous rank in pipeline (forward receive)."""
    if mpu.is_pipeline_first_stage():
        buffer = None
    else:
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            recv_prev_buffer=buffer)


def recv_backward(buffer):
    """Receive tensor from next rank in pipeline (backward receive)."""
    if mpu.is_pipeline_last_stage():
        buffer = None
    else:
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            recv_next_buffer=buffer)


def send_forward(output_tensor):
    """Send tensor to next rank in pipeline (forward send)."""
    if not mpu.is_pipeline_last_stage():
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False)


def send_backward(input_tensor_grad):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False)


def send_forward_recv_backward(output_tensor, recv_next_buffer):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        recv_next_buffer = None
    else:
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True, 
            recv_next_buffer=recv_next_buffer)

def send_backward_recv_forward(input_tensor_grad, recv_prev_buffer):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        recv_prev_buffer = None
    else:
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False, 
            recv_prev_buffer=recv_prev_buffer)



def send_forward_recv_forward(output_tensor, recv_prev, recv_prev_buffer=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False, 
        recv_prev_buffer=recv_prev_buffer)

def send_backward_recv_backward(input_tensor_grad, recv_next, recv_next_buffer=None):
    """Batched recv from next rank and send to previous rank in pipeline."""
    _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next, 
        recv_next_buffer=recv_next_buffer)

def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next,recv_prev_buffer=None, recv_next_buffer=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next, 
        recv_prev_buffer=recv_prev_buffer, 
        recv_next_buffer=recv_next_buffer)