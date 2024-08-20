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

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/b886b7bb972afe72bac0f5de4f42a4a7bae8ebef/megatron/mpu/random.py
# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/activation_checkpointing/checkpointing.py

import copy
import torch
import contextlib
import torch.distributed as dist

import mmap
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from ..utils import logger
from .utils import move_to_device, see_memory_usage
from .. import mpu, print_rank_0


# MP parameters
mp_rank = None
mp_size = None
mp_group = None


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
transport_stream = None
cuda_device = None


def detach_variable(inputs, device=None):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            requires_grad = inp.requires_grad

            if device is not None:
                x = inp.to(device=device)
            else:
                x = inp

            x = x.detach()
            x.requires_grad = requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__)


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """
    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        return copy.copy(self.states_)

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + mpu.get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        logger.info(
            '> initializing model parallel cuda seeds on global rank {}, '
            'model parallel rank {}, and data parallel rank {} with '
            'model parallel seed: {} and data parallel seed: {}'.format(
                torch.distributed.get_rank(),
                mpu.get_model_parallel_rank(),
                mpu.get_data_parallel_rank(),
                model_parallel_seed,
                data_parallel_seed),
        )
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, model_parallel_seed)


def get_partition_start(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    partition_size = size / mp_size
    start = partition_size * mp_rank
    return int(start)


def get_partition_size(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    assert size % mp_size == 0, "Doesn't handle if partition activation if item is not divisible by mp size"
    partition_size = size / mp_size
    return int(partition_size)


def get_full_inputs(tensors, device=None):
    inputs = []
    num_args = int(len(tensors) / 2)
    for i in range(num_args - 1):

        item = tensors[2 * i]
        size = tensors[2 * i + 1]

        partition_size = item.numel()
        tensor_size = partition_size * mp_size
        if device is not None:
            flat_tensor = torch.zeros([tensor_size], dtype=item.dtype, device=device)
        else:
            flat_tensor = torch.zeros([tensor_size],
                                      dtype=item.dtype,
                                      device=item.device)
        partitions = []
        for i in range(mp_size):
            part_i = flat_tensor.narrow(0, partition_size * i, partition_size)
            if i == mp_rank:
                part_i.copy_(item)
            partitions.append(part_i)
        if mp_group is not None:
            dist.all_gather(partitions, partitions[mp_rank], group=mp_group)
        input_tensor = flat_tensor.view(list(size.numpy()))
        item.data = input_tensor.data

        inputs.append(item)
    inputs.append(tensors[-2])

    return tuple(inputs)


def extract_tensors(all_objects):
    """
    Separate objects in list/tuple into tensors and non-tensors and create a mapping to enable re-aggregation.
    The order of tensors and non-tensors is preserved in their respective output groups.

    Parameters:
        all_objects (list/tuple): Objects containing tensors and non-tensors to be split.

    Returns:
        tuple: Containing tensors, non-tensors, and bools of whether each position in original list/tuple was a tensor.

    """
    tensor_objects = [v for v in all_objects if torch.is_tensor(v)]
    non_tensor_objects = [v for v in all_objects if not torch.is_tensor(v)]
    tensor_flags = [torch.is_tensor(v) for v in all_objects]
    if type(all_objects) is tuple:
        return tuple(tensor_objects), tuple(non_tensor_objects), tuple(tensor_flags)
    return tensor_objects, non_tensor_objects, tensor_flags


def merge_tensors(tensor_objects, non_tensor_objects, tensor_flags):
    """
    Merge two lists (or tuples) of tensors and non-tensors using a mapping of positions in merged list (or tuple).

    Parameters:
        tensor_objects (list/tuple): Tensors to merge.
        non_tensor_objects (list/tuple): Non-tensors to merge.
        tensor_flags (list/tuple): Indicates whether each position in output is a tensor.

    Returns:
        tuple: Merge of tensors and non-tensors
    """
    merged_objects = []
    tensor_idx = 0
    non_tensor_idx = 0

    real_tensor_flags = tensor_flags

    for is_tensor in real_tensor_flags:
        if is_tensor:
            merged_objects.append(tensor_objects[tensor_idx])
            tensor_idx += 1
        else:
            merged_objects.append(non_tensor_objects[non_tensor_idx])
            non_tensor_idx += 1

    return tuple(merged_objects)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
           3) Performance activation partitioning, contiguous memory optimization
           4) CPU Checkpointing
           5) Profile forward and backward functions
    """
    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):

        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(all_objects=all_args)
            ctx.save_for_backward(*tensor_args)
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags


        ctx.run_function = run_function
        global mp_rank, mp_size, mp_group

        if mp_rank is None:
            mp_rank = mpu.get_model_parallel_rank()
            mp_size = mpu.get_model_parallel_world_size()
            mp_group = mpu.get_model_parallel_group()

        global cuda_device, transport_stream

        if cuda_device is None:
            see_memory_usage("First Forward Begining", force=False)
            cuda_device = torch.cuda.current_device()
            transport_stream = torch.cuda.Stream(device=cuda_device)

        #just in case something funky is happening such as reuse of inputs
        inputs_cuda = move_to_device(args, cuda_device)

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        see_memory_usage("Before running forward on the layer", force=False)
        # ctx.save_for_backward(*args)
        # print_rank_0('inputs_cuda:')
        # print_rank_0([ [j.requires_grad for j in i ] if isinstance(i, tuple) else i.requires_grad for i in inputs_cuda])

        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        see_memory_usage("After running forward on the layer", force=False)
        del inputs_cuda

        save_args_for_backward(*args)

        # Tensors returned from forward() may not be differentiable.
        if torch.is_tensor(outputs):
            non_grad_outputs = [outputs] if not outputs.is_floating_point() else []
        else:
            non_grad_outputs = [
                o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()
            ]
        ctx.mark_non_differentiable(*non_grad_outputs)

        ## will set requires_grad = True if output is returned???

        if torch.is_tensor(outputs):
            all_outputs += [outputs]
            # print(all_outputs)
            return outputs
        else:
            all_outputs += outputs
            outputs, _, _ = extract_tensors(all_objects=outputs)
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        see_memory_usage("In backward", force=False)
        # removing pointers to the contiguous buffer memory
        # so that they can be garbage collected once the checkpoints
        # have been used

        see_memory_usage("In backward checkpointing code", force=False)
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")

        global cuda_device, transport_stream

        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)

        # Add non tensor input args
        detached_inputs = merge_tensors(tensor_objects=detached_inputs,
                                        non_tensor_objects=ctx.non_tensor_args,
                                        tensor_flags=ctx.tensor_flags)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        see_memory_usage("In backward checkpointing code before forward", force=False)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        see_memory_usage("In backward checkpointing code after forward", force=False)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )

        # Filter out non tensor outputs
        outputs, _, _ = extract_tensors(all_objects=outputs)

        # Construct arguments to autograd.backward().
        # This is usually just outputs and grads, but forward() can return tensors that
        # are not differentiable.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)

        see_memory_usage("In backward checkpointing code before backward", force=False)

        torch.autograd.backward(output_tensors, grad_tensors)

        see_memory_usage("After backward checkpointing code after backward", force=False)

        ret_list = [None, None]  # first None for ctx
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        return tuple(ret_list)


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint. """

    all_outputs = []
    CheckpointFunction.apply(function, all_outputs, *args)
    # print('after apply',all_outputs[0].device, all_outputs[0].is_leaf)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)


class PreCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):

        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(all_objects=all_args)
            ctx.save_for_backward(*tensor_args)
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags

        ctx.run_function = run_function

        cuda_device = torch.cuda.current_device()

        #just in case something funky is happening such as reuse of inputs
        inputs_cuda = move_to_device(args, cuda_device)

        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        del inputs_cuda

        save_args_for_backward(*args)

        # Tensors returned from forward() may not be differentiable.
        if torch.is_tensor(outputs):
            non_grad_outputs = [outputs] if not outputs.is_floating_point() else []
        else:
            non_grad_outputs = [
                o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()
            ]
        ctx.mark_non_differentiable(*non_grad_outputs)
        # print(non_grad_outputs)
        if torch.is_tensor(outputs):
            all_outputs += [outputs]
            # print(all_outputs)
            return outputs
        else:
            all_outputs += outputs
            outputs, _, _ = extract_tensors(all_objects=outputs)
            return tuple(outputs)

    @staticmethod
    def pre_recompute(ctx):

        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)

        # Add non tensor input args
        detached_inputs = merge_tensors(tensor_objects=detached_inputs,
                                        non_tensor_objects=ctx.non_tensor_args,
                                        tensor_flags=ctx.tensor_flags)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)


        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )

        # Filter out non tensor outputs
        ctx.outputs, _, _ = extract_tensors(all_objects=outputs)
        ctx.detached_inputs = detached_inputs

        
    @staticmethod
    def backward(ctx, *grads):

        # removing pointers to the contiguous buffer memory
        # so that they can be garbage collected once the checkpoints
        # have been used

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")

        # print(torch.distributed.get_rank(), 'recompute in backward')
        PreCheckpointFunction.pre_recompute(ctx)
        outputs = ctx.outputs
        detached_inputs = ctx.detached_inputs

        # Construct arguments to autograd.backward().
        # This is usually just outputs and grads, but forward() can return tensors that
        # are not differentiable.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)

        torch.autograd.backward(output_tensors, grad_tensors)

        ret_list = [None, None]  # first None for ctx
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        return tuple(ret_list)


def pre_checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint. """

    all_outputs = []
    PreCheckpointFunction.apply(function, all_outputs, *args)
    # print('after apply',all_outputs[0].device, all_outputs[0].is_leaf)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)


class RNGManager:
    def __init__(self) -> None:
        self.state_dic = {}

    def store_fwd_rng_state(self, buffer_id):
        fwd_cpu_rng_state = torch.get_rng_state()
        fwd_cuda_rng_state = torch.cuda.get_rng_state()
        fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        self.state_dic[buffer_id] = (fwd_cpu_rng_state, fwd_cuda_rng_state, fwd_cuda_rng_state_tracker)
    
    def set_recompute_rng_state(self, buffer_id):
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        fwd_cpu_rng_state, fwd_cuda_rng_state, fwd_cuda_rng_state_tracker = self.state_dic[buffer_id]
        torch.set_rng_state(fwd_cpu_rng_state)
        _set_cuda_rng_state(fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(fwd_cuda_rng_state_tracker)

        self.state_dic[buffer_id] = (bwd_cpu_rng_state, bwd_cuda_rng_state, bwd_cuda_rng_state_tracker)
    


    def restore_bwd_rng_state(self, buffer_id):
        bwd_cpu_rng_state, bwd_cuda_rng_state, bwd_cuda_rng_state_tracker = self.state_dic[buffer_id]
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)
