'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
'''
# https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/utils.py
import os
import psutil
import gc

import torch
from torch._six import inf
import torch.distributed as dist

from ..utils import logger, log_dist
from math import sqrt

# pt-1.9 deprecations
if hasattr(torch.cuda, "memory_reserved"):
    torch_memory_reserved = torch.cuda.memory_reserved
else:
    torch_memory_reserved = torch.cuda.memory_allocated
if hasattr(torch.cuda, "max_memory_reserved"):
    torch_max_memory_reserved = torch.cuda.max_memory_reserved
else:
    torch_max_memory_reserved = torch.cuda.memory_cached

peak_memory = 0

def noop_decorator(func):
    return func


def ensure_directory_exists(filename):
    """Create the directory path to ``filename`` if it does not already exist.

    Args:
        filename (str): A file path.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def move_to_device(item, device):
    """
    Move tensor onto device. Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device

    Returns:
        None
    """
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    else:
        return item


def see_memory_usage(message, force=False, ram=False, ranks=[0], group=None):
    if not force:
        return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() in ranks:
        # torch.cuda.empty_cache()
        # torch.distributed.barrier(group=group)
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()
    global peak_memory
    max_ma = round(torch.cuda.max_memory_allocated() / (1024 * 1024),2)
    peak_memory = max(peak_memory, max_ma)
    # Print message except when distributed but not rank 0
    log_dist(message, ranks=ranks)
    log_dist(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024),2 )} MB \
        Max_MA {max_ma} MB \
        CA {round(torch_memory_reserved() / (1024 * 1024),2)} MB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024))} MB \
        PEAK_MA {peak_memory} MB    ", ranks=ranks)

    if ram:
        vm_stats = psutil.virtual_memory()
        used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
        logger.info(
            f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')
    
    # torch.cuda.empty_cache()
    # torch.distributed.barrier(group=group)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name

def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank()
                        == 0):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item()**norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    # Need to average total_norm across different GPUs due to the presence of moe params
    pg = mpu.get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))

    scaled_norm_tensor = torch.cuda.FloatTensor([float(scaled_norm)])
    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor.item()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    return sqrt(total_norm)

def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = mpu.get_model_parallel_rank()
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if tensor_mp_rank > 0:
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float(
            'inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''
    def __init__(self,
                 param_groups=None,
                 mpu=None,
                 zero_reduce_scatter=False,
                 deepspeed=None):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.deepspeed = deepspeed
        self.has_moe_params = False
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)

    def check_using_norm(self, norm_group, reduce_overflow=True):
        # TODO: I don't think reduce_overflow is needed if mpu is None
        overflow = -1 in norm_group
        overflow_gpu = torch.cuda.FloatTensor([overflow])
        if self.mpu is not None:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
        elif reduce_overflow:
            dist.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX)
            dist.barrier()
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        has_moe_params = False
        if param_groups is None:
            params = self.params
            has_moe_params = self.has_moe_params
        else:
            assert param_groups is not None, \
                "self.params and param_groups both cannot be none"

            for group in param_groups:
                for param in group:
                    params.append(param)

        return self.has_overflow(params, has_moe_params=has_moe_params)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params, has_moe_params=None):
        if has_moe_params is None:
            has_moe_params = self.has_moe_params
        overflow = self.has_overflow_serial(params)
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        overflow_gpu = torch.cuda.ByteTensor([overflow])
        # torch.distributed.all_reduce(overflow_gpu,
        #                             op=torch.distributed.ReduceOp.MAX,
        #                             group=mpu.get_model_parallel_group())
        if self.zero_reduce_scatter:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=torch.distributed.group.WORLD)
        elif self.mpu is not None:
            if self.deepspeed is not None:
                using_pipeline = hasattr(self.deepspeed,
                                         'pipeline_enable_backward_allreduce')
                if (using_pipeline
                        and self.deepspeed.pipeline_enable_backward_allreduce is False
                    ) or (not using_pipeline
                          and self.deepspeed.enable_backward_allreduce is False):
                    torch.distributed.all_reduce(
                        overflow_gpu,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self.mpu.get_data_parallel_group())
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.mpu.get_model_parallel_group())
        elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=torch.distributed.group.WORLD)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, i):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False