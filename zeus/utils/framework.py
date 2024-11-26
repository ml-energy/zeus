"""Utilities for framework-specific code."""

from __future__ import annotations

import types
from typing import Literal, Any, List
from functools import lru_cache

from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}


@lru_cache(maxsize=1)
def torch_is_available(ensure_available: bool = False, ensure_cuda: bool = True):
    """Check if PyTorch is available."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if ensure_cuda and not cuda_available:
            raise RuntimeError("PyTorch is available but does not have CUDA support.")
        MODULE_CACHE["torch"] = torch
        logger.info(
            "PyTorch %s CUDA support is available.",
            "with" if cuda_available else "without",
        )
        return True
    except ImportError as e:
        logger.info("PyTorch is not available.")
        if ensure_available:
            raise RuntimeError("Failed to import Pytorch") from e
        return False


@lru_cache(maxsize=1)
def jax_is_available(ensure_available: bool = False, ensure_cuda: bool = True):
    """Check if JAX is available."""
    try:
        import jax  # type: ignore

        cuda_available = jax.devices("gpu")
        if ensure_cuda and not cuda_available:
            raise RuntimeError("JAX is available but does not have CUDA support.")
        MODULE_CACHE["jax"] = jax
        logger.info(
            "JAX %s CUDA support is available.", "with" if cuda_available else "without"
        )
        return True
    except ImportError as e:
        logger.info("JAX is not available")
        if ensure_available:
            raise RuntimeError("Failed to import JAX") from e
        return False


def sync_execution(
    gpu_devices: list[int], sync_with: Literal["torch", "jax"] = "torch"
) -> None:
    """Block until all computations on the specified devices are finished.

    PyTorch only runs GPU computations asynchronously, so synchronizing computations
    for the given GPU devices is done by calling `torch.cuda.synchronize` on each
    device. On the other hand, JAX runs both CPU and GPU computations asynchronously,
    but by default it only has a single CPU device (id=0). Therefore for JAX, all GPU
    devices passed in and the CPU device (id=0) are synchronized.

    !!! Note
        `jax.device_put` with `block_until_ready` is used to synchronize computations
        on JAX devices. This is a workaround to the lack of a direct API for
        synchronizing computations on JAX devices. Tracking issue:
        https://github.com/google/jax/issues/4335

    !!! Note
        Across the Zeus library, an integer device index corresponds to a single whole
        physical device. This is usually what you want, except when using more advanced
        device partitioning (e.g., using `--xla_force_host_platform_device_count` in JAX
        to partition CPUs into more pieces). In such cases, you probably want to opt out
        from using this function and handle synchronization manually at the appropriate
        granularity.

    Args:
        gpu_devices: GPU device indices to synchronize.
        sync_with: Deep learning framework to use to synchronize computations.
            Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
    """
    if sync_with == "torch" and torch_is_available(ensure_available=True):
        torch = MODULE_CACHE["torch"]
        for device in gpu_devices:
            torch.cuda.synchronize(device)
        return

    if sync_with == "jax" and jax_is_available(ensure_available=True):
        jax = MODULE_CACHE["jax"]
        futures = [
            jax.device_put(0.0, device=jax.devices("gpu")[device]) + 0
            for device in gpu_devices
        ]
        futures.append(jax.device_put(0.0, device=jax.devices("cpu")[0]) + 0)
        jax.block_until_ready(futures)
        return

    raise RuntimeError("No framework is available.")

def all_gather(object: Any) -> List[Any]:
    """Gather objects from all replicas."""

    if torch_is_available():
        torch = MODULE_CACHE["torch"]
        world_size = torch.distributed.get_world_size()
        # If there is only one process (not distributed), return the object as is
        if world_size == 1:
            return [object]
        gathered_objects = [None] * world_size
        torch.distributed.all_gather(gathered_objects, object)
        return gathered_objects

    # TODO: FIX JAX
    if jax_is_available():
        jax = MODULE_CACHE["jax"]
        n_devices = jax.local_device_count()
        if n_devices == 1:
            return [object]

        gathered_objects = [object] * n_devices
        return gathered_objects

    raise RuntimeError("No framework is available.")

# TODO: fix type hints
# def all_reduce_sum(object: List[int] | List[float] | int | float) -> int | float:
#     """Reduce objects from all replicas by summing them."""
#     if torch_is_available():
#         torch = MODULE_CACHE["torch"]
#         # wrap object in a tensor if it is not already
#         if not isinstance(object, torch.Tensor):
#             object = torch.Tensor(object)
#         # compute local sum
#         sum = torch.sum(object)

#         # all-reduce only if torch.distributed is available and initialized
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(sum, op=torch.distributed.ReduceOp.SUM)
#         return sum.item()
#     if jax_is_available():
#         raise NotImplementedError("JAX is not supported yet.")

# # TODO: fix type hints
# def all_reduce_max(object: List[int] | List[float] | int | float) -> int | float:
#     """Reduce objects from all replicas by taking the maximum."""
#     if torch_is_available():
#         torch = MODULE_CACHE["torch"]
#         # wrap object in a tensor if it is not already
#         if not isinstance(object, torch.Tensor):
#             object = torch.Tensor(object)
#         # compute local max
#         max = torch.max(object)
        
#         # all-reduce only if torch.distributed is available and initialized
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(max, op=torch.distributed.ReduceOp.MAX)
#         return max.item()
#     if jax_is_available():
#         raise NotImplementedError("JAX is not supported yet.")
    
def all_reduce(object: List[int] | List[float], operation: Literal["sum", "max"]) -> int | float:
    """Reduce objects from all replicas through the specified operation.
    
    If running in a distributed setting, the objects are reduced across all replicas. 
    If running in a non-distributed setting, the operation is just done on the single object."""
    
    if torch_is_available():
        torch = MODULE_CACHE["torch"]

        # wrap object in a tensor if it is not already
        if not isinstance(object, torch.Tensor):
            object = torch.Tensor(object)

        # determine operation
        if operation == "sum":
            torch_func = torch.sum
            torch_op = torch.distributed.ReduceOp.SUM
        elif operation == "max":
            torch_func = torch.max
            torch_op = torch.distributed.ReduceOp.MAX
        else:
            raise ValueError(f"all_reduce unsupported operation: {operation}")

        # compute local operation
        result = torch_func(object)

        # all-reduce only if torch.distributed is available and initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result.cuda(), op=torch_op)
        return result.item()

    if jax_is_available():
        raise NotImplementedError("JAX is not supported yet.")
    
    raise RuntimeError("No framework is available.")