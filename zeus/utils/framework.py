# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
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

"""Utilities for framework-specific code."""

from __future__ import annotations

import types
from typing import Literal
from functools import lru_cache

from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}


@lru_cache(maxsize=1)
def torch_is_available(ensure_available: bool = False):
    """Check if PyTorch is available."""
    try:
        import torch

        assert (
            torch.cuda.is_available()
        ), "PyTorch is available but does not have CUDA support."
        MODULE_CACHE["torch"] = torch
        logger.info("PyTorch with CUDA support is available.")
        return True
    except ImportError as e:
        logger.info("PyTorch is not available.")
        if ensure_available:
            raise RuntimeError("Failed to import Pytorch") from e
        return False


@lru_cache(maxsize=1)
def jax_is_available(ensure_available: bool = False):
    """Check if JAX is available."""
    try:
        import jax  # type: ignore

        assert jax.devices("gpu"), "JAX is available but does not have CUDA support."
        MODULE_CACHE["jax"] = jax
        logger.info("JAX with CUDA support is available.")
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
