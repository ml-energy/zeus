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


def cuda_sync(
    device: int | None = None, sync_cuda_with: Literal["torch", "jax"] = "torch"
) -> None:
    """Synchronize CPU with CUDA.

    Note: `cupy.cuda.Device.synchronize` may be a good choice to make
          CUDA device synchronization more general. Haven't tested it yet.

    Args:
        device: The device to synchronize.
        sync_cuda_with: Deep learning framework to use to synchronize GPU computations.
            Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
    """
    if sync_cuda_with == "torch" and torch_is_available(ensure_available=True):
        torch = MODULE_CACHE["torch"]

        torch.cuda.synchronize(device)

    elif sync_cuda_with == "jax" and jax_is_available(ensure_available=True):
        jax = MODULE_CACHE["jax"]

        (
            jax.device_put(
                0.0, device=None if device is None else jax.devices("gpu")[device]
            )
            + 0
        ).block_until_ready()

    else:
        raise RuntimeError("No framework is available.")
