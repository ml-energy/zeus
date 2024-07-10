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
from functools import lru_cache

from zeus.utils.logging import get_logger

logger = get_logger(name=__name__)
MODULE_CACHE: dict[str, types.ModuleType] = {}


@lru_cache(maxsize=1)
def torch_is_available():
    """Check if PyTorch is available."""
    try:
        import torch

        assert (
            torch.cuda.is_available()
        ), "PyTorch is available but does not have CUDA support."
        MODULE_CACHE["torch"] = torch
        logger.info("PyTorch with CUDA support is available.")
        return True
    except ImportError:
        logger.info("PyTorch is not available.")
        return False


@lru_cache(maxsize=1)
def jax_is_available():
    """Check if JAX is available."""
    try:
        import jax

        assert jax.devices("gpu"), "JAX is available but does not have CUDA support."
        MODULE_CACHE["jax"] = jax
        logger.info("JAX with CUDA support is available.")
        return True
    except ImportError:
        logger.info("JAX is not available")
        return False


def cuda_sync(device: int | None = None, backend: str = "torch") -> None:
    """Synchronize CPU with CUDA.

    Note: `cupy.cuda.Device.synchronize` may be a good choice to make
          CUDA device synchronization more general. Haven't tested it yet.

    Args:
        device: The device to synchronize.
        backend: The backend framework. Defaults to `torch`
    """
    if backend == "torch" and torch_is_available():
        torch = MODULE_CACHE["torch"]

        def synchronize_cuda_fn(device):
            torch.cuda.synchronize(device)

    elif backend == "jax" and jax_is_available():
        jax = MODULE_CACHE["jax"]

        def synchronize_cuda_fn(device):
            (
                jax.device_put(0.0, device=jax.devices("gpu")[device]) + 0
            ).block_until_ready()

    else:
        raise RuntimeError("No framework is available.")

    synchronize_cuda_fn(device)
