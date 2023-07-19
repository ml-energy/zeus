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

from zeus.util.logging import get_logger

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


def cuda_sync(device: int | None = None) -> None:
    """Synchronize CPU and CUDA.

    Note: `cupy.cuda.Device.synchronize` may be a good choice to make
          CUDA device synchronization more general. Haven't tested it yet.

    Args:
        device: The device to synchronize.
    """
    if torch_is_available():
        torch = MODULE_CACHE["torch"]
        torch.cuda.synchronize(device)
        return

    raise RuntimeError("No frameworks are available.")
