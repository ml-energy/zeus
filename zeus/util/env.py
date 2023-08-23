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

"""Tools related to environment variables."""

from __future__ import annotations

import os
from typing import Type, TypeVar, cast

import pynvml

T = TypeVar("T")


def get_env(name: str, valtype: Type[T], default: T | None = None) -> T:
    """Fetch an environment variable and cast it to the given type."""
    try:
        if valtype == bool:
            val = os.environ[name].lower()
            if val not in ["true", "false"]:
                raise ValueError(f"Strange boolean environment variable value '{val}'")
            return cast(T, val == "true")
        return valtype(os.environ[name])
    except KeyError:
        if default is not None:
            return default
        raise ValueError(f"Missing environment variable '{name}'") from None


def resolve_gpu_indices(
    requested_gpu_indices: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Resolve GPU indices considering `CUDA_VISIBLE_DEVICES`.

    Args:
        requested_gpu_indices: A list of user-specified GPU indices. If `None`,
            assume the user wants all GPUs visible under `CUDA_VISIBLE_DEVICES`.

    Returns:
        A tuple of GPU index lists, where the former is CUDA indices under the
            illusion of `CUDA_VISIBLE_DEVICES` and the latter is the actual CUDA
            indices that NVML understands. The order of the two lists are the same.
    """
    # Initialize NVML.
    pynvml.nvmlInit()

    # Sanity check.
    if requested_gpu_indices is not None and not requested_gpu_indices:
        raise ValueError("`requested_gpu_indices` must be None or non-empty.")

    # Find the NVML GPU indices visible to CUDA, respecting `CUDA_VISIBLE_DEVICES`.
    if (cuda_visible_device := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
        nvml_visible_indices = [int(idx) for idx in cuda_visible_device.split(",")]
    else:
        nvml_visible_indices = list(range(pynvml.nvmlDeviceGetCount()))

    # NVML GPU indices and CUDA GPU indices should be different.
    # We always use CUDA GPU indices when communicating with the outside world,
    # but when dealing with NVML, we use the NVML GPU indices.
    if requested_gpu_indices is None:
        nvml_gpu_indices = nvml_visible_indices
        cuda_gpu_indices = list(range(len(nvml_visible_indices)))
    else:
        nvml_gpu_indices = [nvml_visible_indices[idx] for idx in requested_gpu_indices]
        cuda_gpu_indices = requested_gpu_indices

    # Deinitialize NVML.
    pynvml.nvmlShutdown()

    return cuda_gpu_indices, nvml_gpu_indices
