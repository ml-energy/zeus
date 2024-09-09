"""Collect information about the environment and display it.

- Python version
- Package availablility and versions: Zeus, PyTorch, JAX.
- NVIDIA GPU availability: Number of GPUs and moels.
- AMD GPU availability: Number of GPUs and models.
- Intel RAPL availability: Number of CPUs and whether DRAM measurements are available.
"""

from __future__ import annotations

import platform

import zeus
from zeus.utils import framework
from zeus.device import get_gpus, get_cpus
from zeus.device.cpu import RAPLCPUs
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, EmptyCPUs


SECTION_SEPARATOR = "=" * 80 + "\n"


def show_env():
    """Collect information about the environment and display it."""
    print(SECTION_SEPARATOR)
    print(f"Python version: {platform.python_version()}\n")

    print(SECTION_SEPARATOR)
    package_availability = "\nPackage availability and versions:\n"
    package_availability += f"  Zeus: {zeus.__version__}\n"

    try:
        torch_available = framework.torch_is_available()
        torch_cuda_available = True
    except RuntimeError:
        torch_available = framework.torch_is_available(ensure_cuda=False)
        torch_cuda_available = False

    if torch_available and torch_cuda_available:
        torch = framework.MODULE_CACHE["torch"]
        package_availability += f"  PyTorch: {torch.__version__} (with CUDA support)\n"
    elif torch_available and not torch_cuda_available:
        torch = framework.MODULE_CACHE["torch"]
        package_availability += (
            f"  PyTorch: {torch.__version__} (without CUDA support)\n"
        )
    else:
        package_availability += "  PyTorch: not available\n"

    try:
        jax_available = framework.jax_is_available()
        jax_cuda_available = True
    except RuntimeError:
        jax_available = framework.jax_is_available(ensure_cuda=False)
        jax_cuda_available = False

    if jax_available and jax_cuda_available:
        jax = framework.MODULE_CACHE["jax"]
        package_availability += f"  JAX: {jax.__version__} (with CUDA support)\n"
    elif jax_available and not jax_cuda_available:
        jax = framework.MODULE_CACHE["jax"]
        package_availability += f"  JAX: {jax.__version__} (without CUDA support)\n"
    else:
        package_availability += "  JAX: not available\n"

    print(package_availability)

    print(SECTION_SEPARATOR)
    gpu_availability = "\nGPU availability:\n"
    try:
        gpus = get_gpus()
    except ZeusGPUInitError:
        gpus = EmptyGPUs()
    if len(gpus) > 0:
        for i in range(len(gpus)):
            gpu_availability += f"  GPU {i}: {gpus.getName(i)}\n"
    else:
        gpu_availability += "  No GPUs available.\n"
    print(gpu_availability)

    print(SECTION_SEPARATOR)
    cpu_availability = "\nCPU availability:\n"
    try:
        cpus = get_cpus()
    except ZeusCPUInitError:
        cpus = EmptyCPUs()
    if len(cpus) > 0:
        assert isinstance(cpus, RAPLCPUs)
        for i in range(len(cpus)):
            cpu_availability += f"  CPU {i}:\n    CPU measurements available ({cpus.cpus[i].rapl_file.path})\n"
            if cpus.supportsGetDramEnergyConsumption(i):
                dram = cpus.cpus[i].dram
                assert dram is not None
                cpu_availability += f"    DRAM measurements available ({dram.path})\n"
            else:
                cpu_availability += "    DRAM measurements unavailable\n"
    else:
        cpu_availability += "  No CPUs available.\n"
    print(cpu_availability)

    print(SECTION_SEPARATOR)


if __name__ == "__main__":
    show_env()
