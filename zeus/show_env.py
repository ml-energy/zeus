"""Collect information about the environment and display it.

- Python version
- Package availability and versions: Zeus, PyTorch, JAX, CuPy.
- NVIDIA GPU availability: Number of GPUs and models.
- AMD GPU availability: Number of GPUs and models.
- Intel RAPL availability: Number of CPUs and whether DRAM measurements are available.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil

import zeus
from zeus.device.cpu.rapl import RAPLCPU, ZeusdRAPLCPU
from zeus.device.exception import ZeusBaseCPUError, ZeusBaseGPUError, ZeusBaseSoCError
from zeus.utils import framework
from zeus.device import get_gpus, get_cpus, get_soc
from zeus.device.cpu import RAPLCPUs
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, EmptyCPUs
from zeus.device.soc.common import ZeusSoCInitError, EmptySoC
from zeus.device.soc.apple import AppleSilicon
from zeus.device.soc.jetson import Jetson

SECTION_SEPARATOR = "-" * shutil.get_terminal_size().columns


def show_env():
    """Collect information about the environment and display it."""
    print(SECTION_SEPARATOR)
    print("## Package availability and versions\n")
    print("Logging output:")

    package_availability = "  Python version: " + platform.python_version() + "\n"
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
        package_availability += f"  PyTorch: {torch.__version__} (without CUDA support)\n"
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

    cupy_available = framework.cupy_is_available()
    if cupy_available:
        cupy = framework.MODULE_CACHE["cupy"]
        package_availability += f"  CuPy: {cupy.__version__}\n"
    else:
        package_availability += "  CuPy: not available\n"

    print("\nDetected:\n" + package_availability)

    print(SECTION_SEPARATOR)
    print("## Device availability\n")
    print("Logging output:")

    gpu_availability = ""
    try:
        gpus = get_gpus()
    except ZeusGPUInitError:
        gpus = EmptyGPUs()
    except ZeusBaseGPUError as e:
        gpu_availability += f"  Error initializing GPUs: {e}\n"
        gpus = EmptyGPUs()

    if len(gpus) > 0:
        # Check for visibility environment variables
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "[NOT SET]")
        hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "[NOT SET]")
        gpu_availability += f"  CUDA_VISIBLE_DEVICES={cuda_visible_devices}\n"
        gpu_availability += f"  HIP_VISIBLE_DEVICES={hip_visible_devices}\n"

        # Use Zeus's internal state to get the physical GPU indices
        physical_indices = [gpu.gpu_index for gpu in gpus.gpus]

        # Show device index mapping if env var restricts visibility
        gpu_availability += "  Device index mapping: Physical (e.g., nvidia-smi) -> Zeus (or any CUDA application)\n"
        for zeus_idx, physical_idx in enumerate(physical_indices):
            gpu_availability += f"    GPU {physical_idx} -> GPU {zeus_idx} ({gpus.get_name(zeus_idx)})\n"
    else:
        gpu_availability += "  No GPUs available.\n"
    print("\nDetected:\n" + gpu_availability)

    print(SECTION_SEPARATOR)
    print("## CPU availability\n")
    print("Logging output:")

    cpu_availability = ""
    try:
        cpus = get_cpus()
    except ZeusCPUInitError:
        cpus = EmptyCPUs()
    except ZeusBaseCPUError as e:
        cpu_availability += f"  Error initializing CPUs: {e}\n"
        cpus = EmptyCPUs()

    if len(cpus) > 0:
        assert isinstance(cpus, RAPLCPUs)
        for i, cpu in enumerate(cpus.cpus):
            if isinstance(cpu, ZeusdRAPLCPU):
                cpu_availability += f"  CPU {i}:\n    CPU measurements available (Zeusd at {cpu.zeusd_sock_path})\n"
                if cpu.supports_get_dram_energy_consumption():
                    cpu_availability += f"    DRAM measurements available (Zeusd at {cpu.zeusd_sock_path})\n"
                else:
                    cpu_availability += "    DRAM measurements unavailable\n"
            elif isinstance(cpu, RAPLCPU):
                cpu_availability += f"  CPU {i}:\n    CPU measurements available ({cpu.rapl_file.path})\n"
                if cpu.supports_get_dram_energy_consumption():
                    dram = cpu.dram
                    assert dram is not None
                    cpu_availability += f"    DRAM measurements available ({dram.path})\n"
                else:
                    cpu_availability += "    DRAM measurements unavailable\n"
            else:
                raise TypeError("Unexpected CPU type: " + str(type(cpu)))
    else:
        cpu_availability += "  No CPUs available.\n"
    print("\nDetected:\n" + cpu_availability)

    print(SECTION_SEPARATOR)
    print("## SoC availability\n")
    print("Logging output:")

    soc_availability = ""
    try:
        soc = get_soc()
    except ZeusSoCInitError:
        soc = EmptySoC()
    except ZeusBaseSoCError as e:
        soc_availability += f"  Error initializing SoC: {e}\n"
        soc = EmptySoC()

    if isinstance(soc, AppleSilicon):
        metrics = soc.get_available_metrics()
        soc_availability += "  Apple Silicon SoC available.\n"
        soc_availability += f"  Available metrics: {', '.join(metrics)}\n"
    elif isinstance(soc, Jetson):
        metrics = soc.get_available_metrics()
        soc_availability += "  NVIDIA Jetson SoC available.\n"
        soc_availability += f"  Available metrics: {', '.join(metrics)}\n"
    else:
        soc_availability += "  No SoC available.\n"

        # If this is an Apple Silicon device but AppleSilicon was not detected,
        # print out a warning.
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            soc_availability += (
                "\nThis appears to be an Apple Silicon device, but it wasn't picked up by Zeus.\n"
                "Have you installed Zeus with the `apple` extra dependencies?\n"
                "    pip install 'zeus[apple]'\n"
            )
    print("\nDetected:\n" + soc_availability)

    print(SECTION_SEPARATOR)


if __name__ == "__main__":
    # For the `python -m zeus.show_env` usage
    logging.basicConfig(level=logging.INFO, format="  [%(asctime)s] [%(name)s:%(lineno)d] %(message)s")

    show_env()
