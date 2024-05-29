from zeus.device.cpu.common import CPUs, ZeusCPUInitError
from zeus.device.cpu.intel import rapl_is_available, INTELCPUs

_cpus: CPUs | None = None

def get_cpus() -> CPUs:
    """Initialize and return a singleton CPU monitoring object for INTEL CPUs.

    The function returns a CPU management object that aims to abstract the underlying CPU monitoring libraries
    (RAPL for Intel CPUs).

    This function attempts to initialize CPU mointoring using RAPL. If this attempt fails, it raises 
    a ZeusErrorInit exception.
    """
    global _cpus
    if _cpus is not None:
        return _cpus
    if rapl_is_available():
        _cpus = INTELCPUs()
        return _cpus
    else:
        raise ZeusCPUInitError(
            "RAPL unvailable Failed to initialize CPU management library."
        )
