"""Time, energy, and power monitors for Zeus.

The main class of this module is [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor].

If users wish to monitor power consumption over time, the [`power`][zeus.monitor.power]
module can come in handy.

If users wish to monitor GPU temperature over time, the [`temperature`][zeus.monitor.temperature]
module can come in handy.

If users wish to profile GPU kernels with thermally stable energy measurements,
the [`kernel_profiler`][zeus.monitor.kernel_profiler] module can come in handy.
"""

from zeus.monitor.energy import ZeusMonitor, Measurement
from zeus.monitor.kernel_profiler import (
    measure,
    profile_parameters,
    profile_measurement_duration,
    profile_cooldown_duration,
)
from zeus.monitor.power import PowerMonitor
from zeus.monitor.temperature import TemperatureMonitor
