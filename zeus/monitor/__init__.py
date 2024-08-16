"""Time, energy, and power monitors for Zeus.

The main class of this module is [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor].

If users wish to monitor power consumption over time, the [`power`][zeus.monitor.power]
module can come in handy.
"""

from zeus.monitor.energy import ZeusMonitor, Measurement
from zeus.monitor.power import PowerMonitor
