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

"""Time, energy, and power monitors for Zeus.

The main class of this module is [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor].

If users wish to monitor power consumption over time, the [`power`][zeus.monitor.power]
module can come in handy.
"""

from zeus.monitor.energy import ZeusMonitor, Measurement
from zeus.monitor.power import PowerMonitor
