# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
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

"""Defines the energy-time cost metric function."""

from __future__ import annotations


def zeus_cost(energy: float, time: float, eta_knob: float, max_power: int) -> float:
    """Compute Zeus's energy-time cost metric.

    Trades off ETA and TTA based on the value of `eta_knob`.
    The caller is expected to do bound checking for `eta_knob`,
    because `eta_knob` does not change frequently.

    Args:
        energy: Joules
        time: seconds
        eta_knob: Real number in [0, 1].
        max_power: The maximum power limit of the GPU.

    Returns:
        The cost of the DL training job.
    """
    return eta_knob * energy + (1 - eta_knob) * max_power * time
