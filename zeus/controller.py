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

"""Controllers influence the flow or progress of training."""

from __future__ import annotations

import pynvml

from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.util.metric import zeus_cost
from zeus.util.logging import get_logger


class EarlyStopController(Callback):
    """Controller for early stopping."""

    def __init__(
        self,
        monitor: ZeusMonitor | None = None,
        eta_knob: float = 0.5,
        cost_threshold: float | None = None,
        max_epochs: int | None = None,
        target_metric: float | None = None,
        higher_is_better: bool | None = None,
    ) -> None:
        r"""Initialize the controller.

        Check whether training should terminate through the `should_training_stop` attribute.
        - If you gave `max_epochs`, check after `on_epoch_end()`.
        - If you gave `cost_threshold`, check after `on_epoch_end()`.
        - If you gave `target_metric`, check after `on_evaluate()`.

        Args:
            monitor: The monitor instance to use for measuring time and energy.
                Required if `cost_threshold` is given.
            eta_knob: The $0 \le \eta \le 1$ knob for the Zeus time-energy cost.
                (Default: 0.5)
            cost_threshold: When running the next epoch will exceed this cost.
                Only training cost is considered, not validation or testing cost.
            max_epochs: Maximum number of epochs to run.
            target_metric: Stop training when the metric reaches this value.
            higher_is_better: If `True`, `target_metric` is assumed reached when the
                reported metric is larger than or equal to the `target_metric`.
                Required if `target_metric` is given.
        """
        # Sanity check the arguments.
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if cost_threshold is not None and cost_threshold <= 0:
            raise ValueError("cost_threshold must be positive")
        if (cost_threshold is None) ^ (monitor is None):
            raise ValueError("cost_threshold and monitor must be given together")
        if (target_metric is None) ^ (higher_is_better is None):
            raise ValueError(
                "target_metric and higher_is_better must be given together"
            )

        # Save arguments.
        self.monitor = monitor
        self.eta_knob = eta_knob
        self.cost_threshold = cost_threshold
        self.max_epochs = max_epochs
        self.target_metric = target_metric
        self.higher_is_better = higher_is_better

        # Setup logging.
        self.logger = get_logger(type(self).__name__)

        # Cache NVML device handles if they're needed.
        self.gpu_handles = {}
        self.max_power = {}
        if self.cost_threshold is not None:
            assert self.monitor is not None
            pynvml.nvmlInit()
            for gpu_index in self.monitor.gpu_indices:
                device = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_handles[gpu_index] = device
                self.max_power[gpu_index] = (
                    pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device)[1]
                    // 1000
                )

        # States.
        self.epochs_trained = 0
        self.epoch_costs = []

        # Once switched to `True`, there is no switching back to `False`.
        self.should_training_stop = False

    def on_epoch_begin(self) -> None:
        """Start measuring the cost of the next epoch."""
        if self.cost_threshold is not None:
            assert self.monitor is not None
            self.monitor.begin_window("__EarlyStopController_epoch")

    def on_epoch_end(self) -> None:
        """Check if the training cost of the next epoch will exceed the threshold."""
        if self.max_epochs is not None:
            self.epochs_trained += 1
            if self.epochs_trained >= self.max_epochs:
                self.logger.info(
                    "[Stop training!] Epochs trained %d >= Max epochs %d",
                    self.epochs_trained,
                    self.max_epochs,
                )
                self.should_training_stop = True
                return

        if self.cost_threshold is not None:
            assert self.monitor is not None
            measurement = self.monitor.end_window("__EarlyStopController_epoch")
            cost = sum(
                zeus_cost(
                    energy=measurement.energy[gpu_index],
                    time=measurement.time,
                    eta_knob=self.eta_knob,
                    max_power=self.max_power[gpu_index],
                )
                for gpu_index in self.monitor.gpu_indices
            )
            self.epoch_costs.append(cost)
            if (nec := self._expected_next_epoch_cost()) >= self.cost_threshold:
                self.logger.info(
                    "[Stop training!] Expected next epoch cost %f >= Cost threshold %f",
                    nec,
                    self.cost_threshold,
                )
                self.should_training_stop = True
                return

    def on_evaluate(self, metric: float) -> None:
        """Check if the target metric was reached."""
        if self.target_metric is not None:
            assert self.higher_is_better is not None
            # ruff: noqa: SIM108
            if self.higher_is_better:
                reached = metric >= self.target_metric
            else:
                reached = metric <= self.target_metric
            if reached:
                self.logger.info(
                    "[Stop training!] Evaluation metric %f reached target metric %f",
                    metric,
                    self.target_metric,
                )
                self.should_training_stop = True

    def _expected_next_epoch_cost(self) -> float:
        """Predict the total cost if the next training epoch is to be run."""
        cost_until_now = sum(self.epoch_costs)
        average_epoch_cost = cost_until_now / len(self.epoch_costs)
        return cost_until_now + average_epoch_cost
