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

"""Monitor the power usage of GPUs."""

from __future__ import annotations

import atexit
import typing
import tempfile
from time import time, sleep
import multiprocessing as mp

import pynvml
import pandas as pd
from sklearn.metrics import auc

from zeus.util.logging import get_logger
from zeus.util.env import resolve_gpu_indices


def infer_counter_update_period(nvml_handles: list[pynvml.c_nvmlDevice_t]) -> float:
    """Infer the update period of the NVML power counter.

    NVML counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    pynvml.nvmlInit()

    logger = get_logger(__name__)

    # For each unique GPU model, infer the update period.
    update_period = 0.0
    gpu_models_covered = set()
    for handle in nvml_handles:
        if (model := pynvml.nvmlDeviceGetName(handle)) not in gpu_models_covered:
            logger.info(
                "Detected %s, inferring NVML power counter update period.", model
            )
            gpu_models_covered.add(model)
            detected_period = _infer_counter_update_period_single(handle)
            logger.info(
                "Counter update period for %s is %.2f s",
                model,
                detected_period,
            )
            if update_period > detected_period:
                update_period = detected_period

    pynvml.nvmlShutdown()

    # Target half the update period to ensure that the counter is enough.
    update_period /= 2.0

    # Anything less than ten times a second is probably too slow.
    if update_period > 0.1:
        logger.warning(
            "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
            update_period,
        )
        update_period = 0.1
    return update_period


def _infer_counter_update_period_single(nvml_handle: pynvml.c_nvmlDevice_t) -> float:
    """Infer the update period of the NVML power counter for a single GPU."""
    # Collect 1000 samples of the power counter with timestamps.
    time_power_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(1000)]
    for i in range(len(time_power_samples)):
        time_power_samples[i] = (
            time(),
            pynvml.nvmlDeviceGetPowerUsage(nvml_handle),
        )

    # Find the timestamps when the power readings changed.
    changed_times = []
    prev_power = time_power_samples[0][1]
    for t, p in time_power_samples:
        if p != prev_power:
            changed_times.append(t)
            prev_power = p

    # Compute the minimum time difference between power change timestamps.
    return min(time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:]))


class PowerMonitor:
    """Monitor power usage from GPUs.

    This class acts as a lower level wrapper around a Python process that polls
    the power consumption of GPUs. This is primarily used by
    [`ZeusMonitor`][zeus.monitor.ZeusMonitor] for older architecture GPUs that
    do not support the nvmlDeviceGetTotalEnergyConsumption API.

    Attributes:
        gpu_indices (list[int]): Indices of the GPUs to monitor.
        update_period (int): Update period of the power monitor in seconds.
            Holds inferred update period if `update_period` was given as `None`.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float | None = None,
    ) -> None:
        """Initialize the power monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the power monitor in seconds. If None,
                infer the update period by max speed polling the power counter for
                each GPU model.
        """
        if gpu_indices is not None and not gpu_indices:
            raise ValueError("`gpu_indices` must be either `None` or non-empty")

        # Initialize NVML.
        pynvml.nvmlInit()

        # Set up logging.
        self.logger = get_logger(type(self).__name__)

        # Get GPU indices and NVML handles.
        self.gpu_indices, nvml_gpu_indices = resolve_gpu_indices(gpu_indices)
        nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in nvml_gpu_indices]
        self.logger.info("Monitoring power usage of GPUs %s", self.gpu_indices)

        # Infer the update period if necessary.
        if update_period is None:
            update_period = infer_counter_update_period(nvml_handles)
        self.update_period = update_period

        # Create the CSV file for power measurements.
        power_csv = tempfile.mkstemp(suffix=".csv", text=True)[1]
        open(power_csv, "w").close()
        self.power_f = open(power_csv)
        self.power_df_columns = ["time"] + [f"power{i}" for i in self.gpu_indices]
        self.power_df = pd.DataFrame(columns=self.power_df_columns)

        # Spawn the power polling process.
        atexit.register(self._stop)
        self.process = mp.get_context("spawn").Process(
            target=_polling_process, args=(nvml_gpu_indices, power_csv, update_period)
        )
        self.process.start()

    def _stop(self) -> None:
        """Stop monitoring power usage."""
        pynvml.nvmlShutdown()
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process.kill()
            self.process = None

    def _update_df(self) -> None:
        """Add rows to the power dataframe from the CSV file."""
        try:
            additional_df = typing.cast(
                pd.DataFrame,
                pd.read_csv(self.power_f, header=None, names=self.power_df_columns),  # type: ignore
            )
        except pd.errors.EmptyDataError:
            return
        self.power_df = pd.concat([self.power_df, additional_df], axis=0)

    def get_energy(self, start_time: float, end_time: float) -> dict[int, float] | None:
        """Get the energy used by the GPUs between two times.

        Args:
            start_time: Start time of the interval, from time.time().
            end_time: End time of the interval, from time.time().

        Returns:
            A dictionary mapping GPU indices to the energy used by the GPU between the
            two times. GPU indices are from the DL framework's perspective after
            applying `CUDA_VISIBLE_DEVICES`.
            If there are no power readings, return None.
        """
        self._update_df()

        if self.power_df.empty:
            return None

        df = typing.cast(
            pd.DataFrame, self.power_df.query(f"{start_time} <= time <= {end_time}")
        )

        try:
            return {
                i: float(auc(df["time"], df[f"power{i}"])) for i in self.gpu_indices
            }
        except ValueError:
            return None

    def get_power(self, time: float | None = None) -> dict[int, float] | None:
        """Get the power usage of the GPUs at a specific time point.

        Args:
            time: Time point to get the power usage at. If None, get the power usage
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the power usage of the GPU at the
            specified time point. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
            If there are no power readings, return None.
        """
        self._update_df()

        if self.power_df.empty:
            return None

        if time is None:
            row = self.power_df.iloc[-1]
        else:
            ind = self.power_df.time.searchsorted(time)
            try:
                row = self.power_df.iloc[ind]
            except IndexError:
                # This means that the time is after the last recorded power reading.
                row = self.power_df.iloc[-1]

        return {i: float(row[f"power{i}"]) for i in self.gpu_indices}


def _polling_process(
    nvml_gpu_indices: list[int],
    power_csv: str,
    update_period: float,
) -> None:
    """Run the power monitor."""
    try:
        pynvml.nvmlInit()
        nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in nvml_gpu_indices]

        # Use line buffering.
        with open(power_csv, "w", buffering=1) as power_f:
            while True:
                power: list[float] = []
                now = time()
                for nvml_handle in nvml_handles:
                    power.append(pynvml.nvmlDeviceGetPowerUsage(nvml_handle))
                power_str = ",".join(map(lambda p: str(p / 1000), power))
                power_f.write(f"{now},{power_str}\n")
                if (sleep_time := update_period - (time() - now)) > 0:
                    sleep(sleep_time)
    except KeyboardInterrupt:
        return
    finally:
        pynvml.nvmlShutdown()
