"""Monitor the power usage of GPUs."""

from __future__ import annotations

import atexit
import typing
import tempfile
from time import time, sleep
import multiprocessing as mp

import pandas as pd
from sklearn.metrics import auc

from zeus.utils.logging import get_logger
from zeus.device import get_gpus


def infer_counter_update_period(gpu_indicies: list[int]) -> float:
    """Infer the update period of the NVML power counter.

    NVML counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    logger = get_logger(__name__)

    # get gpus
    gpus = get_gpus()

    # For each unique GPU model, infer the update period.
    update_period = 0.0
    gpu_models_covered = set()
    for index in gpu_indicies:
        if (model := gpus.getName(index)) not in gpu_models_covered:
            logger.info(
                "Detected %s, inferring NVML power counter update period.", model
            )
            gpu_models_covered.add(model)
            detected_period = _infer_counter_update_period_single(index)
            logger.info(
                "Counter update period for %s is %.2f s",
                model,
                detected_period,
            )
            update_period = min(update_period, detected_period)

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


def _infer_counter_update_period_single(gpu_index: int) -> float:
    """Infer the update period of the NVML power counter for a single GPU."""
    # get gpus
    gpus = get_gpus()
    # Collect 1000 samples of the power counter with timestamps.
    time_power_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(1000)]
    for i in range(len(time_power_samples)):
        time_power_samples[i] = (
            time(),
            gpus.getInstantPowerUsage(gpu_index),
        )

    # Find the timestamps when the power readings changed.
    time_power_samples = time_power_samples[10:]
    changed_times = []
    prev_power = time_power_samples[0][1]
    for t, p in time_power_samples:
        if p != prev_power:
            changed_times.append(t)
            prev_power = p

    # Compute the minimum time difference between power change timestamps.
    intervals = [
        time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:])
    ]
    if len(intervals) == 0:
        return 0.1
    return min(intervals)


class PowerMonitor:
    """Monitor power usage from GPUs.

    This class acts as a lower level wrapper around a Python process that polls
    the power consumption of GPUs. This is primarily used by
    [`ZeusMonitor`][zeus.monitor.ZeusMonitor] for older architecture GPUs that
    do not support the nvmlDeviceGetTotalEnergyConsumption API.

    !!! Warning
        Since the monitor spawns a child process, **it should not be instantiated as a global variable**.
        Python puts a protection to prevent creating a process in global scope.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    Attributes:
        gpu_indices (list[int]): Indices of the GPUs to monitor.
        update_period (int): Update period of the power monitor in seconds.
            Holds inferred update period if `update_period` was given as `None`.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float | None = None,
        power_csv_path: str | None = None,
    ) -> None:
        """Initialize the power monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the power monitor in seconds. If None,
                infer the update period by max speed polling the power counter for
                each GPU model.
            power_csv_path: If given, the power polling process will write measurements
                to this path. Otherwise, a temporary file will be used.
        """
        if gpu_indices is not None and not gpu_indices:
            raise ValueError("`gpu_indices` must be either `None` or non-empty")

        # Get GPUs
        gpus = get_gpus()

        # Set up logging.
        self.logger = get_logger(type(self).__name__)

        # Get GPUs
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(len(gpus)))
        )
        self.logger.info("Monitoring power usage of GPUs %s", self.gpu_indices)

        # Infer the update period if necessary.
        if update_period is None:
            update_period = infer_counter_update_period(self.gpu_indices)
        self.update_period = update_period

        # Create and open the CSV to record power measurements.
        if power_csv_path is None:
            power_csv_path = tempfile.mkstemp(suffix=".csv", text=True)[1]
        open(power_csv_path, "w").close()
        self.power_f = open(power_csv_path)
        self.power_df_columns = ["time"] + [f"power{i}" for i in self.gpu_indices]
        self.power_df = pd.DataFrame(columns=self.power_df_columns)

        # Spawn the power polling process.
        atexit.register(self._stop)
        self.process = mp.get_context("spawn").Process(
            target=_polling_process,
            args=(self.gpu_indices, power_csv_path, update_period),
        )
        self.process.start()

    def _stop(self) -> None:
        """Stop monitoring power usage."""
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
                pd.read_csv(self.power_f, header=None, names=self.power_df_columns),
            )
        except pd.errors.EmptyDataError:
            return

        if additional_df.empty:
            return

        if self.power_df.empty:
            self.power_df = additional_df
        else:
            self.power_df = pd.concat(
                [self.power_df, additional_df],
                axis=0,
                ignore_index=True,
                copy=False,
            )

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
            If there are no power readings (e.g., future timestamps), return None.
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
    gpu_indices: list[int],
    power_csv_path: str,
    update_period: float,
) -> None:
    """Run the power monitor."""
    try:
        # Get GPUs
        gpus = get_gpus()

        # Use line buffering.
        with open(power_csv_path, "w", buffering=1) as power_f:
            while True:
                power: list[float] = []
                now = time()
                for index in gpu_indices:
                    power.append(gpus.getInstantPowerUsage(index))
                power_str = ",".join(map(lambda p: str(p / 1000), power))
                power_f.write(f"{now},{power_str}\n")
                if (sleep_time := update_period - (time() - now)) > 0:
                    sleep(sleep_time)
    except KeyboardInterrupt:
        return
