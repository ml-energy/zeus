"""Monitor the wrapping around of RAPL counters."""

from __future__ import annotations

import atexit
import typing
import tempfile
import os
from time import time, sleep
import multiprocessing as mp

import pandas as pd
from sklearn.metrics import auc

from zeus.utils.logging import get_logger
from zeus.device import get_gpus

def infer_counter_update_period(rapl_file_path: str) -> float: 
    """Determine how long to sleep until next the next poll

    """
    return 1.0

class RaplMonitor:
    """Monitor the wrapping around of RAPL counters.

    This class acts as a lower level wrapper around a Python process that polls
    the wrapping of RAPL counters. This is primarily used by
    [`RAPLCPUs`][zeus.device.cpu.rapl.RAPLCPUs].

    !!! Warning
        Since the monitor spawns a child process, **it should not be instantiated as a global variable**.
        Python puts a protection to prevent creating a process in global scope.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    Attributes:
    """

    def __init__(
        self,
        rapl_file_path: str,
        max_energy_uj: float,
        rapl_csv_path: str | None = None,
    ) -> None:
        """Initialize the rapl monitor.

        Args:
            rapl_file_path: File path where the RAPL file is located
            rapl_csv_path: If given, the wrap around polling will write measurements
                to this path. Otherwise, a temporary file will be used.
        """
        if not os.path.exists(rapl_file_path):
            raise ValueError(f"{rapl_file_path} is not a valid file path")
        self.rapl_file_path = rapl_file_path

        # Set up logging.
        self.logger = get_logger(type(self).__name__)

        self.logger.info(f"Monitoring wrap around of {rapl_file_path}")

        # Create and open the CSV to record power measurements.
        if rapl_csv_path is None:
            rapl_csv_path = tempfile.mkstemp(suffix=".csv", text=True)[1]
        open(rapl_csv_path, "w").close()
        self.rapl_f = open(rapl_csv_path)
        self.rapl_df_columns = ["time", "energy"]
        self.rapl_df = pd.DataFrame(columns=self.rapl_df_columns)

        self.num_wraparounds = 0
        self.last_time = time()

        # Spawn the power polling process.
        atexit.register(self._stop)
        self.process = mp.get_context("spawn").Process(
            target=_polling_process,
            args=(rapl_file_path, max_energy_uj, rapl_csv_path),
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
                pd.read_csv(self.rapl_f, header=None, names=self.rapl_df_columns),
            )
        except pd.errors.EmptyDataError:
            return

        if additional_df.empty:
            return

        if self.rapl_df.empty:
            self.rapl_df = additional_df
        else:
            self.rapl_df = pd.concat(
                [self.rapl_df, additional_df],
                axis=0,
                ignore_index=True,
                copy=False,
            )

    def get_num_wraparounds(self) -> int:
        self._update_df()
        curr_time = time()

        # Get entries between last measured time and current time
        filtered = self.rapl_df[(self.rapl_df['time'] >= self.last_time) & (self.rapl_df['time'] <= curr_time)]
        for i in range(len(filtered)-1):
            if filtered.iloc[i+1]['energy'] < filtered.iloc[i]['energy']:
                self.num_wraparounds+=1
        print(filtered)

        self.last_time = curr_time
        return self.num_wraparounds

def _polling_process(
    rapl_file_path: str,
    max_energy_uj: float,
    rapl_csv_path: str,
) -> None:
    """Run the rapl monitor."""
    try:
        # Use line buffering.
        with open(rapl_csv_path, "w", buffering=1) as rapl_f:
            while True:
                now = time()
                sleep_time = 1.0
                with open(rapl_file_path, "r") as rapl_file:
                    energy_uj = float(rapl_file.read().strip())
                    if max_energy_uj - energy_uj < 1000:
                        sleep_time = 0.1
                rapl_f.write(f"{now},{energy_uj}\n")
                sleep(sleep_time)
    except KeyboardInterrupt:
        return
