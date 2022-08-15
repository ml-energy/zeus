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

"""Defines the ZeusDataLoader class."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
from functools import cached_property
from pathlib import Path
from typing import Generator

import pynvml
import torch
from torch.utils.data import DataLoader

from zeus import analyze
from zeus.util.check import get_env
from zeus.util.metric import zeus_cost

# JIT profiling states
NOT_PROFILING = "NOT_PROFILING"
WARMING_UP = "WARMING_UP"
PROFILING = "PROFILING"


class ZeusDataLoader(DataLoader):
    r"""Profiles and optimizes GPU power limit.

    `ZeusDataLoader` is integrated into the DNN training script, and transparently
    profiles power and time consumption to determine the optimal GPU power limit.

    # Integration example

    ```python
    from zeus.run import ZeusDataLoader

    # The one instantiated with max_epochs becomes the train dataloader
    train_loader = ZeusDataLoader(train_set, batch_size=256, max_epochs=100)
    eval_loader = ZeusDataLoader(eval_set, batch_size=256)

    for epoch_number in train_loader.epochs():
        for batch in train_loader:
            # Learn from batch
        for batch in eval_loader:
            # Evaluate on batch

        train_loader.report_metric(validation_metric)
    ```

    # Environment variables

    `ZeusDataLoader` interfaces with the outside world via environment variables.
    Thus, while `ZeusDataLoader` is paired together with
    [`ZeusMaster`][zeus.run.ZeusMaster] in example scripts, any other "driver" can
    use `ZeusDataLoader` as long as it sets appropriate environment variables.

      - `ZEUS_TARGET_METRIC` : Required. Zeus will stop training when this target
                               validation metric is reached. Will be cast to float.
      - `ZEUS_LOG_DIR`       : Directory to store profiling logs. (Default:` "zeus_log"`)
      - `ZEUS_JOB_ID`        : String to prefix in logs. (Default:` "zeus"`)
      - `ZEUS_COST_THRESH`   : Stop training when the energy-time cost will exceed
                               this threshold.  (Default:` "inf"`)
      - `ZEUS_ETA_KNOB`      : $\eta$ knob to tradeoff between energy and time.
                               Larger values reduce more energy and sacrifice time.
                               (Default:` "0.5"`)
      - `ZEUS_MONITOR_PATH`  : Path to the Zeus power monitor binary.
                               (Default:` "/workspace/zeus/zeus_monitor/zeus_monitor"`)
      - `ZEUS_PROFILE_PARAMS`: Warmup and measure seconds for each power limit,
                               separated by a comma. (Default:` "1.0,4.0"`)
      - `ZEUS_USE_OPTIMAL_PL`: Whether to actually use the optimal power limit found.
                               Setting this to false is the Observer Mode described
                               in Section 5. (Default:` "True"`)
    """

    # A global power monitor instance.
    monitor: subprocess.Popen | None = None

    # The power limit currently set for the GPU.
    current_gpu_pl: int = 0

    # Train batch size to be accessed by the eval dataloader.
    train_batch_size: int = 0

    # Length of the eval dataloader. `epochs` in the train dataloader needs this.
    eval_num_samples: int = 0

    # Train-time power profiling result. Maps power limit to avg_power & throughput.
    train_power_result: dict[int, float] = {}
    train_tput_result: dict[int, float] = {}

    # Eval-time power profiling result. Maps power limit to avg_power & throughput.
    eval_power_result: dict[int, float] = {}
    eval_tput_result: dict[int, float] = {}

    # Cost-optimal power limit. Set by the train dataloader after the last power limit
    # was explored.
    optimal_pl: int = 0

    # Train epoch measurements for time/energy accounting.
    train_epoch_time: list[float] = []
    train_epoch_energy: list[float] = []

    # Eval-time latency profiling result. Maps power limit to epoch latency.
    eval_epoch_time: list[float] = []
    eval_epoch_energy: list[float] = []

    def __init__(
        self,
        *args,
        batch_size: int,
        max_epochs: int = -1,
        **kwargs,
    ) -> None:
        """Initialize the dataloader.

        Args:
            batch_size: Batch size to use for training.
            max_epochs: Maximum number of epochs to train. **Specify this parameter only
                to the train data loader.**
        """
        # Sanity checks.
        self.batch_size = batch_size
        self.split = "train" if max_epochs != -1 else "eval"
        self.max_epochs = max_epochs
        self.log_prefix = f"\n[ZeusDataLoader({self.split})]"
        if self._is_train:
            if ZeusDataLoader.train_batch_size != 0:
                raise ValueError("Specify max_epochs only to the train dataloader.")
            ZeusDataLoader.train_batch_size = batch_size

        # Initialize the DataLoader.
        super().__init__(*args, batch_size=batch_size, **kwargs)

        # Retrieve environment variables from ZeusMaster.
        self.target_metric = get_env("ZEUS_TARGET_METRIC", float)
        self.logdir = get_env("ZEUS_LOG_DIR", str, default="zeus_log")
        self.job_id = get_env("ZEUS_JOB_ID", str, default="zeus")
        self.cost_thresh = get_env("ZEUS_COST_THRESH", float, default=float("inf"))
        self.eta_knob = get_env("ZEUS_ETA_KNOB", float, default=0.5)
        self.monitor_path = get_env(
            "ZEUS_MONITOR_PATH",
            str,
            default="/workspace/zeus/zeus_monitor/zeus_monitor",
        )
        self.warmup_sec, self.profile_sec = map(
            float, get_env("ZEUS_PROFILE_PARAMS", str, default="1.0,4.0").split(",")
        )
        self.use_optimal_pl = get_env("ZEUS_USE_OPTIMAL_PL", bool, default=True)

        # Create ZEUS_LOG_DIR if it does not exist.
        os.makedirs(self.logdir, exist_ok=True)

        # Check whether the monitor path is good.
        if not os.access(self.monitor_path, os.X_OK):
            raise RuntimeError(f"'{self.monitor_path}' is not executable")

        # Whether the target metric was reached.
        self.target_metric_reached = False

        # Construct relevant paths.
        self.train_json = (
            f"{self.logdir}/{self.job_id}+bs{self.train_batch_size}.train.json"
        )
        self.power_json = f"{self.logdir}/bs{self.train_batch_size}.power.json"

        # Numbers related to the dataloader.
        self.epoch_num = 0
        self.sample_num = 0
        self.num_samples = len(self)

        # Pass the length of the eval dataloader for `epochs`.
        if not self._is_train:
            ZeusDataLoader.eval_num_samples = self.num_samples

        # Power profiling windows
        #
        # +----- warmup_start (change power limit)
        # |        +----- prof_start (record timestamp)
        # |        |                      +----- prof_end (compute and save results)
        # | warmup |        profile       |
        # v   sec  v           sec        v
        # ================================= =====================
        # |      power limit = 250W       | | power limit = 225W  ...
        # ================================= =====================
        #
        # =======================================================
        # |                         Epoch 1                       ...
        # =======================================================
        #
        self.warmup_start_time = 0.0
        self.prof_start_time = 0.0
        self.prof_start_sample = 0
        self.prof_state = NOT_PROFILING
        self.prof_pl_index = 0
        self.prev_cutoff_pl = -1

        # Initialize NVML and get GPU handle. Currently only supports single-GPU.
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU index 0.
        # Set persistent mode.
        pynvml.nvmlDeviceSetPersistenceMode(
            self.gpu_handle, pynvml.NVML_FEATURE_ENABLED
        )
        # Query NVML for the possible power limit range. Unit is mW.
        min_pl, self.max_pl = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
            self.gpu_handle
        )
        self.power_limits = list(range(self.max_pl, min_pl - 25_000, -25_000))
        if self._is_train:
            self._log(f"Power limit range: {self.power_limits}")

        if self._is_train:
            should_profile = self._should_profile
            self._log(f"Power profiling: {'ON' if should_profile else 'OFF'}")
            # If we need to do profiling, no need to touch the power limit.
            # If profiling is already done, load profile information from power_json.
            # Only then do we have the optimal PL available.
            # We only load in the train dataloader since it populates classvars.
            if not should_profile:
                self._load_power_results()
                self._set_gpu_steady_power_limit()

        # Make sure NVML is shutdown and the monitor is killed when the training script exits.
        if self._is_train:

            def exit_hook():
                pynvml.nvmlShutdown()
                if self.monitor is not None:
                    self.monitor.kill()
                    print("[ZeusDataLoader] Stopped Zeus monitor.")

            atexit.register(exit_hook)

    def epochs(self) -> Generator[int, None, None]:
        """Yield the current epoch number from 0 until when training should stop.

        Training should stop when

        - the cost reached the cost threshold, or
        - the maximum number of epochs was eached, or
        - the target metric was reached.

        When done, stores the job results in `train_json`.

        Yields:
            Epoch indices starting from zero.
        """
        # Sanity check.
        assert self._is_train, "Use epochs() on the train dataloader."

        while True:
            # Sanity checks.
            enum = self.epoch_num
            assert len(self.train_epoch_time) == enum, f"{len(self.train_epoch_time)=}"
            assert (
                len(self.train_epoch_energy) == enum
            ), f"{len(self.train_epoch_energy)=}"
            assert len(self.eval_epoch_time) == enum, f"{len(self.eval_epoch_time)=}"
            assert (
                len(self.eval_epoch_energy) == enum
            ), f"{len(self.eval_epoch_energy)=}"

            # Compute time and energy consumption up to now.
            time_consumed = sum(self.train_epoch_time + self.eval_epoch_time)
            energy_consumed = sum(self.train_epoch_energy + self.eval_epoch_energy)
            cost = zeus_cost(
                energy_consumed, time_consumed, self.eta_knob, self.max_pl // 1000
            )
            self._log(
                f"Up to epoch {self.epoch_num}: "
                f"time={time_consumed:.2f}, energy={energy_consumed:.2f}, cost={cost:.2f}"
            )

            # Stop if the target metric was reached.
            if self.target_metric_reached:
                self._log(f"Target metric {self.target_metric} was reached! Stopping.")
                self._save_train_results(energy_consumed, time_consumed, cost, True)
                return

            # Max epoch is a hard stop.
            if self.epoch_num >= self.max_epochs:
                self._log(
                    f"Maximum number of epochs {self.max_epochs} reached. Stopping."
                )
                self._save_train_results(energy_consumed, time_consumed, cost, False)
                return

            # No need to do anything in the first epoch.
            if self.epoch_num == 0:
                yield 0
                continue

            # Just continue if we're profiling.
            # This will ignore and continue training even if the cost threshold was exceeded.
            # However, the profiling cost actually exceeding the cost threshold would not
            # happen frequently. It's more like a wrong cost threshold.
            if self._should_profile:
                if cost >= self.cost_thresh:
                    self._log(
                        f"{cost=:.2f} exceeded threshold {self.cost_thresh:.2f}, "
                        "but just continue since we're profiling."
                    )
                yield self.epoch_num
                continue

            # We want to predict whether running the next epoch will exceed the cost threshold.
            next_train_time = self.num_samples / self.train_tput_result[self.optimal_pl]
            next_eval_time = (
                self.eval_num_samples / self.eval_tput_result[self.optimal_pl]
            )
            next_time = next_train_time + next_eval_time
            next_train_energy = (
                next_train_time * self.train_power_result[self.optimal_pl]
            )
            next_eval_energy = next_eval_time * self.eval_power_result[self.optimal_pl]
            next_energy = next_train_energy + next_eval_energy
            self._log(
                f"Optimal PL train & eval expected time={next_time:.2f} energy={next_energy:.2f}"
            )
            next_time_consumed = time_consumed + next_time
            next_energy_consumed = energy_consumed + next_energy
            next_cost = zeus_cost(
                next_energy_consumed,
                next_time_consumed,
                self.eta_knob,
                self.max_pl // 1000,
            )
            self._log(
                f"Expected next epoch: time={next_time_consumed:.2f}, "
                f"energy={next_energy_consumed:.2f}, "
                f"cost={next_cost:.2f}"
            )

            # Stop if the predicted cost of the next epoch exceeds the cost threshold.
            if next_cost >= self.cost_thresh:
                self._log(
                    f"Next expected cost exceeds cost threshold {self.cost_thresh:.2f}! Stopping."
                )
                self._save_train_results(energy_consumed, time_consumed, cost, False)
                return

            yield self.epoch_num

    def report_metric(self, metric: float, higher_is_better: bool) -> None:
        """Report the validation metric to the train dataloader.

        Args:
            metric: The validation metric of the current epoch.
            higher_is_better: For example, this should be `True` for accuracy
                and `False` for error.
        """
        assert self._is_train, "Use report_metric on the train dataloader."
        if higher_is_better:
            if metric >= self.target_metric:
                self.target_metric_reached = True
        else:
            if metric <= self.target_metric:
                self.target_metric_reached = True

    @property
    def _should_profile(self) -> bool:
        """Whether profiling is not done."""
        return not Path(self.power_json).exists()

    @property
    def _power_limits_left(self) -> bool:
        """Whether there are power limits left to profile."""
        return self.prof_pl_index < len(self.power_limits)

    def _compute_optimal_pl(self) -> int:
        """Return the cost-optimal power limit."""
        # Sanity checks.
        assert ZeusDataLoader.train_tput_result
        assert ZeusDataLoader.train_power_result

        # Compute power cost
        tput = ZeusDataLoader.train_tput_result
        power = ZeusDataLoader.train_power_result
        cost_map = {
            pl: (self.eta_knob * power[pl] + (1 - self.eta_knob) * self.max_pl)
            / tput[pl]
            for pl in self.power_limits
        }
        optimal_pl = min(cost_map.keys(), key=cost_map.get)  # type: ignore
        self._log(f"Cost-optimal power limit is {optimal_pl//1000}W")
        return optimal_pl

    def _set_gpu_power_limit(self, power_limit: int) -> None:
        """Set the GPU's power limit using NVML.

        This method only invokes NVML when `power_limit` is not the same as
        the current GPU power limit.

        Args:
            power_limit: Power limit to set.
        """
        if self.current_gpu_pl != power_limit:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.gpu_handle, power_limit)
            ZeusDataLoader.current_gpu_pl = power_limit
        self._log(f"Set GPU power limit to {self.current_gpu_pl//1000}W.")

    def _set_gpu_steady_power_limit(self) -> None:
        """Set the steady power limit based on self.use_optimal_pl."""
        power_limit = ZeusDataLoader.optimal_pl if self.use_optimal_pl else self.max_pl
        self._log(
            "Steady state power limit: "
            f"{'OPT' if self.use_optimal_pl else 'MAX'} {power_limit//1000}W"
        )
        self._set_gpu_power_limit(power_limit)

    def _log(self, message: str) -> None:
        """Print out message with prefix.

        Args:
            message: The message to log out.
        """
        print(self.log_prefix, message, flush=True)

    @cached_property
    def _is_train(self) -> bool:
        """Return whether this dataloader is for training."""
        return self.split == "train"

    @property
    def _power_log_path(self) -> str:
        """Build the path for the power monitor log file."""
        return f"{self.logdir}/bs{self.train_batch_size}+e{self.epoch_num}.power.log"

    def _start_monitor(self) -> None:
        """Start the power monitor subprocess."""
        # Sanity checks.
        assert ZeusDataLoader.monitor is None

        # Start monitor.
        ZeusDataLoader.monitor = subprocess.Popen(
            [self.monitor_path, self._power_log_path, "0", "100"],
        )

    def _kill_monitor(self) -> None:
        """Kill the power monitor subprocess."""
        # Sanity checks.
        assert ZeusDataLoader.monitor is not None

        # Kill monitor.
        ZeusDataLoader.monitor.send_signal(signal.SIGINT)
        ZeusDataLoader.monitor.wait(timeout=1.0)
        ZeusDataLoader.monitor = None

    def _start_warmup(self) -> None:
        """Let the GPU run for some time with the poewr limit to profile."""
        # Sanity checks.
        assert self._should_profile, "start_warmup: should_profile=False"
        assert self.prof_pl_index < len(
            self.power_limits
        ), f"start_warmup: {self.prof_pl_index=}"

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Change power limit.
        power_limit = self.power_limits[self.prof_pl_index]
        self._set_gpu_power_limit(power_limit)

        # Start warmup timer.
        self.warmup_start_time = time.monotonic()

        # Set profiling state.
        self.prof_state = WARMING_UP

        self._log(f"Warm-up started with power limit {self.current_gpu_pl//1000}W")

    def _start_prof(self) -> None:
        """Start profiling power consumption by spawning the power monitor."""
        # Sanity checks.
        assert self._should_profile, "start_prof: should_profile=False"

        # Start profile timer.
        self.prof_start_time = time.monotonic()

        # Set the sample number when we started profiling.
        self.prof_start_sample = self.sample_num

        # Set profiling state.
        self.prof_state = PROFILING

        self._log(f"Profile started with power limit {self.current_gpu_pl//1000}W")

    def _end_prof(self, cutoff: bool) -> None:
        """End profiling power consumption for this power limit.

        Args:
            cutoff: True if the window was terminated before `profile_sec`
                because the current epoch ended before that.
        """
        # Sanity checks.
        assert self._should_profile, f"end_prof: {self._should_profile=}"

        # Set profiling state.
        self.prof_state = NOT_PROFILING

        # Nothing left to do if the window was terminated prematurely.
        if cutoff:
            self._log(
                f"Profiling with power limit {self.current_gpu_pl//1000}W cut off!"
            )
            # The same power limit was cut off two times.
            # This means that the epoch ends too quick for the full profile window
            # to fit in. Currently this is treated as an error.
            if self.prev_cutoff_pl == self.current_gpu_pl:
                raise RuntimeError(
                    "Epoch ends too quickly, and even one power limit cannot be profiled"
                    " within it. Consider decreasing the warmup and/or measure time"
                    " configuration. If you're using ZeusMaster, you can configure them"
                    " in ZeusMaster's __init__ function."
                )
            self.prev_cutoff_pl = self.current_gpu_pl
            return

        # Call cudaSynchronize to make sure this is the iteration boundary.
        torch.cuda.synchronize()

        # Freeze time.
        now = time.monotonic()

        # Compute and save average power.
        # The monitor is still running, so we just integrate from the beginning
        # of this profiling window (of course exclude warmup) up to now.
        avg_power = analyze.avg_power(
            self._power_log_path, start=self.prof_start_time - self.epoch_start_time
        )
        self.train_power_result[self.current_gpu_pl] = avg_power

        # Compute and save throughput.
        time_consumed = now - self.prof_start_time
        samples_processed = self.sample_num - self.prof_start_sample
        throughput = samples_processed / time_consumed
        self.train_tput_result[self.current_gpu_pl] = throughput

        # Advance to the next power limit. Affects self.power_limits_left.
        self.prof_pl_index += 1

        self._log(f"Profile done with power limit {self.current_gpu_pl//1000}W")

        # If we're done with all power limits,, compute the optimal power limit
        # and change to that power limit for the rest of the epoch.
        # This will lead to the eval epoch being run with the optimal power limit,
        # and since self.should_profile is still True, tput/power will be profiled.
        if not self._power_limits_left:
            self._log("This was the last power limit to explore.")
            ZeusDataLoader.optimal_pl = self._compute_optimal_pl()
            self._set_gpu_power_limit(ZeusDataLoader.optimal_pl)

    def _save_power_results(self) -> None:
        """Write the power profiling results to `power_json`."""
        prof_result = dict(
            job_id=self.job_id,  # Not used. Just for the purpose of record.
            train_power=self.train_power_result,
            train_throughput=self.train_tput_result,
            eval_power=self.eval_power_result,
            eval_throughput=self.eval_tput_result,
            optimal_pl=self.optimal_pl,
        )
        # NOTE: Write-then-move needed if we're handling concurrent jobs.
        with open(self.power_json, "w") as f:
            json.dump(prof_result, f)
        with open(self.power_json, "r") as f:
            self._log(f"Power profiling done.\nSaved {self.power_json}: {f.read()}")

    def _load_power_results(self) -> None:
        """Load power profiling information into the class from `power_json`."""
        # Helper function that casts the keys of a dictionary to integer.
        def as_int_key(dictionary: dict[str, float]) -> dict[int, float]:
            result = {}
            for key, value in dictionary.items():
                result[int(key)] = value
            return result

        with open(self.power_json, "r") as f:
            power_results = json.load(f)

        ZeusDataLoader.train_power_result = as_int_key(power_results["train_power"])
        ZeusDataLoader.train_tput_result = as_int_key(power_results["train_throughput"])
        ZeusDataLoader.eval_power_result = as_int_key(power_results["eval_power"])
        ZeusDataLoader.eval_tput_result = as_int_key(power_results["eval_throughput"])
        ZeusDataLoader.optimal_pl = power_results["optimal_pl"]

        self._log(f"Loaded {self.power_json}: {power_results}")

    def _save_train_results(
        self, energy: float, time_: float, cost: float, reached: bool
    ) -> None:
        """Write the job training results to `train_json`."""
        train_result = dict(
            energy=energy,
            time=time_,
            cost=cost,  # Not used. Just for reference.
            num_epochs=self.epoch_num,  # Not used. Just for reference.
            reached=reached,
        )
        with open(self.train_json, "w") as f:
            json.dump(train_result, f)
        with open(self.train_json, "r") as f:
            self._log(f"Training done.\nSaved {self.train_json}: {f.read()}")

    # pylint: disable=attribute-defined-outside-init
    def __iter__(self):
        """Signal the beginning of an epoch."""
        # Update counters.
        self.epoch_num += 1
        self.sample_num = 0
        self._log(f"Epoch {self.epoch_num} begin.")

        # Start epoch timer.
        self.epoch_start_time = time.monotonic()

        # Cache the dataloader iterator.
        self.iter = super().__iter__()

        # The power limit of the GPU is only changed by the train dataloader.
        if self._is_train:
            # The train loader always starts the monitor, and the eval loader kills it.
            self._start_monitor()
            # If we're not profiling, use the steady state power limit.
            # If we are profiling, the power limit will be set in __next__ with warmup.
            if not self._should_profile:
                self._set_gpu_steady_power_limit()

        return self

    def __next__(self):
        """Signal the beginning of an iteration."""
        # Update counters.
        self.sample_num += 1

        # Try to fetch next batch.
        try:
            data = next(self.iter)
        except StopIteration:
            # End of this epoch. Make sure all GPU operations are done so that
            # now is the *actual* end of this epoch.
            torch.cuda.synchronize()

            # The eval dataloader kills the monitor.
            if not self._is_train:
                self._kill_monitor()

            # Compute epoch time and energy consumption.
            # We're interested in the actual time/energy consumption here.
            #
            #   <----------------- monitor lifetime ------------------>
            #
            #   =======================================================
            #   |                Train                    ||   Eval   |
            #   =======================================================
            #   ^                                         ^^          ^
            #   |                                        / |          |
            #   epoch_start_time          time.monotonic() |          |
            #   for train loader          for train loader |          |
            #                                              |          |
            #                               epoch_start_time   time.monotonic()
            #                                for eval loader   for eval loader
            #
            time_consumption = time.monotonic() - self.epoch_start_time
            if self._is_train:
                # The monitor is still running, and we integrate over the entire log.
                energy_consumption = analyze.energy(self._power_log_path)
                self.train_epoch_time.append(time_consumption)
                self.train_epoch_energy.append(energy_consumption)
            else:
                # We just killed the monitor. Integrate the last time_consumption seconds.
                energy_consumption = analyze.energy(
                    self._power_log_path, start=-time_consumption
                )
                self.eval_epoch_time.append(time_consumption)
                self.eval_epoch_energy.append(energy_consumption)
                # For the eval dataloader, we want to record the throughput and power
                # for the current power limit. Since the train dataloader sets the power limit
                # to the optimal power limit right after profiling is done, this will naturally
                # record the tput/power of the optimal PL. From the following epochs where we
                # don't profile anything, we directly use these values to compute the time and
                # energy consumed.
                if self._should_profile:
                    self.eval_tput_result[self.current_gpu_pl] = (
                        self.num_samples / time_consumption
                    )
                    self.eval_power_result[self.current_gpu_pl] = (
                        energy_consumption / time_consumption
                    )
                    # The optimal PL being known means that all power limits have been explored.
                    # Let us end profiling by writing profile information to `power_json`.
                    if self.optimal_pl != 0:
                        self._save_power_results()
            self._log(
                f"{self.split} epoch {self.epoch_num} done: "
                f"time={time_consumption:.2f} energy={energy_consumption:.2f}"
            )

            # Epoch ended in the middle of a profiling window. Reset.
            if self._is_train and self.prof_state != NOT_PROFILING:
                self._end_prof(cutoff=True)

            # Re-raise StopIteration.
            raise

        # We're in the middle of an epoch. The train loader has power limits left to profile.
        if self._is_train and self._should_profile and self._power_limits_left:
            # We weren't doing anything. Start warming up.
            if self.prof_state == NOT_PROFILING:
                self._start_warmup()
            # We're done warming up. Start the actual profiling window.
            elif (
                self.prof_state == WARMING_UP
                and time.monotonic() - self.warmup_start_time >= self.warmup_sec
            ):
                self._start_prof()
            # We're done profiling. Stop the profiling window and gather results.
            elif (
                self.prof_state == PROFILING
                and time.monotonic() - self.prof_start_time >= self.profile_sec
            ):
                self._end_prof(cutoff=False)

        return data
