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

"""Defines the `ZeusMaster` class, which drives batch size optimization and training."""

from __future__ import annotations

import json
import os
import pprint
import subprocess
from copy import deepcopy
from pathlib import Path
from time import localtime, sleep, strftime

import numpy as np
import pynvml
import torch

from zeus.analyze import HistoryEntry
from zeus.job import Job
from zeus.policy import BatchSizeOptimizer
from zeus.util import zeus_cost


class ZeusMaster:
    """Drives Zeus across multiple recurrences of a job.

    The main purpose of `ZeusMaster` is to launch training scripts integrated
    with [`ZeusDataLoader`][zeus.run.ZeusDataLoader], controlling it by setting
    environment variables. For the environment variables, see the
    [`run_job`][zeus.run.ZeusMaster.run_job] method as well as
    [`ZeusDataLoader`][zeus.run.ZeusDataLoader]'s class docstring.

    The optimal batch size is searched for and exploited using the
    [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] object passed in
    through the constructor.
    """

    def __init__(
        self,
        batch_size_optimizer: BatchSizeOptimizer,
        log_base: str,
        monitor_path: str,
        seed: int = 123456,
        observer_mode: bool = False,
        profile_warmup_iters: int = 10,
        profile_measure_iters: int = 40,
    ) -> None:
        """Initialize the master.

        Args:
            batch_size_optimizer: The user is expected to construct the
                [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] with the desired
                policy and pass it into the master class.
            log_base: Absolute path where logs will be stored. A separate directory
                will be created inside, whose name is determined by the job and current time.
            monitor_path: Absolute path to the power monitor binary.
            seed: The random seed. Every invocation of the [`run`][zeus.run.ZeusMaster.run]
                method in this class is deterministic given the random seed, because the
                internal RNG states are deepcopied before servicing jobs.
            observer_mode: When Observer Mode is on, the maximum power limit is
                always used instead of the optimal power limit. However, internal time and
                energy accounting will be done as if the cost-optimal power limit is used.
            profile_warmup_iters: Number of iterations to warm up on a specific power limit.
                This is passed to the [`ZeusDataLoader`][zeus.run.ZeusDataLoader].
            profile_measure_iters: Number of iterations to measure on a specific power limit.
                This is passed to the [`ZeusDataLoader`][zeus.run.ZeusDataLoader].

        """
        # Knob optimizers.
        self.bso = batch_size_optimizer

        # Check if monitor_path is absolute.
        # This is needed since we may change the cwd based on the job's workdir.
        if not Path(monitor_path).is_absolute():
            raise ValueError("monitor_path must be specified as an absolute path.")

        # Log base directory.
        # Needs to be absolute because the training job script may have a different
        # current working directory (when job.workdir is not None).
        if not Path(log_base).is_absolute():
            raise ValueError("log_base must be specified as an absolute path.")
        os.makedirs(log_base, exist_ok=True)
        self.log_base = log_base

        # Save arguments.
        self.seed = seed
        self.monitor_path = monitor_path
        self.observer_mode = observer_mode
        self.profile_warmup_iters = profile_warmup_iters
        self.profile_measure_iters = profile_measure_iters

        # Query the max power limit of the GPU.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        minmax = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)  # unit: mW
        self.max_pl = minmax[1] // 1000  # unit: W
        print(
            f"[Zeus Master] Max power limit of {pynvml.nvmlDeviceGetName(handle)}: {self.max_pl}W"
        )
        pynvml.nvmlShutdown()

    def build_logdir(
        self,
        job: Job,
        num_recurrence: int,
        eta_knob: float,
        beta_knob: float,
        exist_ok: bool = True,
    ) -> str:
        r"""Build the `ZEUS_LOG_DIR` string and create the directory.

        Args:
            job: Job to run.
            num_recurrence: The total number of recurrences.
            eta_knob: $\eta$ used in the cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
            beta_knob: `beta_knob * min_cost` is the early stopping cost threshold.
                Set to `np.inf` to disable early stopping.
            exist_ok: Passed to `os.makedirs`. If `False`, will err if the directory
                already exists.
        """
        now = strftime("%Y%m%d%H%M%s", localtime())
        logdir = (
            job.to_logdir() + f"+x{num_recurrence}+eta{eta_knob}+beta{beta_knob}+{now}"
        )
        logdir = f"{self.log_base}/{logdir}"
        os.makedirs(logdir, exist_ok=exist_ok)
        return logdir

    def run_job(
        self,
        job: Job,
        batch_size: int,
        learning_rate: float,
        seed: int,
        logdir: str,
        rec_i: int,
        tries: int,
        eta_knob: float,
        cost_ub: float,
    ) -> tuple[float, float, bool]:
        r"""Launch the training job.

        Args:
            job: The job to run.
            batch_size: The batch size to use.
            learning_rate: The learning rate to use, scaled based on `batch_size`.
            seed: The random seed to use for training.
            logdir: Directory to store log files in.
            rec_i: Recurrence number of this run of the job.
            tries: Retry number of this recurrence of the job.
            eta_knob: $\eta$ used in the cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
            cost_ub: Cost upper bound. The job is terminated when the next epoch is going
                to exceed the cost upper bound.

        Returns:
            A tuple of energy consumption, time consumption, and whether the job reached the target metric.
        """
        # Generate job command
        command = job.gen_command(batch_size, learning_rate, seed, rec_i)

        # Set environment variables
        job_id = f"rec{rec_i:02d}+try{tries:02d}"
        zeus_env = dict(
            ZEUS_LOG_DIR=logdir,
            ZEUS_JOB_ID=job_id,
            ZEUS_COST_THRESH="inf" if cost_ub == np.inf else str(cost_ub),
            ZEUS_ETA_KNOB=str(eta_knob),
            ZEUS_TARGET_METRIC=str(job.target_metric),
            ZEUS_MONITOR_PATH=self.monitor_path,
            ZEUS_PROFILE_PARAMS=f"{self.profile_warmup_iters},{self.profile_measure_iters}",
            ZEUS_USE_OPTIMAL_PL=str(not self.observer_mode),
        )
        env = deepcopy(os.environ)
        env.update(zeus_env)

        # Training script output captured by the master.
        job_output = f"{logdir}/{job_id}.train.log"

        # Training stats (energy, time, reached, end_epoch) written by ZeusDataLoader.
        # This file being found means that the training job is done.
        train_json = Path(f"{logdir}/{job_id}+bs{batch_size}.train.json")

        # Reporting
        print(f"[run job] Launching job with BS {batch_size}:")
        print(f"[run job] {zeus_env=}")
        if job.workdir is not None:
            print(f"[run job] cwd={job.workdir}")
        print(f"[run job] {command=}")
        print(f"[run job] {cost_ub=}")
        print(f"[run job] Job output logged to '{job_output}'")

        # Run the job.
        with open(job_output, "w") as f:
            # Launch subprocess.
            # stderr is redirected to stdout, and stdout to the job_output file.
            proc = subprocess.Popen(
                command,
                cwd=job.workdir,
                stderr=subprocess.STDOUT,
                stdout=f,
            )

            # Check if training is done.
            with open(job_output, "r") as jobf:
                while proc.poll() is None:
                    print(jobf.read(), end="")
                    sleep(1.0)

                # Print out the rest of the script output.
                f.flush()
                print(jobf.read())

                # Report exitcode.
                exitcode = proc.poll()
                print(f"[run job] Job terminated with exit code {exitcode}.")

            # `train_json` must exist at this point.
            if not train_json.exists():
                raise RuntimeError(f"{train_json} does not exist.")

        # Read `train_json` for the training stats.
        with open(train_json, "r") as f:
            stats = json.load(f)
            print(f"[run job] {stats=}")

        # Casting
        if not isinstance(stats["reached"], bool):
            stats["reached"] = stats["reaached"].lower() == "true"

        return float(stats["energy"]), float(stats["time"]), stats["reached"]

    def run(
        self,
        job: Job,
        num_recurrence: int,
        batch_sizes: list[int],
        beta_knob: float,
        eta_knob: float,
    ) -> list[HistoryEntry]:
        r"""Run a job that sequentially recurs without overlap.

        Args:
            job: The job to run.
            num_recurrence: How many times the job recurs.
            batch_sizes: List of feasible batch sizes.
            beta_knob: `beta_knob * min_eta` is the early stopping cost threshold.
                Set to `np.inf` to disable early stopping.
            eta_knob: $\eta$ used in the cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$

        Returns:
            A list of [`HistoryEntry`][zeus.analyze.HistoryEntry] objects for each job run.
        """
        # Sanity checks
        if job.default_bs is None:
            raise ValueError("You must provide a default batch size for the job.")
        if job.command is None:
            raise ValueError("You must provide a command format string for the job.")
        if eta_knob < 0.0 or eta_knob > 1.0:
            raise ValueError("eta_knob must be in [0.0, 1.0].")

        print(f"[Zeus Master] {job} x {num_recurrence}")
        print(f"[Zeus Master] Batch sizes: {batch_sizes}")

        # Copy all internal state so that simulation does not modify any
        # internal state and is deterministic w.r.t. the random seed.
        bso = deepcopy(self.bso)
        seed = self.seed

        # ZEUS_LOG_DIR: Where all the logs and files are stored for this run.
        logdir = self.build_logdir(job, num_recurrence, eta_knob, beta_knob)

        # Job history list to return.
        history: list[HistoryEntry] = []

        # Save job history to this file, continuously.
        history_file = f"{logdir}/history.py"

        # beta_knob * min_cost is the early stopping cost threshold.
        min_cost = np.inf

        # Register the job in the BSO.
        bso.register_job(job, batch_sizes)

        # Job recurs.
        for rec_i in range(1, num_recurrence + 1):
            print(f"\n[Zeus Master] Recurrence: {rec_i}")

            # The retrying loop. Retry until convergence.
            cost_acc = 0.0
            for tries in range(1, 21):
                # Fetch the knobs to use.
                bs = bso.predict(job)

                # Scale the learning rate.
                lr = job.scale_lr(bs)

                # Launch the job.
                # Power profiling and optimization is done entirely by the ZeusDataLoader.
                # Early stops based on cost_ub.
                energy, time, reached = self.run_job(
                    job=job,
                    batch_size=bs,
                    learning_rate=lr,
                    seed=seed,
                    logdir=logdir,
                    rec_i=rec_i,
                    tries=tries,
                    eta_knob=eta_knob,
                    cost_ub=beta_knob * min_cost,
                )

                # The random seed will be unique for each run, but still jobs will be
                # deterministic w.r.t. each call to `run`.
                seed += 1

                # Compute the cost of this try.
                num_gpus = torch.cuda.device_count()
                cost = zeus_cost(energy, time, eta_knob, self.max_pl * num_gpus)
                print(f"[Zeus Master] {cost=}")

                # Accumulate the cost to track the total cost of this recurrence.
                cost_acc += cost

                # Provide feedback to the BSO.
                bso.observe(job, bs, cost, reached)

                # Record history for visualization.
                history.append(HistoryEntry(bs, None, energy, reached, time))
                with open(history_file, "w") as f:
                    # Intended use:
                    #
                    # ```python
                    # from zeus.analyze import HistoryEntry
                    # history = eval(open(history_file).read())
                    # ```
                    f.write(pprint.pformat(history) + "\n")

                # Reached the target metric. Go to next recurrence.
                if reached:
                    print(
                        "\n[Zeus Master] Reached target metric in "
                        f"{tries} {'try' if tries == 1 else 'tries'}."
                    )
                    # Track the minimum cost.
                    if min_cost > cost_acc:
                        print(
                            f"\n[Zeus Master] Minimum cost updated from {min_cost} to {cost_acc}."
                        )
                        min_cost = cost_acc
                    break
                # Didn't reach the target metric.
                # We assume that the default BS (set by the user) will converge.
                if rec_i == 1:
                    raise RuntimeError(
                        f"The default batch size {job.default_bs} did not converge."
                    )
            else:
                print(
                    "\n[Zeus Master] Job did not reach the target metric in 20 trials!"
                )
                raise RuntimeError("Unreachable target metric.")

        print(f"[Zeus Master]\n{history}")

        return history
