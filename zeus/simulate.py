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

"""A simulator for running trace-driven Zeus experiments."""

from __future__ import annotations

import operator
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from zeus.analyze import HistoryEntry
from zeus.job import Job
from zeus.policy import BatchSizeOptimizer, PowerLimitOptimizer
from zeus.util import zeus_cost


class Simulator:
    """Simulates job execution optimized by Zeus."""

    def __init__(
        self,
        summary_train: str | pd.DataFrame,
        summary_power: str | pd.DataFrame,
        batch_size_optimizer: BatchSizeOptimizer,
        power_limit_optimizer: PowerLimitOptimizer,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialize the simulator.

        Args:
            summary_train: Path to or `pd.DataFrame` of the train trace.
            summary_power: Path to or `pd.DataFrame` of the power trace.
            batch_size_optimizer: The user is expected to construct
                the BSO with the desired policy and pass it into the simulator.
            power_limit_optimizer: The user is expected to construct
                the PLO with the desired policy and pass it into the simulator.
            seed: The random seed. Every invocation of any simulation method in this
                class is deterministic given the random seed, because the internal RNG is
                deepcopied before running the simulation.
            verbose: Whether to log out the internal states of the simulator.
        """
        # Generate relevant data.
        train_df = (
            pd.read_csv(summary_train)
            if isinstance(summary_train, str)
            else summary_train
        )
        power_df = (
            pd.read_csv(summary_power)
            if isinstance(summary_power, str)
            else summary_power
        )
        df = train_df.merge(power_df, how="inner")  # type: ignore
        df["TTA"] = df.target_epoch * df.time_per_epoch
        df["ETA"] = df.TTA * df.average_power
        # 'energy_per_epoch' is used to compare different power limits with the same batch size
        # when trying to figure out which power limit is the best one.
        df["energy_per_epoch"] = df.time_per_epoch * df.average_power
        self.df = df

        # Knob optimizers.
        self.bso = batch_size_optimizer
        self.plo = power_limit_optimizer

        # Save arguments.
        self.seed = seed
        self.verbose = verbose

    def simulate_one_job(
        self,
        job: Job,
        num_recurrence: int,
        beta_knob: float,
        eta_knob: float,
    ) -> list[HistoryEntry]:
        r"""Simulate a sequentially recurring job. Explore with early stopping.

        Args:
            job: Job spec to simulate.
            num_recurrence: How many times the job recurs.
            beta_knob: `beta_knob * min_eta` is the early stopping cost threshold.
                Set to `np.inf` to disable early stopping.
            eta_knob: $\eta$ used in the hybrid cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$

        Returns:
            A list of [`HistoryEntry`][zeus.analyze.HistoryEntry] objects for each job run.
        """
        # Copy all internal state so that simulation does not modify any
        # internal state and is deterministic w.r.t. the random seed.
        bso = deepcopy(self.bso)
        plo = deepcopy(self.plo)
        rng = np.random.default_rng(self.seed)

        # Figure out MAXPOWER.
        max_power = self.df.power_limit.max().item()
        if self.verbose:
            print(f"[Simulator] Max power = {max_power}W")

        # Track the minimum cost observed for the early stopping energy threshold.
        min_cost = np.inf

        # Simulate each job one at a time.
        history: list[HistoryEntry] = []

        if self.verbose:
            print(f"[Simulator] {job} recurring {num_recurrence} times.")

        # A new job. Profile the feasible batch size range.
        batch_sizes = self._profile_batch_size_range(job)

        # Register the job in the BSO.
        bso.register_job(job, batch_sizes)

        # Job recurs.
        for i in range(num_recurrence):
            if self.verbose:
                print(f"\nRecurrence: {i+1}")

            # Run the job until convergence. Upper bound the number of retries to 20.
            # Accumulate the cost of retries before convergence.
            cost_acc = 0.0
            for tries in range(1, 21):
                # Whether this run of the job needed to profile power.
                profiled_power = False

                # Fetch knobs to use.
                bs = bso.predict(job)
                pl = plo.predict(job, bs)

                # When the batch size is first explored, we need to profile power limit.
                if pl is None:
                    profiled_power = True
                    result = self._profile_power_limit(job, bs, eta_knob)
                    for pl, epe in result.items():
                        plo.observe(job, bs, pl, epe)
                    pl = plo.predict(job, bs)
                    assert pl

                # Run the job, potentially early stopping it.
                eta, tta, reached = self._run_job(
                    job=job,
                    batch_size=bs,
                    power_limit=pl,
                    rng=rng,
                    cost_ub=beta_knob * min_cost,
                    eta_knob=eta_knob,
                    profile_power=profiled_power,
                )

                # The job never ran because even one epoch exceeds the cost threshold.
                # Let the BSO observe that this batch size is bad, but since the job
                # did not run, do not add to the history and retry.
                if eta == 0 and tta == 0 and not reached:
                    bso.observe(job, bs, 100 * beta_knob * min_cost, False)
                    continue

                # Compute the hybrid cost metric.
                cost = zeus_cost(eta, tta, eta_knob, max_power)
                cost_acc += cost

                # Provide feedback to the BSO.
                bso.observe(job, bs, cost, reached)

                # Record history for analysis.
                history.append(HistoryEntry(bs, pl, eta, reached, tta))

                # Reached the target metric. Update min_cost and go on to the next recurrence.
                if reached:
                    if self.verbose:
                        print()
                        print(
                            f"[Simulator] Reached target metric in {tries} {'try' if tries == 1 else 'tries'}."
                        )
                    if min_cost > cost_acc:
                        if self.verbose:
                            print(
                                f"[Simulator] Minimum cost updated from {min_cost:.2f} to {cost_acc:.2f}."
                            )
                        min_cost = cost_acc
                    break
                # Didn't reach the target metric.
                # We assume that the default BS (set by the user) will always converge.
                # That is, reaching the target metric with the model should be a feasible task.
                if i == 0:
                    raise RuntimeError(
                        f"The default batch size {job.default_bs} did not converge."
                    )

            # Target metric was not reached in 20 tries. We consider this target metric to be unreachable.
            else:
                raise RuntimeError("Job did not reach the target metric in 20 tries.")

        if self.verbose:
            print()
            print(
                f"[Simulator] {job} (BS, PL, ETA, whether_reached, TTA) history: \n{history}"
            )

        return history

    def simulate_one_alibaba_group(
        self,
        job: Job,
        group_df: pd.DataFrame,
        beta_knob: float,
        eta_knob: float,
    ) -> list[HistoryEntry]:
        r"""Run simulation on one group in the Alibaba trace.

        Concurrent job submissions (jobs that start before the previous job
        finishes) are launched with the batch size known to be of minimum
        cost at that time. The BSO also observes the results of these jobs
        when they are done, and these jobs may well finish before a job that
        started before it. See `observe` in PruningGTSBatchSizeOptimizer for
        an example of handing such a scenario.

        Args:
            job: Job spec of this group.
            group_df: DataFrame of this group. Fields:
                - group: Group ID in trace. Identical across all rows.
                - dataset: Dataset name. Identical across all rows.
                - start_time: Job start time in the trace.
                - end_time: Job end time in the trace.
                - runtime_ratio: runtime of this job over the mean runtime
                    of all the jobs of this dataset. Captures intra-dataset
                    job runtime differences.
            beta_knob: `beta_knob * min_eta` is the early stopping cost threshold.
                Set to `np.inf` to disable early stopping.
            eta_knob: $\eta$ used in the hybrid cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$

        Returns:
            A list of [`HistoryEntry`][zeus.analyze.HistoryEntry] objects for each job run.
        """
        # Copy all internal state so that simulation does not modify any
        # internal state and is deterministic w.r.t. the random seed.
        bso = deepcopy(self.bso)
        plo = deepcopy(self.plo)
        rng = np.random.default_rng(self.seed)

        # Sanity check
        if job.default_bs is None:
            raise ValueError("You must give the job a default batch size.")

        # Figure out MAXPOWER.
        max_power = self.df.power_limit.max().item()
        if self.verbose:
            print(f"[Simulator] Max power = {max_power}W")

        # Track the minimum cost observed for the early stopping energy threshold.
        # Also track the corresponding minimum cost BS to handle concurrent jobs.
        min_cost = np.inf
        best_bs = job.default_bs

        # Simulate each job one at a time.
        history: list[HistoryEntry] = []

        if self.verbose:
            print(f"[Simulator] {job} recurring {len(group_df)} times.")

        # A new job. Profile the feasible batch size range.
        batch_sizes = self._profile_batch_size_range(job)

        # Register the job in the BSO.
        bso.register_job(job, batch_sizes)

        # List of jobs in flight.
        @dataclass
        class RunningJob:
            """Represents a job that is running.

            We know what's going to happen to this job when we launch it.
            Thus, pre-compute all results (using self.run_job) and have this
            instance hold the information. Then, jobs will be removed from the
            list of running jobs when the current time passes self.end_time and
            the result will be observed.
            """

            start_time: float
            end_time: float
            runtime_ratio: float
            batch_size: int
            power_limit: int
            reached: bool
            time: float
            energy: float
            cost: float

        running_jobs: list[RunningJob] = []

        # Jobs in group_df are already sorted by start_time.
        current_time = 0.0
        for rec_i, (_, job_row) in enumerate(group_df.iterrows()):
            if self.verbose:
                print(f"\nRecurrence: {rec_i+1}")

            # Update the current time.
            current_time = job_row.start_time
            if self.verbose:
                print(f"[Simulator] Current time is {current_time:.1f}")

            # Scan the in-flight job list to reap jobs that have finished.
            # We need a while loop here because we might have submitted a retry job
            # while reaping jobs that failed to reach the target metric, and that retry job
            # may finish before the current job.
            # pylint: disable=cell-var-from-loop
            while any(map(lambda j: j.end_time <= current_time, running_jobs)):
                if self.verbose:
                    print(f"[Simulator] Running jobs: {running_jobs}")

                # We copy running_jobs because we want to mutate it as we traverse it.
                running_jobs_copy = deepcopy(running_jobs)

                # Sort the jobs in the order they end.
                for rjob in sorted(
                    running_jobs_copy, key=operator.attrgetter("end_time")
                ):
                    # We're only interested in jobs that finished at this point in time.
                    if rjob.end_time > current_time:
                        continue

                    # Job is finished at this point in time.
                    if self.verbose:
                        print(
                            f"[Simulator] Job(BS={rjob.batch_size},time={rjob.time},"
                            f"energy={rjob.energy},reached={rjob.reached}) finished"
                        )

                    # Remove the job from the in-flight job list.
                    running_jobs.remove(rjob)

                    # Let the BSO observe the cost for this batch size.
                    bso.observe(job, rjob.batch_size, rjob.cost, rjob.reached)

                    # If the job never ran because even one epoch exceeds the cost threshold,
                    # do not add to the history and retry.
                    if rjob.energy != 0 or rjob.time != 0 or rjob.reached:
                        # Record history for analysis.
                        history.append(
                            HistoryEntry(
                                rjob.batch_size,
                                rjob.power_limit,
                                rjob.energy
                                * rjob.runtime_ratio,  # Scale the energy of this job by the runtime ratio.
                                rjob.reached,
                                rjob.time
                                * rjob.runtime_ratio,  # Scale the runtime of this job by the runtime ratio.
                            )
                        )

                    # Reached target metric (no need to retry)
                    if rjob.reached:
                        if min_cost > rjob.cost:
                            if self.verbose:
                                print(
                                    f"[Simulator] Minimum cost updated from {min_cost:.2f} to {rjob.cost:.2f}"
                                )
                            min_cost = rjob.cost
                            best_bs = rjob.batch_size

                    # Didn't reach target metric. Need to run a retry job.
                    else:
                        profiled_power = False

                        # If there are jobs in-flight, we just run additional concurrent
                        # submissions with the best known knobs.
                        if running_jobs:
                            if self.verbose:
                                print(
                                    f"[Simulator] There are in-flight jobs. Use BS {best_bs}."
                                )
                            bs = best_bs
                            pl = plo.predict(job, bs)
                            assert pl, f"Power not profiled for best known BS {bs}"

                        # There are no jobs in-flight. Just submit the job normally.
                        else:
                            # Determine the knobs.
                            bs = bso.predict(job)
                            pl = plo.predict(job, bs)

                            if self.verbose:
                                print(
                                    f"[Simulator] There are no in-flight jobs. Use BSO's prediction {bs}."
                                )

                            # When the batch size is first explored, we need to profile power limit.
                            if pl is None:
                                profiled_power = True
                                result = self._profile_power_limit(job, bs, eta_knob)
                                for pl, epe in result.items():
                                    plo.observe(job, bs, pl, epe)
                                pl = plo.predict(job, bs)
                                assert pl

                        # Pre-compute the result of the job.
                        eta, tta, reached = self._run_job(
                            job=job,
                            batch_size=bs,
                            power_limit=pl,
                            rng=rng,
                            cost_ub=beta_knob * min_cost,
                            eta_knob=eta_knob,
                            profile_power=profiled_power,
                        )

                        # Compute the hybrid cost metric.
                        cost = zeus_cost(eta, tta, eta_knob, max_power)

                        # Create the RunningJob instance.
                        running_job = RunningJob(
                            start_time=rjob.end_time,
                            end_time=rjob.end_time
                            + (rjob.end_time - rjob.start_time),  # Assume same runtime.
                            runtime_ratio=rjob.runtime_ratio,
                            batch_size=bs,
                            power_limit=pl,
                            reached=reached,
                            time=tta,
                            energy=eta,
                            cost=cost,
                        )
                        running_jobs.append(running_job)

                        if self.verbose:
                            print(f"[Simulator] Started retry job {running_job}")

                        # We must break from the loop that scans the running_jobs list, because
                        # the job we just submitted might actually be the next job that finishes.
                        break

            # We're (finally) done reaping all finished jobs. Run the current job.
            profiled_power = False

            # If there are jobs in-flight, we just run additional concurrent
            # submissions with the best known knobs.
            if running_jobs:
                if self.verbose:
                    print(f"[Simulator] There are in-flight jobs. Use BS {best_bs}.")
                bs = best_bs
                pl = plo.predict(job, bs)
                assert pl is not None, f"Power not profiled for best known BS {bs}"

            # There are no jobs in-flight. Just submit the job.
            else:
                # Determine the knobs.
                bs = bso.predict(job)
                pl = plo.predict(job, bs)

                if self.verbose:
                    print(
                        f"[Simulator] There are no in-flight jobs. Use BSO's prediction {bs}."
                    )

                # When the batch size is first explored, we need to profile power limit.
                if pl is None:
                    profiled_power = True
                    result = self._profile_power_limit(job, bs, eta_knob)
                    for pl, epe in result.items():
                        plo.observe(job, bs, pl, epe)
                    pl = plo.predict(job, bs)
                    assert pl

            # Run the job, potentially early stopping it.
            eta, tta, reached = self._run_job(
                job=job,
                batch_size=bs,
                power_limit=pl,
                rng=rng,
                cost_ub=beta_knob * min_cost,
                eta_knob=eta_knob,
                profile_power=profiled_power,
            )

            # Compute the hybrid cost metric.
            cost = zeus_cost(eta, tta, eta_knob, max_power)

            # Create the RunningJob instance and enqueue.
            running_job = RunningJob(
                start_time=job_row.start_time,
                end_time=job_row.end_time,
                runtime_ratio=job_row.runtime_ratio,
                batch_size=bs,
                power_limit=pl,
                reached=reached,
                time=tta,
                energy=eta,
                cost=cost,
            )
            running_jobs.append(running_job)

            if self.verbose:
                print(f"[Simulator] Started job {running_job}")

        # Now, we're done iterating the rows of group_df.
        # Set the current time to infinity so that all running jobs finish.
        current_time = np.inf
        if self.verbose:
            print("\n[Simulator] Reap all jobs")

        # Scan the remaining in-flight job list to reap jobs that have finished.
        # Since current_time is infinity, this while loop will reap all the jobs ever launched.
        while any(map(lambda j: j.end_time <= current_time, running_jobs)):
            if self.verbose:
                print(f"[Simulator] Running jobs: {running_jobs}")

            # We copy running_jobs because we want to mutate it as we traverse it.
            running_jobs_copy = deepcopy(running_jobs)

            # Sort the jobs in the order they end.
            for rjob in sorted(running_jobs_copy, key=operator.attrgetter("end_time")):
                # We're only interested in jobs that finished at this point in time.
                if rjob.end_time > current_time:
                    continue

                # Job is finished at this point in time.
                if self.verbose:
                    print(
                        f"[Simulator] Job(BS={rjob.batch_size},time={rjob.time},"
                        f"energy={rjob.energy},reached={rjob.reached}) finished"
                    )

                # Remove the job from the in-flight job list.
                running_jobs.remove(rjob)

                # Let the BSO observe the cost for this batch size.
                bso.observe(job, rjob.batch_size, rjob.cost, rjob.reached)

                # If the job never ran because even one epoch exceeds the cost threshold,
                # do not add to the history and retry.
                if rjob.energy != 0 or rjob.time != 0 or rjob.reached:
                    # Record history for analysis.
                    history.append(
                        HistoryEntry(
                            rjob.batch_size,
                            rjob.power_limit,
                            rjob.energy
                            * rjob.runtime_ratio,  # Scale the energy of this job by the runtime ratio.
                            rjob.reached,
                            rjob.time
                            * rjob.runtime_ratio,  # Scale the runtime of this job by the runtime ratio.
                        )
                    )

                # Reached target metric (no need to retry)
                if rjob.reached:
                    if min_cost > rjob.cost:
                        if self.verbose:
                            print(
                                f"[Simulator] Minimum cost updated from {min_cost:.2f} to {rjob.cost:.2f}"
                            )
                        min_cost = rjob.cost
                        best_bs = rjob.batch_size

                # Didin't reach target metric. Need to run a retry job.
                else:
                    profiled_power = False

                    # If there are jobs in-flight, we just run additional concurrent
                    # submissions with the best known knobs.
                    if running_jobs:
                        if self.verbose:
                            print(
                                f"[Simulator] There are in-flight jobs. Use BS {best_bs}."
                            )
                        bs = best_bs
                        pl = plo.predict(job, bs)
                        assert pl, f"Power not profiled for best known BS {bs}"

                    # There are no jobs in-flight. Just submit the job normally.
                    else:
                        # Determine the knobs.
                        bs = bso.predict(job)
                        pl = plo.predict(job, bs)

                        if self.verbose:
                            print(
                                f"[Simulator] There are no in-flight jobs. Use BSO's prediction {bs}."
                            )

                        # When the batch size is first explored, we need to profile power limit.
                        if pl is None:
                            profiled_power = True
                            result = self._profile_power_limit(job, bs, eta_knob)
                            for pl, epe in result.items():
                                plo.observe(job, bs, pl, epe)
                            pl = plo.predict(job, bs)
                            assert pl

                    # Pre-compute the result of the job.
                    eta, tta, reached = self._run_job(
                        job=job,
                        batch_size=bs,
                        power_limit=pl,
                        rng=rng,
                        cost_ub=beta_knob * min_cost,
                        eta_knob=eta_knob,
                        profile_power=profiled_power,
                    )

                    # Compute the hybrid cost metric.
                    cost = zeus_cost(eta, tta, eta_knob, max_power)

                    # Create the RunningJob instance.
                    running_job = RunningJob(
                        start_time=rjob.end_time,
                        end_time=rjob.end_time
                        + (rjob.end_time - rjob.start_time),  # Assume same runtime.
                        runtime_ratio=rjob.runtime_ratio,
                        batch_size=bs,
                        power_limit=pl,
                        reached=reached,
                        time=tta,
                        energy=eta,
                        cost=cost,
                    )
                    running_jobs.append(running_job)

                    if self.verbose:
                        print(f"[Simulator] Started retry job {running_job}")

                    # We must break from the loop that scans the running_jobs list, because
                    # the job we just submitted might actually be the next job that finishes.
                    break

        return history

    def _run_job(
        self,
        job: Job,
        batch_size: int,
        power_limit: int,
        rng: np.random.Generator,
        cost_ub: float,
        eta_knob: float,
        profile_power: bool,
    ) -> tuple[float, float, bool]:
        r"""Simulate running the job and return the energy consumed and whether it converged.

        This method will randomly choose one of the possible training "paths". Then,
        based on cost_ub, it will compute the maximum number of epochs the job can run.
        If the path's target_epoch is smaller than or equal to the maximum number of
        epochs, the cost incurred until target_epoch is returned. Otherwise, the cost
        incurred until the maximum number of epochs is returned.

        It is important to note that the job may never run when the first epoch's cost
        is already expected to exceed the cost upper bound. In such a case, the returned
        time and energy consumptions will be zero. This case must be treated separately
        in the calling code.

        If profile_power is True, the first epoch will JIT-profile power limits coarsely
        by dividing the epoch evenly into len(available_power_limits) pieces. Thus the
        the first epoch's energy and time consumption will be slightly adjusted.

        Args:
            job: Job spec to run.
            batch_size: Batch size to use.
            power_limit: Power limit to use. Regardless of whether this run of this
                batch size requires power profiling, the simulator will input the optimal
                power limit for the batch size. The first epoch energy consumption from
                power profiling is adjusted in the latter half of this method based on the
                profile_power flag.
            rng: Random number generator used to sample one training path.
            cost_ub: Cost upper bound. The job is terminated when the next epoch is going
                to exceed the cost upper bound.
            eta_knob: $\eta$ used in the hybrid cost metric.
                $\textrm{cost} = \eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}$
            profile_power: Whether this run of the job should profile power during the
                first epoch.

        Returns:
            Tuple of energy consumption, time consumption, and whether the job reached the target metric.
        """
        # df is filtered with job spec, BS, and PL. We sample one possible training path.
        # power_df is filtered with job spec and BS. We use this to compute the energy
        # consumption of profiling power during the first epoch.
        df = job.filter_df(self.df)
        power_df = df.loc[df.batch_size == batch_size]
        df = power_df.loc[df.power_limit == power_limit]
        path = df.sample(n=1, random_state=rng)

        # Max number of epochs is bound by either the cost upper bound or the user-specified
        # max_epochs, whichever is smaller.
        if cost_ub == np.inf:
            # cost_ub is infinity in two cases:
            # 1. The simulator has never observed any cost value in the early part of simulation.
            # 2. We're simulating with no early stopping, i.e. beta_knob is infinity.
            max_epochs = job.max_epochs
            if self.verbose:
                print(f"[run job] Cost UB is inf. {max_epochs=}")
        else:
            # Stop right before the first epoch when cost will cross the upper bound.
            cost_per_epoch = (
                eta_knob * path.energy_per_epoch.item()
                + (1 - eta_knob)
                * power_df.power_limit.max().item()
                * path.time_per_epoch.item()
            )
            max_epochs = min(cost_ub // cost_per_epoch, job.max_epochs)
            if self.verbose:
                print(f"[run job] {cost_ub=}")
                print(f"[run job] {cost_per_epoch=}")
                print(f"[run job] {max_epochs=}")

        # The job virtually never ran. Time and Energy being zero will be treated specially outside.
        # If the min_cost is so low, this might even prevent this BS from running at all.
        if max_epochs == 0:
            print(
                f"[run job] {job} cannot run even one epoch without exceeding the cost UB."
                f" BS {batch_size}, PL {power_limit}, {eta_knob=}"
            )
            return 0.0, 0.0, False

        def compute_energy_and_time(
            num_epochs: int, profile_power: bool
        ) -> tuple[float, float]:
            """Compute the energy and time consumed for running the job for num_epochs."""
            # This is the first run of this batch size, and we need to profile power
            # during the first epoch.
            if profile_power:
                # Note that power_df holds rows with all power limits. Evenly splitting the
                # epochs with the number of samples and running each slice with each power
                # limit consumes (1/N) * e_100 + (1/N) * e_125 + ... + (1/N) * e_250.
                # Also there are all runs 1, 2, ... included, but power info is actually
                # completely duplicated across different runs in the DataFrame.
                # Thus, taking the mean across the entire power_df gets us what we want.
                energy_first_epoch = power_df.energy_per_epoch.mean().item()
                energy_from_second_epoch = path.energy_per_epoch.item() * (
                    num_epochs - 1
                )
                energy_consumption = energy_first_epoch + energy_from_second_epoch
                time_first_epoch = power_df.time_per_epoch.mean().item()
                time_from_second_epoch = path.time_per_epoch.item() * (num_epochs - 1)
                time_consumption = time_first_epoch + time_from_second_epoch
            # Just run num_epochs with the given power limit. Simple.
            else:
                energy_consumption = path.energy_per_epoch.item() * num_epochs
                time_consumption = path.time_per_epoch.item() * num_epochs
            return energy_consumption, time_consumption

        # Job reached target metric.
        target_epoch = path.target_epoch.item()
        if path.target_epoch.notnull().item() and target_epoch <= max_epochs:
            eta, tta = compute_energy_and_time(target_epoch, profile_power)
            if self.verbose:
                print(
                    f"[run job] {job} @ {batch_size},{power_limit}W{' prof' if profile_power else ''} "
                    f"=> \033[31mReached in {int(target_epoch)} epochs, "
                    f"TTA {tta:.2f} seconds, ETA {eta:.2f}\033[0m"
                )
            return eta, tta, True

        # Job failed to reach the target metric.
        energy_consumption, time_consumption = compute_energy_and_time(
            max_epochs, profile_power
        )
        if self.verbose:
            print(
                f"[run job] {job} @ {batch_size},{power_limit}W{' prof' if profile_power else ''} "
                f"=> \033[31mFailed (stopped after {int(max_epochs)} epochs), "
                f"TTA {time_consumption:.2f} seconds, ETA {energy_consumption:.2f}\033[0m"
            )
        return energy_consumption, time_consumption, False

    def _profile_power_limit(
        self, job: Job, batch_size: int, eta_knob: float
    ) -> dict[int, float]:
        """Simulate running the job and profiling the power limit.

        Returns:
            Dictionary mapping PL to `energy_per_epoch`. PL is inserted in high to low order.
        """
        # Filter by job spec and BS.
        df = job.filter_df(self.df)
        df = df.loc[(df.batch_size == batch_size)]

        # Compute the epoch cost of each power limit (Equation 7).
        max_pl = df.power_limit.max().item()
        df = df.groupby(["power_limit"], as_index=False).mean()
        df["epoch_cost"] = (
            eta_knob * df["average_power"] + (1 - eta_knob) * max_pl
        ) * df["time_per_epoch"]

        # We'll be profiling energy from larger to smaller power limits.
        df = df.sort_values(by="power_limit", ascending=False)
        result = {rec.power_limit: rec.epoch_cost for rec in df.to_records(index=False)}
        if self.verbose:
            print(f"[PL profile] {job} @ {batch_size} => PL = {min(result, key=result.get)}W")  # type: ignore
        return result

    def _profile_batch_size_range(self, job: Job) -> list[int]:
        """Simulate profiling the available batch size range of the job.

        Returns:
            A list of feasible batch sizes.
        """
        df = self.df
        # Do not filter by target_metric here since we do not want to constrain
        # the feasible batch size range to only those that reached the target metric.
        df = df.loc[
            (df.dataset == job.dataset)
            & (df.network == job.network)
            & (df.optimizer == job.optimizer)
        ]
        result = sorted(list(df.batch_size.unique()))
        if self.verbose:
            print(f"[BS profile] {job} => BS = {result}")
        return result
