"""A simulator for running trace-driven Zeus experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from zeus._legacy.policy import PowerLimitOptimizer
from zeus._legacy.simulate import HistoryEntry
from zeus._legacy.job import Job
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobSpecFromClient,
    TrainingResult,
)
from zeus.utils.metric import zeus_cost


class BatchSizeOptimizerDummyClient:
    def __init__(self, url=""):
        self.url = url
        self.trial_number = 0

    def register_job(self, job: JobSpecFromClient):
        res = httpx.post(self.url + REGISTER_JOB_URL, content=job.json())
        assert res.status_code == 200 or res.status_code == 201, res.text

    def predict(self, job_id: str):
        res = httpx.get(self.url + GET_NEXT_BATCH_SIZE_URL, params={"job_id": job_id})
        bs = res.json()["batch_size"]
        self.trial_number = res.json()["trial_number"]
        return bs

    def observe(
        self,
        job: JobSpecFromClient,
        batch_size: int,
        total_energy: float,
        time: float,
        max_power: int,
        converged: bool,
        epoch: int,
    ):
        training_result = TrainingResult(
            job_id=job.job_id,
            batch_size=batch_size,
            trial_number=self.trial_number,
            error=False,
            time=time,
            energy=total_energy,
            max_power=max_power,
            metric=job.target_metric + 1 if converged else job.target_metric - 1,
            current_epoch=epoch,
        )

        # report to the server about the result of this training
        res = httpx.post(self.url + REPORT_RESULT_URL, content=training_result.json())


# ruff: noqa: PLR0912, PLR0915
class SimulatorWithServer:
    """Simulates job execution optimized by Zeus."""

    def __init__(
        self,
        summary_train: str | pd.DataFrame,
        summary_power: str | pd.DataFrame,
        power_limit_optimizer: PowerLimitOptimizer,
        gpu: Literal["a40", "v100", "p100", "rtx6000"],
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
        # self.bso = batch_size_optimizer
        self.plo = power_limit_optimizer
        self.bso = BatchSizeOptimizerDummyClient()

        # Save arguments.
        self.seed = seed
        self.verbose = verbose
        self.gpu = gpu

    def simulate_one_job(
        self,
        job: Job,  #  Use this to create a job batch_sizes = self._profile_batch_size_range(job)
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
            A list of [`HistoryEntry`][zeus._legacy.simulate.HistoryEntry] objects for each job run.
        """
        # Figure out MAXPOWER.
        max_power = self.df.power_limit.max().item()
        if self.verbose:
            print(f"[Simulator] Max power = {max_power}W")

        # Copy all internal state so that simulation does not modify any
        # internal state and is deterministic w.r.t. the random seed.
        # A new job. Profile the feasible batch size range.
        jobSpec = JobSpecFromClient(
            job_id="simulation-one-job",
            job_id_prefix="simulation",
            batch_sizes=self._profile_batch_size_range(job),
            default_batch_size=job.default_bs,
            target_metric=job.target_metric,
            max_epochs=job.max_epochs,
            beta_knob=beta_knob,
            eta_knob=eta_knob,
            mab_seed=self.seed,
            max_power=max_power,
            gpu_model=self.gpu,
            number_of_gpus=1,
            window_size=0,
        )

        # register job in the server
        self.bso.register_job(jobSpec)

        ## Should be mocked
        plo = deepcopy(self.plo)
        rng = np.random.default_rng(self.seed)

        # Track the minimum cost observed for the early stopping energy threshold.
        min_cost = np.inf

        # Simulate each job one at a time.
        history: list[HistoryEntry] = []

        if self.verbose:
            print(f"[Simulator] {job} recurring {num_recurrence} times.")

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
                bs = self.bso.predict(jobSpec.job_id)
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
                eta, tta, reached, epoch = self._run_job(
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
                    # bso.observe(job, bs, 100 * beta_knob * min_cost, False)
                    self.bso.observe(jobSpec, bs, eta, tta, max_power, False, epoch)
                    continue

                # Compute the hybrid cost metric.
                cost = zeus_cost(eta, tta, eta_knob, max_power)
                cost_acc += cost

                # Provide feedback to the BSO.
                # bso.observe(job, bs, cost, reached)
                self.bso.observe(jobSpec, bs, eta, tta, max_power, reached, epoch)

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

    def _run_job(
        self,
        job: Job,
        batch_size: int,
        power_limit: int,
        rng: np.random.Generator,
        cost_ub: float,
        eta_knob: float,
        profile_power: bool,
    ) -> tuple[float, float, bool, int]:
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
            Tuple of energy consumption, time consumption, whether the job reached the target metric, and max_epochs indicating how many epochs we ran.
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

        # The job virtually never ran. Time and Energy being zero will be treated specially outside.
        # If the min_cost is so low, this might even prevent this BS from running at all.
        if max_epochs == 0:
            eta, tta = compute_energy_and_time(max_epochs + 1, profile_power)
            print(
                f"[run job] {job} cannot run even one epoch without exceeding the cost UB."
                f" BS {batch_size}, PL {power_limit}, {eta_knob=}"
            )
            return eta, tta, False, max_epochs + 1

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
            return eta, tta, True, max_epochs

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
        return (
            energy_consumption,
            time_consumption,
            False,
            job.max_epochs,
        )  # reached max epoch or the next epoch will reach cost ub. -> server will give another chance if epoch is less than max epoch

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
        df = df.groupby(["power_limit"], as_index=False).mean(numeric_only=True)
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
