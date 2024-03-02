from __future__ import annotations
import asyncio

from collections import defaultdict
import json
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from zeus.optimizer.batch_size.common import (
    JobSpec,
    MabSetting,
    ReportResponse,
    Stage,
    TrainingResult,
    ZeusBSOJobSpecMismatch,
    ZeusBSOOperationOrderError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.database.dbapi import DBapi
from zeus.optimizer.batch_size.server.database.models import (
    ExplorationState,
    GaussianTsArmState,
    Job,
    Measurement,
)
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.util.metric import zeus_cost

# from dbapis import DBAPI

GLOBAL_ZEUS_SERVER: ZeusBatchSizeOptimizer | None = None


class ZeusBatchSizeOptimizer:
    """A singleton class that manages training jobs. Why singleton? Not sure yet.
    (1) Does this class maintain any state?
    Or (2)are we storing everything in DB and everytime we call a function, it just grab values from DB and compute

    Pruning stage will be handled by throwing an error such as cannot find batch size that converges.
    User provides max Epoch, and if the cost is too much or cannot achieve target accuracy, we throw convergence fail.

    So all we need to do is, In DB, we specify if we are in pruning stage(filterting batch sizes that cannot achieve target acc)
    OR training stage(have some batch sizes that can achieve target accuracy)
    """

    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        """Initialize the server."""
        self.verbose = verbose
        self.mab = GaussianTS(verbose)

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        return "Pruning GaussianTS BSO"

    async def register_job(self, job: JobSpec, db: AsyncSession) -> bool:
        """Register a user-submitted job. Return number of newly created job. Return the number of job that is registered"""
        registered_job = await DBapi.get_job(db, job.job_id)

        if registered_job is not None:
            # Job exists
            if self.verbose:
                self._log(f"Job({job.job_id}) already exists")
            equality = await registered_job.equal_to(job)
            if not equality:
                raise ZeusBSOJobSpecMismatch(
                    "JobSpec doesn't match with existing jobSpec. Use a new job_id for different configuration"
                )
            return False

        DBapi.add_job(db, job)
        if self.verbose:
            self._log(f"Registered {job.job_id}")

        return True

    async def test(self, db: AsyncSession, job_id: UUID):
        tr = TrainingResult(
            job_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
            batch_size=1024,
            time=12321,
            energy=3000,
            max_power=300,
            metric=0.55,
            current_epoch=98,
        )
        db.add(
            Measurement(
                job_id=tr.job_id,
                batch_size=tr.batch_size,
                time=tr.time,
                energy=tr.energy,
                converged=False,
            )
        )
        # await DBapi.insert_measurement(db, tr, False)
        res = await DBapi.get_measurements_of_bs(db, tr.job_id, 10, 1024)
        print("Test res", str(res[0]), len(res))

    async def predict(self, db: AsyncSession, job_id: UUID) -> int:
        """return a batch size to use. Probably get the MAB from DB? then do some computation
        Return the batch size to use for the job."""
        # Try to see if the exploration manager has something.
        job = await DBapi.get_job_with_explorations(db, job_id)

        if job == None:
            raise ZeusBSOValueError(
                f"Unknown job({job_id}). Please register the job first"
            )

        batch_size = await PruningExploreManager.next_batch_size(
            db,
            job_id,
            job.batch_sizes,
            job.num_pruning_rounds,
            job.exp_default_batch_size,
        )

        if isinstance(batch_size, Stage) and batch_size == Stage.Exploration:
            return (await job.get_min_cost())[1]
        elif isinstance(batch_size, Stage):
            # MAB stage
            # Construct MAB (based on most recent trials pick bs that is converged and cost is under cost_ub)
            mab_setting = job.get_mab_setting()
            arms = job.arm_states
            if len(arms) == 0:  # MAB is not constructed
                arms = await self._construct_mab(db, job, mab_setting)
            return self.mab.predict(mab_setting, arms)
        else:
            # Exploration stage and got the next available bs
            return batch_size

    async def report(self, db: AsyncSession, result: TrainingResult) -> ReportResponse:
        """
        BEGIN
        1. ADD result to the measurement
        2. Check which stage are we in - Exploration or MAB (By checking explorations and see if there is an entry with "Exploring" state)
            2.a IF Exploration: Report to explore_manager
            2.b IF MAB: Report to MAB (observe)
        COMMIT
        """
        cost_ub = np.inf
        job = await DBapi.get_job_with_explorations(db, result.job_id)

        min_cost = (await job.get_min_cost())[0]

        if job.beta_knob > 0:  # Early stop enabled
            cost_ub = job.beta_knob * min_cost

        cost = zeus_cost(
            result.energy,
            result.time,
            job.eta_knob,
            result.max_power,
        )

        converged = (
            job.high_is_better_metric and job.target_metric <= result.metric
        ) or (not job.high_is_better_metric and job.target_metric >= result.metric)

        if (
            cost_ub >= cost
            and result.current_epoch < job.max_epochs
            and converged == False
        ):
            # If it's not converged but below cost upper bound and haven't reached max_epochs, give more chance
            # Training ongoing
            return ReportResponse(
                stop_train=False,
                converged=False,
                message="Stop condition not met, keep training",
            )

        # Two cases below here (training ended)
        # 1. Converged == true
        # 2. reached max_epoch OR excceded upper bound cost (error case)
        message = (
            "Train succeeded"
            if converged
            else f"Train failed to converge within max_epoch({job.max_epochs})"
        )

        DBapi.add_measurement(db, result, converged)

        if self._get_stage(job) == Stage.MAB:
            # We are in MAB stage
            # Since we're learning the reward precision, we need to
            # 1. re-compute the precision of this arm based on the reward history,
            # 2. update the arm's reward precision
            # 3. and `fit` the new MAB instance on all the reward history.
            # Note that `arm_rewards` always has more than one entry (and hence a
            # non-zero variance) because we've been through pruning exploration.

            history = await self._get_history_for_bs(
                db,
                job.job_id,
                result.batch_size,
                job.window_size,
                job.eta_knob,
                job.max_power,
            )
            arm_rewards = np.array(history)
            precision = np.reciprocal(np.var(arm_rewards))
            arm = next(
                (arm for arm in job.arm_states if arm.batch_size == result.batch_size),
                None,
            )
            if arm == None:
                raise ZeusBSOValueError(
                    f"MAB stage but Arm for batch size({result.batch_size}) is not found."
                )
            arm.reward_precision = precision
            self.mab.fit_arm(job.get_mab_setting(), arm, arm_rewards, reset=True)
            await DBapi.update_arm_state(db, arm)

            if self.verbose:
                arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
                self._log(
                    f"{result.job_id} @ {result.batch_size}: "
                    f"arm_rewards = [{arm_rewards_repr}], reward_prec = {precision}"
                )
        else:
            # We are in Exploration Stage
            if self.verbose:
                self._log(
                    f"{result.job_id} in pruning stage, Current BS {result.batch_size} that did {'not ' * (not converged)}converge."
                )

            bs = next(
                (bs for bs in job.batch_sizes if bs.batch_size == result.batch_size),
                None,
            )

            if bs == None:
                raise ZeusBSOValueError(
                    f"Current batch_size({result.batch_size}) is not in the batch_size list({[bs.batch_size for bs in job.batch_sizes]})"
                )

            await PruningExploreManager.report_batch_size_result(
                db, bs, converged, cost, cost < cost_ub
            )

            if cost_ub < cost:
                message = f"""Batch Size({result.batch_size}) exceeded the cost upper bound: current cost({cost}) >
                        beta_knob({job.beta_knob})*min_cost({min_cost})"""

        return ReportResponse(stop_train=True, converged=converged, message=message)

    def _get_job(self, job_id: UUID) -> JobSpec:
        """Return jobSpec based on job_id"""

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[{self.name}] {message}")

    @staticmethod
    async def _get_history_for_bs(
        db: AsyncSession,
        job_id: UUID,
        batch_size: int,
        window_size: int,
        eta_knob: float,
        max_power: float,
    ) -> list[float]:
        """Return the windowed history for the given job's batch size."""
        history = await DBapi.get_measurements_of_bs(
            db, job_id, window_size, batch_size
        )
        rewards = []
        # Collect rewards starting from the most recent ones and backwards.
        for m in history:
            rewards.append(-zeus_cost(m.energy, m.time, eta_knob, max_power))

        # There's no need to return this in time order, but just in case.
        return list(reversed(rewards))

    async def _construct_mab(
        self, db: AsyncSession, job: Job, mab_setting: MabSetting
    ) -> list[GaussianTsArmState]:
        """When exploration is over, this method is called to construct and learn GTS.
        batch_sizes are the ones which can converge"""

        good_bs = []
        for bs in job.batch_sizes:
            for exp in bs.explorations:
                if (
                    exp.trial_number == job.num_pruning_rounds
                    and exp.state == ExplorationState.State.Converged
                ):
                    good_bs.append(bs.batch_size)
                    break

        arms = [
            GaussianTsArmState(
                job_id=job.job_id,
                batch_size=bs,
                param_mean=job.mab_prior_mean,
                param_precision=job.mab_prior_precision,
                reward_precision=0.0,
            )
            for bs in good_bs
        ]

        if self.verbose:
            self._log(f"Construct MAB for {job.job_id} with arms {good_bs}")

        # Fit the arm for each good batch size.
        for i, batch_size in enumerate(good_bs):
            history = await self._get_history_for_bs(
                db,
                job.job_id,
                batch_size,
                job.window_size,
                job.eta_knob,
                job.max_power,
            )
            arm_rewards = np.array(history)
            assert (
                len(arm_rewards) >= job.num_pruning_rounds
            ), f"Number of observations for {batch_size} is {len(arm_rewards)}."
            self._log(f"Number of observations for {batch_size} is {len(arm_rewards)}.")

            # mab.arm_reward_prec[batch_size] = np.reciprocal(np.var(arm_rewards))
            arms[i].reward_precision = np.reciprocal(np.var(arm_rewards))
            self.mab.fit_arm(mab_setting, arms[i], arm_rewards, reset=True)

        DBapi.add_arms(db, arms)
        return arms

    def _get_stage(self, job: Job) -> Stage:
        """Return the stage."""
        for bs in job.batch_sizes:
            if len(bs.explorations) > 0 and any(
                exp.state == ExplorationState.State.Exploring for exp in bs.explorations
            ):
                return Stage.Exploration
        return Stage.MAB


## End of class ZeusBatchSizeOptimizer


def init_global_zeus_batch_size_optimizer(setting) -> ZeusBatchSizeOptimizer:
    """Initialize the global singleton `ZeusBatchSizeOptimizer`."""
    global GLOBAL_ZEUS_SERVER
    GLOBAL_ZEUS_SERVER = ZeusBatchSizeOptimizer(setting)
    return GLOBAL_ZEUS_SERVER


def get_global_zeus_batch_size_optimizer() -> ZeusBatchSizeOptimizer:
    """Fetch the global singleton `ZeusBatchSizeOptimizer`."""
    return GLOBAL_ZEUS_SERVER
