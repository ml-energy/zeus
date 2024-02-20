from __future__ import annotations

from collections import defaultdict
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import NoResultFound

from zeus.optimizer.batch_size.common import (
    JobSpec,
    ReportResponse,
    TrainingResult,
    ZeusBSOJobSpecMismatch,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.database.dbapi import DBapi
from zeus.optimizer.batch_size.server.database.models import Job
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
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
        self.concurrency = True
        # self.seed = seed
        self.verbose = verbose

        # One MAB for each job.
        self.mabs: dict[UUID, GaussianTS] = {}

        # Observation history (batch size, reward) for each job.
        self.history: dict[UUID, defaultdict[int, list[float]]] = {}

        # Min_cost observed so far
        self.min_costs: dict[UUID, float] = {}

        # One PruningExplorationManager for each job.
        self.exp_manager: dict[UUID, PruningExploreManager] = {}

        # Job UUID -> Job spec (will be replaced to hit DB prob)
        self.jobs: dict[UUID, JobSpec] = {}

        # # Track the batch size range for each job.
        # self.batch_sizes: dict[JobSpec, list[int]] = {}

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        return "Pruning GaussianTS BSO"

    async def register_job(self, job: JobSpec, db: AsyncSession) -> bool:
        """Register a user-submitted job. Return number of newly created job. Return the number of job that is registered"""
        registered_job = await DBapi.create_job(db, job)

        if registered_job is None:
            # Job already exists
            if self.verbose:
                self._log(f"Job({job.job_id}) already exists")
            registered_job = await DBapi.get_job(db, job.job_id)

            if registered_job == None:
                raise NoResultFound("Can't create job but job doesn't exist")
            elif not registered_job.equal_to(job):
                raise ZeusBSOJobSpecMismatch(
                    "JobSpec doesn't match with existing jobSpec. Use a new job_id for different configuration"
                )
            return False

        # Job just created

        self.jobs[job.job_id] = job
        self.min_costs[job.job_id] = np.inf  # initialize it to inf.

        # Set internal states.
        self.exp_manager[job.job_id] = PruningExploreManager(
            job.batch_sizes,
            job.default_batch_size,
            job.num_pruning_rounds,
        )
        self.history[job.job_id] = []
        if self.verbose:
            self._log(f"Registered {job.job_id}")
        return True

    def predict(self, job_id: UUID) -> None:
        """return a batch size to use. Probably get the MAB from DB? then do some computation
        Return the batch size to use for the job."""
        # Try to see if the exploration manager has something.
        try:
            batch_size = self.exp_manager[job_id].next_batch_size()
            if self.verbose:
                self._log(
                    f"{job_id} in pruning stage -> \033[31mBS = {batch_size}\033[0m"
                )
        except StopIteration as exp:
            # Pruning stage is over. We construct MAB after pruning
            if job_id not in self.mabs:
                self._construct_mab(job_id, exp.value)
            batch_size = self.mabs[job_id].predict()
            if self.verbose:
                self._log(
                    f"{job_id} in Thompson Sampling stage -> \033[31mBS = {batch_size}\033[0m"
                )

        return batch_size

    def report(self, result: TrainingResult) -> ReportResponse:
        """Give feedback to MAB
        Learn from the cost of using the given batch size for the job."""
        # Add observation to history.
        cost_ub = np.inf

        if self.jobs[result.job_id].beta_knob > 0:  # Early stop enabled
            cost_ub = self.jobs[result.job_id].beta_knob * self.min_costs[result.job_id]

        cost = zeus_cost(
            result.energy,
            result.time,
            self.jobs[result.job_id].eta_knob,
            result.max_power,
        )

        converged = (
            self.jobs[result.job_id].high_is_better_metric
            and self.jobs[result.job_id].target_metric <= result.metric
        ) or (
            not self.jobs[result.job_id].high_is_better_metric
            and self.jobs[result.job_id].target_metric >= result.metric
        )

        if (
            cost_ub >= cost
            and result.current_epoch < self.jobs[result.job_id].max_epochs
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
            else f"Train failed to converge within max_epoch({self.jobs[result.job_id].max_epochs})"
        )

        self.history[result.job_id].append((result.batch_size, -cost))

        # update min cost
        self.min_costs[result.job_id] = min(self.min_costs[result.job_id], cost)

        # We're in Thompson Sampling stage.
        if result.job_id in self.mabs:
            # Since we're learning the reward precision, we need to
            # 1. re-compute the precision of this arm based on the reward history,
            # 2. update the arm's reward precision
            # 3. and `fit` the new MAB instance on all the reward history.
            # Note that `arm_rewards` always has more than one entry (and hence a
            # non-zero variance) because we've been through pruning exploration.
            arm_rewards = np.array(
                self._get_history_for_bs(result.job_id, result.batch_size)
            )
            precision = np.reciprocal(np.var(arm_rewards))
            mab = self.mabs[result.job_id]
            mab.arm_reward_prec[result.batch_size] = precision
            mab.fit_arm(result.batch_size, arm_rewards, reset=True)
            if self.verbose:
                arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
                self._log(
                    f"{result.job_id} @ {result.batch_size}: "
                    f"arm_rewards = [{arm_rewards_repr}], reward_prec = {precision}"
                )

        # We're in pruning stage.
        else:
            # Log before we potentially error out.
            if self.verbose:
                self._log(
                    f"{result.job_id} in pruning stage, expecting BS {self.exp_manager[result.job_id].expecting}."
                    f" Current BS {result.batch_size} that did {'not ' * (not converged)}converge."
                )

            # If we don't support concurrency, we can just pass the results to the
            # exploration manager, and the manager will err if the order of batch sizes
            # is screwed up.
            if not self.concurrency:
                self.exp_manager[result.job_id].report_batch_size_result(
                    result.batch_size,
                    cost,
                    converged,
                )
                return ReportResponse(
                    stop_train=True, converged=converged, message=message
                )

            # If we are supporting concurrency, there's a subtle issue.
            # Pruning exploration demands a specific order of trying out a batch size
            # and receiving the results (cost and whether reached). This breaks in the
            # following situation, for example:
            # 1. Job with BS 32 that is part of pruning exploration starts.
            # 2. Concurrent job comes in, and we launch it with the best known BS 64.
            # 3. Job with BS 64 finishes first, and calls bso.observe with BS 64.
            # This breaks the observation order assumption of PruningExplorationManager.
            # Thus we check whether the current batch size is the one expected by
            # PruningExplorationManager, and then only if so, call bso.observe.
            # Otherwise, we silently insert the cost observation into the bso's history
            # (first line of this method) and don't touch the PruningExplorationManager.
            if self.exp_manager[result.job_id].expecting == result.batch_size:
                self.exp_manager[result.job_id].report_batch_size_result(
                    result.batch_size, cost, converged
                )

            # Early stopping
            # Result can be converged / max epoch reached / in the middle of epoch
            # If the cost is above the upper bound, we should stop no matter what, report to explore manger that we failed
            if cost_ub < cost:
                message = f"""Batch Size({result.batch_size}) exceeded the cost upper bound: current cost({cost}) >
                          beta_knob({self.jobs[result.job_id].beta_knob})*min_cost({self.min_costs[result.job_id]})"""

        return ReportResponse(stop_train=True, converged=converged, message=message)

    def _get_job(self, job_id: UUID) -> JobSpec:
        """Return jobSpec based on job_id"""

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[{self.name}] {message}")

    def _get_history_for_bs(self, job_id: UUID, batch_size: int) -> list[float]:
        """Return the windowed history for the given job's batch size."""
        history = self.history[job_id]
        rewards = []
        # Collect rewards starting from the most recent ones and backwards.
        for bs, reward in reversed(history):
            if bs == batch_size:
                rewards.append(reward)
                if len(rewards) == self.jobs[job_id].window_size:
                    break
        # There's no need to return this in time order, but just in case.
        return list(reversed(rewards))

    def _construct_mab(self, job_id: UUID, batch_sizes: list[int]) -> None:
        """When exploration is over, this method is called to construct and learn GTS.
        batch_sizes are the ones which can converge"""
        # Sanity check.
        if not batch_sizes:
            raise ZeusBSOValueError(
                "Empty batch size set when constructing MAB. "
                "Probably all batch sizes have been pruned."
            )

        if self.verbose:
            self._log(f"Construct MAB for {job_id} with arms {batch_sizes}")

        mab = GaussianTS(
            arms=batch_sizes,  # The MAB only has "good" arms.
            reward_precision=0.0,
            prior_mean=self.jobs[job_id].mab_setting.prior_mean,
            prior_precision=self.jobs[job_id].mab_setting.prior_precision,
            num_exploration=self.jobs[job_id].mab_setting.num_exploration,
            seed=self.jobs[job_id].mab_setting.seed,
            verbose=self.verbose,
        )
        # Fit the arm for each good batch size.
        for batch_size in self.exp_manager[job_id].batch_sizes:
            arm_rewards = np.array(self._get_history_for_bs(job_id, batch_size))
            assert (
                len(arm_rewards) >= 2
            ), f"Number of observations for {batch_size} is {len(arm_rewards)}."
            mab.arm_reward_prec[batch_size] = np.reciprocal(np.var(arm_rewards))
            mab.fit_arm(batch_size, arm_rewards, reset=True)
        # Save the MAB.
        self.mabs[job_id] = mab


## End of class ZeusBatchSizeOptimizer


def init_global_zeus_batch_size_optimizer(setting) -> ZeusBatchSizeOptimizer:
    """Initialize the global singleton `ZeusBatchSizeOptimizer`."""
    global GLOBAL_ZEUS_SERVER
    GLOBAL_ZEUS_SERVER = ZeusBatchSizeOptimizer(setting)
    return GLOBAL_ZEUS_SERVER


def get_global_zeus_batch_size_optimizer() -> ZeusBatchSizeOptimizer:
    """Fetch the global singleton `ZeusBatchSizeOptimizer`."""
    return GLOBAL_ZEUS_SERVER
