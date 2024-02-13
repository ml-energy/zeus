from __future__ import annotations

from collections import defaultdict
from uuid import UUID
from fastapi import HTTPException

import numpy as np
from zeus.optimizer import batch_size

from zeus.optimizer.batch_size.server.models import (
    JobSpec,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.util.metric import zeus_cost

# from dbapis import DBAPI

GLOBAL_ZEUS_SERVER: ZeusBatchSizeOptimizer | None = None

"""
TODO: ADD EARLY STOPPING SOMEHOW?. If cost is too much, stop training 
"""


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

    def register_job(self, job: JobSpec) -> int:
        """Register a user-submitted job. Return number of newly created job"""
        if job.job_id in self.jobs:
            if self.verbose:
                self._log(f"Job({job.job_id}) already exists")
            return 0

        # TODO: Append the job to Jobs table by POST to DBServer
        # await DBAPI.insert_job(job)

        if job.default_batch_size not in job.batch_sizes:
            raise HTTPException(
                status_code=422,
                detail=f"Default BS({job.default_batch_size}) not in batch_sizes({job.batch_sizes}).",
            )

        self.jobs[job.job_id] = job

        # Set internal states.
        self.exp_manager[job.job_id] = PruningExploreManager(
            job.batch_sizes,
            job.default_batch_size,
            job.num_pruning_rounds,
        )
        self.history[job.job_id] = []
        if self.verbose:
            self._log(f"Registered {job.job_id}")
        return 1

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

    def report(self, result: TrainingResult) -> None:
        """Give feedback to MAB
        Learn from the cost of using the given batch size for the job."""
        # Add observation to history.
        cost = zeus_cost(
            result.energy,
            result.time,
            self.jobs[result.job_id].eta_knob,
            result.max_power,
        )

        self.history[result.job_id].append((result.batch_size, -cost))

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
            assert result.converged is not None
            # Log before we potentially error out.
            if self.verbose:
                self._log(
                    f"{result.job_id} in pruning stage, expecting BS {self.exp_manager[result.job_id].expecting}."
                    f" Current BS {result.batch_size} that did {'not ' * (not result.converged)}converge."
                )

            # If we don't support concurrency, we can just pass the results to the
            # exploration manager, and the manager will err if the order of batch sizes
            # is screwed up.
            if not self.concurrency:
                self.exp_manager[result.job_id].report_batch_size_result(
                    result.batch_size,
                    cost,
                    result.converged,  # TODO: Double check -cost vs cost.
                )
                return

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
                    result.batch_size, cost, result.converged
                )

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
            raise ValueError(
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


def init_global_zeus_server(setting) -> ZeusBatchSizeOptimizer:
    """Initialize the global singleton `ZeusServer`."""
    global GLOBAL_ZEUS_SERVER
    GLOBAL_ZEUS_SERVER = ZeusBatchSizeOptimizer(setting)
    return GLOBAL_ZEUS_SERVER


def get_global_zeus_server() -> ZeusBatchSizeOptimizer:
    """Fetch the global singleton `ZeusServer`."""
    return GLOBAL_ZEUS_SERVER
