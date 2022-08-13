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

"""
Implementations for various optimization policies.

[`JITPowerLimitOptimizer`][zeus.policy.optimizer.JITPowerLimitOptimizer] and
[`PruningGTSBatchSizeOptimizer`][zeus.policy.optimizer.PruningGTSBatchSizeOptimizer]
are the implementations used in Zeus's publication.
"""

from __future__ import annotations

import unittest
from collections import defaultdict
from typing import Generator

import numpy as np

from zeus.job import Job
from zeus.policy.interface import BatchSizeOptimizer, PowerLimitOptimizer
from zeus.policy.mab import GaussianTS


class GTSBatchSizeOptimizer(BatchSizeOptimizer):
    """One Gaussian Thompson Sampling MAB for each job."""

    def __init__(
        self,
        learn_reward_precision: bool,
        reward_precision: float = 0.0,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        num_exploration: int = 1,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the optimizer.

        Refer to the constructor of [`GaussianTS`][zeus.policy.mab.GaussianTS]
        for descriptions of other arguments.

        Args:
            learn_reward_precision: Whether to learn the reward precision of
                each arm as we accumulate observations.
        """
        self.learn_reward_precision = learn_reward_precision
        self.reward_precision = reward_precision
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.num_exploration = num_exploration
        self.seed = seed
        self.verbose = verbose

        # One MAB for each job.
        self.mabs: dict[Job, GaussianTS] = {}

        # Track the batch size range for each job.
        self.batch_sizes: dict[Job, list[int]] = {}

        # Observation history (batch size, reward) for each job.
        self.history: dict[Job, defaultdict[int, list[float]]] = {}

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        return "GaussianTS BSO"

    def register_job(self, job: Job, batch_sizes: list[int]) -> None:
        """Instantiate a new GaussianTS MAB for the new job."""
        # We do not want to reset the state related to this job if
        # anything already exists.
        if job in self.mabs:
            return
        self.mabs[job] = GaussianTS(
            arms=batch_sizes,
            reward_precision=self.reward_precision,
            prior_mean=self.prior_mean,
            prior_precision=self.prior_precision,
            num_exploration=self.num_exploration,
            seed=self.seed,
            verbose=self.verbose,
        )
        self.batch_sizes[job] = batch_sizes
        self.history[job] = defaultdict(list)
        if self.verbose:
            self._log(f"Registered {job}")

    def predict(self, job: Job) -> int:
        """Return the batch size to use for the job."""
        if self.verbose:
            self._log(f"Prediction for {job}")
        pred = self.mabs[job].predict()
        if self.verbose:
            self._log(f"{job} -> \033[31mBS = {pred}\033[0m")
        return pred

    def observe(
        self, job: Job, batch_size: int, cost: float, converged: bool | None = None
    ) -> None:
        """Learn from the cost of using the given batch size for the job."""
        if batch_size not in self.batch_sizes[job]:
            raise ValueError(f"Unknown batch size '{batch_size}' for {job}.")

        # No normalization needed since we learn a separate bandit for each job.
        reward = -cost

        # Add observation to history.
        self.history[job][batch_size].append(reward)

        # When we're not learning the reward precision, everyting is
        # simple. We can just call `partial_fit` on the job's MAB instance.
        if not self.learn_reward_precision:
            self.mabs[job].fit([batch_size], [reward], reset=False)
            if self.verbose:
                self._log(f"{job} @ {batch_size}: reward = {reward:.2f}")

        # When we're learning the reward precision, we need to
        # 1. re-compute the precision this arm based on the history,
        # 2. update the arm's reward precision
        # 3. and `fit` the new MAB instance on all past data.
        else:
            arm_rewards = np.array(self.history[job][batch_size])
            variance = np.var(arm_rewards)
            # When there is only one observation for the arm, the variance is zero.
            # NOTE: We might still want to have a pre-determined reward precision here
            #       because sampling from an infinite precision Gaussian distribution
            #       always returns the mean (the observation), and it will hamper
            #       exploration in the early stage.
            if variance == 0.0:
                precision = np.inf
            else:
                precision = np.reciprocal(variance)
            mab = self.mabs[job]
            mab.arm_reward_prec[batch_size] = precision
            mab.fit_arm(batch_size, arm_rewards, reset=True)
            self.mabs[job] = mab
            if self.verbose:
                arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
                self._log(
                    f"{job} @ {batch_size}: "
                    f"arm_rewards = [{arm_rewards_repr}], reward_prec = {precision}"
                )


class PruningExploreManager:
    """Helper class that generates batch sizes to explore and prune."""

    def __init__(
        self,
        batch_sizes: list[int],
        default: int,
        num_pruning_rounds: int = 2,
    ) -> None:
        """Initialze the object.

        Args:
            batch_sizes: The initial set of batch sizes to prune from.
            default: The default batch size (b0) to begin exploration from.
            num_pruning_rounds: How many rounds to run pruning.
        """
        # Sanity checks.
        if default not in batch_sizes:
            raise ValueError(f"Default batch size {default} not in {batch_sizes}.")

        # Save arguments.
        self.batch_sizes = batch_sizes
        self.default = default
        self.num_pruning_rounds = num_pruning_rounds

        # State
        self.expecting = default

        # Generator that returns batch sizes.
        self.gen = self._exploration_engine()

    def _exploration_engine(
        self,
    ) -> Generator[int | None, tuple[int, float, bool], list[int]]:
        """Drive pruning exploration.

        Yields the batch size to be explored.
        The caller should `send` a tuple of (explored batch size, cost, whether reached).
        As a safety measure, the explored batch size must match the most recently yielded
        batch size, and otherwise a `RuntimeError` is raised.
        Finally, when exploration is over, returns a sorted list of batch sizes that
        survived pruning.
        """
        for _ in range(self.num_pruning_rounds):
            # A list of batch sizes that reached the target metric.
            good: list[int] = []

            # We first explore downwards form the default batch size, and then go upwards.
            idx = self.batch_sizes.index(self.default)
            down = sorted(self.batch_sizes[: idx + 1], reverse=True)
            up = sorted(self.batch_sizes[idx + 1 :])

            # We track the best cost because the default batch size is updated to the batch
            # size that performed the best.
            best_cost = np.inf

            for bs_list in [down, up]:
                for bs in bs_list:
                    # We tell the outside world to explore `bs`, and we expect the outside
                    # world to give us back the cost of that `bs`.
                    self.expecting = bs
                    batch_size, cost, reached = yield bs
                    if self.expecting != batch_size:
                        raise RuntimeError(
                            f"PruningExplorationManager: {self.expecting=}, {batch_size=}"
                        )
                    self.expecting = 0

                    # An empty `yield` to not proceed to the next batch size when the caller
                    # `send`s in the results.
                    yield

                    # Only batch sizes that reached the target mteric are good.
                    if reached:
                        if best_cost > cost:
                            best_cost = cost
                            self.default = bs
                        good.append(bs)
                    # If the batch size did not reach the target metric, `break`ing here will
                    # allow us to move on to either the next direction of exploration (upwards)
                    # or end this round of pruning exploration.
                    else:
                        break

            self.expecting = 0
            self.batch_sizes = sorted(good)

        return sorted(self.batch_sizes)

    def next_batch_size(self) -> int:
        """Return the next batch size to explore.

        Raises `StopIteration` when pruning exploration phase is over.
        The exception instance contains the final set of batch sizes to consider.
        Access it through `exception.value`.
        """
        batch_size = next(self.gen)
        assert batch_size is not None, "Call order may have been wrong."
        return batch_size

    def report_batch_size_result(
        self, batch_size: int, cost: float, reached: bool
    ) -> None:
        """Report whether the previous batch size reached the target metric.

        Args:
            batch_size: The batch size which this cost observation is from.
            cost: The energy-time cost of running the job with this batch size.
            reached: Whether the job reached the target metric.
        """
        none = self.gen.send((batch_size, cost, reached))
        assert none is None, "Call order may have been wrong."


class TestPruningExploreManager(unittest.TestCase):
    """Unit test class for pruning exploration."""

    batch_sizes: list[int] = [8, 16, 32, 64, 128, 256]

    def run_exploration(
        self,
        manager: PruningExploreManager,
        exploration: list[tuple[int, float, bool]],
        result: list[int],
    ) -> None:
        """Drive the pruning explore manager and check results."""
        for bs, cost, reached in exploration:
            self.assertEqual(manager.next_batch_size(), bs)
            manager.report_batch_size_result(bs, cost, reached)
        with self.assertRaises(StopIteration) as raised:
            manager.next_batch_size()
        self.assertEqual(raised.exception.value, result)

    def test_normal(self):
        """Test a typical case."""
        manager = PruningExploreManager(self.batch_sizes, 128)
        exploration = [
            (128, 10.0, True),
            (64, 9.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 21.0, False),
            (256, 15.0, True),
            (32, 8.0, True),
            (16, 12.0, False),
            (64, 9.0, True),
            (128, 10.0, True),
            (256, 17.0, False),
        ]
        result = [32, 64, 128]
        self.run_exploration(manager, exploration, result)

    def test_default_is_largest(self):
        """Test the case when the default batch size is the largest one."""
        manager = PruningExploreManager(self.batch_sizes, 256)
        exploration = [
            (256, 7.0, True),
            (128, 8.0, True),
            (64, 9.0, True),
            (32, 13.0, True),
            (16, 22.0, False),
            (256, 8.0, True),
            (128, 8.5, True),
            (64, 9.0, True),
            (32, 12.0, True),
        ]
        result = [32, 64, 128, 256]
        self.run_exploration(manager, exploration, result)

    def test_default_is_smallest(self):
        """Test the case when the default batch size is the smallest one."""
        manager = PruningExploreManager(self.batch_sizes, 8)
        exploration = [
            (8, 10.0, True),
            (16, 17.0, True),
            (32, 20.0, True),
            (64, 25.0, False),
            (8, 10.0, True),
            (16, 21.0, False),
        ]
        result = [8]
        self.run_exploration(manager, exploration, result)

    def test_all_converge(self):
        """Test the case when every batch size converges."""
        manager = PruningExploreManager(self.batch_sizes, 64)
        exploration = [
            (64, 10.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 15.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
            (32, 7.0, True),
            (16, 10.0, True),
            (8, 15.0, True),
            (64, 10.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
        ]
        result = self.batch_sizes
        self.run_exploration(manager, exploration, result)

    def test_every_bs_is_bs(self):
        """Test the case when every batch size other than the default fail to converge."""
        manager = PruningExploreManager(self.batch_sizes, 64)
        exploration = [
            (64, 10.0, True),
            (32, 22.0, False),
            (128, 25.0, False),
            (64, 9.0, True),
        ]
        result = [64]
        self.run_exploration(manager, exploration, result)


class PruningGTSBatchSizeOptimizer(BatchSizeOptimizer):
    """One Gaussian Thompson Sampling MAB for each job with double pruning exploration."""

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        window_size: int = 0,
        concurrency: bool = False,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the optimizer.

        Refer to the constructor of [`GaussianTS`][zeus.policy.mab.GaussianTS]
        for descriptions of other arguments.

        Args:
            window_size: Size of the window for the MAB (for drift handling).
            concurrency: Whether to support concurrent job submissions.
        """
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.window_size = window_size
        self.concurrency = concurrency
        self.seed = seed
        self.verbose = verbose

        # One MAB for each job.
        self.mabs: dict[Job, GaussianTS] = {}

        # One PruningExplorationManager for each job.
        self.exp_manager: dict[Job, PruningExploreManager] = {}

        # Observation history (batch size, reward) for each job.
        self.history: dict[Job, list[tuple[int, float]]] = {}

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        return "Pruning GaussianTS BSO"

    def register_job(self, job: Job, batch_sizes: list[int]) -> None:
        """Register the job."""
        # Sanity checks.
        if job.default_bs is None:
            raise ValueError(f"Default BS not specified for {job}.")
        if not batch_sizes:
            raise ValueError(f"Batch size list for {job} is empty.")

        # Set internal states.
        self.exp_manager[job] = PruningExploreManager(
            sorted(batch_sizes), job.default_bs
        )
        self.history[job] = []
        if self.verbose:
            self._log(f"Registered {job}")

    def predict(self, job: Job) -> int:
        """Return the batch size to use for the job."""
        # Try to see if the exploration manager has something.
        try:
            batch_size = self.exp_manager[job].next_batch_size()
            if self.verbose:
                self._log(f"{job} in pruning stage -> \033[31mBS = {batch_size}\033[0m")
        except StopIteration as exp:
            # Pruning stage is over.
            if job not in self.mabs:
                self._construct_mab(job, exp.value)
            batch_size = self.mabs[job].predict()
            if self.verbose:
                self._log(
                    f"{job} in Thompson Sampling stage -> \033[31mBS = {batch_size}\033[0m"
                )

        return batch_size

    def observe(
        self, job: Job, batch_size: int, cost: float, converged: bool | None = None
    ) -> None:
        """Learn from the cost of using the given batch size for the job."""
        # Add observation to history.
        self.history[job].append((batch_size, -cost))

        # We're in Thompson Sampling stage.
        if job in self.mabs:
            # Since we're learning the reward precision, we need to
            # 1. re-compute the precision of this arm based on the reward history,
            # 2. update the arm's reward precision
            # 3. and `fit` the new MAB instance on all the reward history.
            # Note that `arm_rewards` always has more than one entry (and hence a
            # non-zero variance) because we've been through pruning exploration.
            arm_rewards = np.array(self._get_history_for_bs(job, batch_size))
            precision = np.reciprocal(np.var(arm_rewards))
            mab = self.mabs[job]
            mab.arm_reward_prec[batch_size] = precision
            mab.fit_arm(batch_size, arm_rewards, reset=True)
            if self.verbose:
                arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
                self._log(
                    f"{job} @ {batch_size}: "
                    f"arm_rewards = [{arm_rewards_repr}], reward_prec = {precision}"
                )

        # We're in pruning stage.
        else:
            assert converged is not None
            # Log before we potentially error out.
            if self.verbose:
                self._log(
                    f"{job} in pruning stage, expecting BS {self.exp_manager[job].expecting}."
                    f" Current BS {batch_size} that did {'not ' * converged}converge."
                )

            # If we don't support concurrency, we can just pass the results to the
            # exploration manager, and the manager will err if the order of batch sizes
            # is screwed up.
            if not self.concurrency:
                self.exp_manager[job].report_batch_size_result(
                    batch_size, cost, converged
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
            if self.exp_manager[job].expecting == batch_size:
                self.exp_manager[job].report_batch_size_result(
                    batch_size, cost, converged
                )

    def _get_history_for_bs(self, job: Job, batch_size: int) -> list[float]:
        """Return the windowed history for the given job's batch size."""
        history = self.history[job]
        rewards = []
        # Collect rewards starting from the most recent ones and backwards.
        for bs, reward in reversed(history):
            if bs == batch_size:
                rewards.append(reward)
                if len(rewards) == self.window_size:
                    break
        # There's no need to return this in time order, but just in case.
        return list(reversed(rewards))

    def _construct_mab(self, job: Job, batch_sizes: list[int]) -> None:
        """When exploration is over, this method is called to construct and learn GTS."""
        # Sanity check.
        if not batch_sizes:
            raise ValueError(
                "Empty batch size set when constructing MAB. "
                "Probably all batch sizes have been pruned."
            )

        if self.verbose:
            self._log(f"Construct MAB for {job} with arms {batch_sizes}")

        mab = GaussianTS(
            arms=batch_sizes,  # The MAB only has "good" arms.
            reward_precision=0.0,
            prior_mean=self.prior_mean,
            prior_precision=self.prior_precision,
            num_exploration=2,
            seed=self.seed,
            verbose=self.verbose,
        )
        # Fit the arm for each good batch size.
        for batch_size in self.exp_manager[job].batch_sizes:
            arm_rewards = np.array(self._get_history_for_bs(job, batch_size))
            assert (
                len(arm_rewards) >= 2
            ), f"Number of observations for {batch_size} is {len(arm_rewards)}."
            mab.arm_reward_prec[batch_size] = np.reciprocal(np.var(arm_rewards))
            mab.fit_arm(batch_size, arm_rewards, reset=True)
        # Save the MAB.
        self.mabs[job] = mab


class JITPowerLimitOptimizer(PowerLimitOptimizer):
    """Returns the best power limit to use for the job & batch size."""

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the object."""
        self.verbose = verbose

        self.best_pl: defaultdict[Job, dict[int, int]] = defaultdict(dict)
        self.best_cost: defaultdict[Job, dict[int, float]] = defaultdict(dict)
        self.observe_count: defaultdict[Job, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @property
    def name(self) -> str:
        """Name of the power limit optimizer."""
        return "JITPSO"

    def predict(self, job: Job, batch_size: int) -> int | None:
        """Return the best power limit for the job, or None if unknown."""
        pred = self.best_pl[job].get(batch_size)
        if self.verbose:
            self._log(
                f"{job} @ {batch_size} -> \033[31mPL = "
                f"{'needs profiling' if pred is None else str(pred) + 'W'}\033[0m"
            )
        return pred

    def observe(self, job: Job, batch_size: int, power_limit: int, cost: float) -> None:
        """Learn from the cost of using the given knobs for the job."""
        self.observe_count[job][batch_size] += 1
        prev_best_cost = self.best_cost[job].get(batch_size)
        if prev_best_cost is None or prev_best_cost > cost:
            self.best_pl[job][batch_size] = power_limit
            self.best_cost[job][batch_size] = cost


if __name__ == "__main__":
    unittest.main()
