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

"""Multi-Armed Bandit implementations."""

from __future__ import annotations

import warnings

import numpy as np


class GaussianTS:
    """Thompson Sampling policy for Gaussian bandits.

    For each arm, the reward is modeled as a Gaussian distribution with
    known precision. The conjugate priors are also Gaussian distributions.
    """

    def __init__(
        self,
        arms: list[int],
        reward_precision: list[float] | float,
        prior_mean: float = 0.0,
        prior_precision: float = 0.0,
        num_exploration: int = 1,
        seed: int = 123456,
        verbose: bool = True,
    ) -> None:
        """Initialze the object.

        Args:
            arms: Bandit arm values to use.
            reward_precision: Precision (inverse variance) of the reward distribution.
                Pass in a list of `float`s to set the reward precision differently for
                each arm.
            prior_mean: Mean of the belief prior distribution.
            prior_precision: Precision of the belief prior distribution.
            num_exploration: How many static explorations to run when no observations
                are available.
            seed: The random seed to use.
            verbose: Whether to print out what's going on.
        """
        self.name = "GaussianTS"

        self.arms = arms
        self.prior_mean = prior_mean
        self.prior_prec = prior_precision
        self.num_exploration = num_exploration
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        # Set the precision of the reward distribution of each arm.
        if isinstance(reward_precision, list):
            self.arm_reward_prec = dict(zip(arms, reward_precision))
        else:
            self.arm_reward_prec = {arm: reward_precision for arm in arms}

        # Initialze the parameter distribution with the prior parameters.
        self.arm_param_mean = dict.fromkeys(arms, prior_mean)
        self.arm_param_prec = dict.fromkeys(arms, prior_precision)

        # Track how many times an arm reward has been observed.
        self.arm_num_observations = dict.fromkeys(arms, 0)

    def fit(
        self,
        decisions: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        reset: bool,
    ) -> None:
        """Fit the bandit on the given list of observations.

        Args:
            decisions: A list of arms chosen.
            rewards: A list of rewards that resulted from choosing the arms in `decisions`.
        """
        decisions_arr = np.array(decisions)
        rewards_arr = np.array(rewards)

        # Fit all arms.
        for arm in self.arms:
            self.fit_arm(arm, rewards_arr[decisions_arr == arm], reset)

    def fit_arm(self, arm: int, rewards: np.ndarray, reset: bool) -> None:
        """Update the parameter distribution for one arm.

        Reference: <https://en.wikipedia.org/wiki/Conjugate_prior>

        Args:
            arm: Arm to fit.
            rewards: Array of rewards observed by pulling that arm.
            reset: Whether to reset the parameters of the arm before fitting.
        """
        if reset:
            self.arm_param_mean[arm] = self.prior_mean
            self.arm_param_prec[arm] = self.prior_prec
            self.arm_num_observations[arm] = 0

        if len(rewards) == 0:
            return

        # Read previous state.
        reward_prec = self.arm_reward_prec[arm]
        mean = self.arm_param_mean[arm]
        prec = self.arm_param_prec[arm]

        # Compute the parameters of the posterior distribution.
        # The reward distribution's precision is given as infinite only when we
        # have exactly one observation for the arm, s.t. sampling yields that
        # exact observation.
        if reward_prec == np.inf:
            new_prec = np.inf
            new_mean = rewards.mean()
        else:
            new_prec = prec + len(rewards) * reward_prec
            new_mean = (prec * mean + reward_prec * rewards.sum()) / new_prec

        # Update state.
        self.arm_param_mean[arm] = new_mean
        self.arm_param_prec[arm] = new_prec
        self.arm_num_observations[arm] += len(rewards)

    def predict(self) -> int:
        """Return the arm with the largest sampled expected reward."""
        # Exploration-only phase.
        # Order is random considering concurrent bandit scenarios.
        arrms = np.array(self.arms)
        for arm in self.rng.choice(arrms, len(arrms), replace=False):
            if self.arm_num_observations[arm] < self.num_exploration:
                if self.verbose:
                    print(f"[{self.name}] Explore arm {arm}.")
                return arm

        # Thomopson Sampling phase.
        expectations = self.predict_expectations()
        if self.verbose:
            print(f"[{self.name}] Sampled mean rewards:")
            for arm, sample in expectations.items():
                print(
                    f"[{self.name}] Arm {arm:4d}: mu ~ N({self.arm_param_mean[arm]:.2f}, "
                    f"{1/self.arm_param_prec[arm]:.2f}) -> {sample:.2f}"
                )
        return max(expectations, key=expectations.get)  # type: ignore

    def predict_expectations(self) -> dict[int, float]:
        """Sample the expected reward for each arm.

        Assumes that each arm has been explored at least once. Otherwise,
        a value will be sampled from the prior.

        Returns:
            A mapping from every arm to their sampled expected reward.
        """
        expectations = {}
        for arm in self.arms:
            mean = self.arm_param_mean[arm]
            prec = self.arm_param_prec[arm]
            if prec == self.prior_prec:
                warnings.warn(f"predict_expectations called when arm '{arm}' is cold.")
            expectations[arm] = self.rng.normal(mean, np.sqrt(np.reciprocal(prec)))
        return expectations
