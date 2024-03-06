# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
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

import json
import warnings
from datetime import datetime

import numpy as np
from numpy.random import Generator as np_Generator
from zeus.optimizer.batch_size.common import ZeusBSOValueError
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    UpsertGaussianTsArmState,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    GaussianTsArmStateModel,
)
from zeus.optimizer.batch_size.server.database.schema import GaussianTsArmState


class GaussianTS(object):
    """Thompson Sampling policy for Gaussian bandits.

    For each arm, the reward is modeled as a Gaussian distribution with
    known precision. The conjugate priors are also Gaussian distributions.
    """

    @staticmethod
    def fit_arm(
        bs_base: BatchSizeBase,
        prior_mean: float,
        prior_precision: float,
        rewards: np.ndarray,
    ) -> GaussianTsArmStateModel:
        """Update the parameter distribution for one arm.

        Reference: <https://en.wikipedia.org/wiki/Conjugate_prior>

        Args:
            arm: Arm to fit.
            rewards: Array of rewards observed by pulling that arm.
        """
        if len(rewards) == 0:
            return

        reward_prec = np.reciprocal(np.var(rewards))

        # Reset to priors
        mean = prior_mean
        prec = prior_precision

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
        return GaussianTsArmStateModel(
            job_id=bs_base.job_id,
            batch_size=bs_base.batch_size,
            param_mean=new_mean,
            param_precision=new_prec,
            reward_precision=reward_prec,
            num_observations=len(rewards),
        )

    def predict(
        self,
        mab_setting: MabSetting,
        arms: list[GaussianTsArmState],
    ) -> int:
        """Return the arm with the largest sampled expected reward."""
        rng = np.random.default_rng(int(datetime.now().timestamp()))

        if mab_setting.seed != None:
            if mab_setting.random_generator_state == None:
                raise ZeusBSOValueError(
                    "MAB seed is set but state is unknown. Need to re-register the job"
                )
            state = json.loads(mab_setting.random_generator_state)
            rng.__setstate__(state)

        arm_dict = {arm.batch_size: arm for arm in arms}
        # Exploration-only phase.
        # Order is random considering concurrent bandit scenarios.
        arrms = np.array([arm.batch_size for arm in arms])

        for arm in rng.choice(arrms, len(arrms), replace=False):
            if arm_dict[arm].num_observations < mab_setting.num_exploration:
                if self.verbose:
                    print(f"[{self.name}] Explore arm {arm}.")
                return arm

        # Thomopson Sampling phase.
        expectations = self._predict_expectations(
            arms, rng, mab_setting.prior_precision
        )
        if self.verbose:
            print(f"[{self.name}] Sampled mean rewards:")
            for arm, sample in expectations.items():
                print(
                    f"[{self.name}] Arm {arm:4d}: mu ~ N({arm_dict[arm].param_mean:.2f}, "
                    f"{1/arm_dict[arm].param_precision:.2f}) -> {sample:.2f}"
                )
        return max(expectations, key=expectations.get)  # type: ignore

    def _predict_expectations(
        self,
        arms: list[GaussianTsArmState],
        rng: np_Generator,
        prior_prec: float,
    ) -> dict[int, float]:
        """Sample the expected reward for each arm.

        Assumes that each arm has been explored at least once. Otherwise,
        a value will be sampled from the prior.

        Returns:
            A mapping from every arm to their sampled expected reward.
        """
        expectations = {}
        for arm in arms:
            if arm.param_precision == prior_prec:
                warnings.warn(
                    f"predict_expectations called when arm '{arm.batch_size}' is cold.",
                    stacklevel=1,
                )
            expectations[arm.batch_size] = rng.normal(
                arm.param_mean, np.sqrt(np.reciprocal(arm.param_precision))
            )
        return expectations
