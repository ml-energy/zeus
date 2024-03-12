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

from uuid import UUID
import warnings

import numpy as np
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationStateModel,
    GaussianTsArmStateModel,
    MeasurementOfBs,
)
from zeus.optimizer.batch_size.server.database.schema import GaussianTsArmState, State
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError
from zeus.optimizer.batch_size.server.job.commands import UpdateJobStage
from zeus.optimizer.batch_size.server.job.models import JobState, Stage
from zeus.optimizer.batch_size.server.services.commands import (
    CreateArms,
    GetNormal,
    GetRandomChoices,
)
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.util.logging import get_logger
from zeus.util.metric import zeus_cost

logger = get_logger(__name__)

class GaussianTS:
    """Thompson Sampling policy for Gaussian bandits.

    For each arm, the reward is modeled as a Gaussian distribution with
    known precision. The conjugate priors are also Gaussian distributions.
    """

    def __init__(self, service: ZeusService):
        self.service = service
        self.name = "GaussianTS"

    def fit_arm(
        self,
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
        job_id: UUID,
        prior_precision: float,
        num_exploration: int,
        arms: list[GaussianTsArmStateModel],
    ) -> int:
        """Return the arm with the largest sampled expected reward."""
        arm_dict = {arm.batch_size: arm for arm in arms}

        # Exploration-only phase.
        # Order is random considering concurrent bandit scenarios.
        choices = self.service.get_random_choices(
            GetRandomChoices(job_id=job_id, choices=[arm.batch_size for arm in arms])
        )

        for arm in choices:
            if arm_dict[arm].num_observations < num_exploration:
                # if self.verbose:
                print(f"[{self.name}] Explore arm {arm}.")
                return arm

        """Thomopson Sampling phase.

        Sample the expected reward for each arm.
        Assumes that each arm has been explored at least once. Otherwise,
        a value will be sampled from the prior.

        """
        expectations = {}  # A mapping from every arm to their sampled expected reward.
        for arm in arms:
            if arm.param_precision == prior_precision:
                warnings.warn(
                    f"predict_expectations called when arm '{arm.batch_size}' is cold.",
                    stacklevel=1,
                )
            expectations[arm.batch_size] = self.service.get_normal(
                GetNormal(
                    job_id=job_id,
                    loc=arm.param_mean,
                    scale=np.sqrt(np.reciprocal(arm.param_precision)),
                )
            )

        # if self.verbose:
        print(f"[{self.name}] Sampled mean rewards:")
        for arm, sample in expectations.items():
            print(
                f"[{self.name}] Arm {arm:4d}: mu ~ N({arm_dict[arm].param_mean:.2f}, "
                f"{1/arm_dict[arm].param_precision:.2f}) -> {sample:.2f}"
            )

        bs = max(expectations, key=expectations.get)
        self._log(
            f"{job_id} in Thompson Sampling stage -> \033[31mBS = {bs}\033[0m"
        )
        return bs 

    async def construct_mab(
        self, job: JobState, arms: CreateArms
    ) -> list[GaussianTsArmStateModel]:
        """
        1. From Explorations,
        2. Get converged bs (good_bs)
        3. get measurement of each of good_bs
        4. create arms
        5. update stage to MAB!
        """
        arms.validate_exp_rounds(job.num_pruning_rounds)

        good_bs: list[ExplorationStateModel] = []

        for bs, exps_per_bs in arms.explorations_per_bs.items():
            for exp in exps_per_bs.explorations:
                if (
                    exp.round_number == job.num_pruning_rounds
                    and exp.state == State.Converged
                ):
                    good_bs.append(exp)
                    break

        if len(good_bs) == 0:
            raise ZeusBSOValueError("While creating arms, no batch size is selected")

        print(
            f"Construct MAB for {job.job_id} with arms {[exp.batch_size for exp in good_bs]}"
        )

        new_arms: list[GaussianTsArmStateModel] = []

        # Fit the arm for each good batch size.
        for i, exp in enumerate(good_bs):
            history = await self.service.get_measurements_of_bs(
                BatchSizeBase(job_id=job.job_id, batch_size=exp.batch_size)
            )

            rewards = []
            # Collect rewards starting from the most recent ones and backwards.
            for m in history.measurements:
                rewards.append(
                    -zeus_cost(m.energy, m.time, job.eta_knob, job.max_power)
                )

            new_arms.append(
                self.fit_arm(
                    BatchSizeBase(job_id=exp.job_id, batch_size=exp.batch_size),
                    job.mab_prior_mean,
                    job.mab_prior_precision,
                    np.array(rewards),
                )
            )

        self.service.create_arms(new_arms)
        self.service.update_job_stage(
            UpdateJobStage(job_id=job.job_id, stage=Stage.MAB)
        )
        return new_arms

    async def report(self, job: JobState, current_meausurement: MeasurementOfBs):
        batch_size_key = BatchSizeBase(
            job_id=job.job_id, batch_size=current_meausurement.batch_size
        )

        # Get measurements of this bs in descending order. At most window_size length
        history = await self.service.get_measurements_of_bs(batch_size_key)

        # Add current measurement to the window
        if len(history.measurements) > job.window_size and job.window_size > 0:
            # if the history is already above the window size, pop the last one.
            history.measurements.pop()
            history.measurements.reverse() # Now ascending order.
        
        history.measurements.append(current_meausurement)

        print(f"History of bs: {[
                -zeus_cost(m.energy, m.time, job.eta_knob, job.max_power)
                for m in history.measurements
            ]}, max_power({job.max_power}), eta_knob({job.eta_knob}))")
        arm_rewards = np.array(
            [
                -zeus_cost(m.energy, m.time, job.eta_knob, job.max_power)
                for m in history.measurements
            ]
        )
        print(f"Arm_rewards: {arm_rewards}")
        arm = await self.service.get_arm(batch_size_key)

        if arm == None:
            raise ZeusBSOValueError(
                f"MAB stage but Arm for batch size({current_meausurement.batch_size}) is not found."
            )
        new_arm = self.fit_arm(
            batch_size_key, job.mab_prior_mean, job.mab_prior_precision, arm_rewards
        )

        await self.service.update_arm_state(current_meausurement, new_arm)

        arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
        self._log(
            f"{job.job_id} @ {current_meausurement.batch_size}: "
            f"arm_rewards = [{arm_rewards_repr}], reward_prec = {new_arm.reward_precision}"
        )

    def _log(self, msg: str):
        print(f"[GaussianTs] {msg}")
