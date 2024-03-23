"""Thompson Sampling policy for Gaussian bandits. MAB related logic is implented here."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationState,
    GaussianTsArmState,
    Measurement,
)
from zeus.optimizer.batch_size.server.database.schema import State
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError
from zeus.optimizer.batch_size.server.job.commands import UpdateJobStage
from zeus.optimizer.batch_size.server.job.models import JobState, Stage
from zeus.optimizer.batch_size.server.services.commands import (
    ConstructMAB,
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
        """Set up zeus service to interact with database."""
        self.service = service
        self.name = "GaussianTS"

    def _fit_arm(
        self,
        bs_base: BatchSizeBase,
        prior_mean: float,
        prior_precision: float,
        rewards: np.ndarray,
    ) -> GaussianTsArmState:
        """Update the parameter distribution for one arm.

        Reference: <https://en.wikipedia.org/wiki/Conjugate_prior>

        Args:
            bs_base: job id and batch size tha represents this arm
            prior_mean: Mean of the belief prior distribution.
            prior_precision: Precision of the belief prior distribution.
            rewards: Array of rewards observed by pulling that arm.

        Returns:
            Updated arm state
        """
        if len(rewards) == 0:
            return

        variance = np.var(rewards)
        reward_prec = np.inf if variance == 0.0 else np.reciprocal(variance)

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

        # Updated state.
        return GaussianTsArmState(
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
        arms: list[GaussianTsArmState],
    ) -> int:
        """Return the arm with the largest sampled expected reward.

        Args:
            job_id: job id
            prior_precision: Precision of the belief prior distribution.
            num_exploration: How many static explorations to run when no observations are available.
            arms: list of arms

        Returns:
            batch size to use
        """
        arm_dict = {arm.batch_size: arm for arm in arms}

        # Exploration-only phase.
        # Order is random considering concurrent bandit scenarios.
        choices = self.service.get_random_choices(
            GetRandomChoices(job_id=job_id, choices=[arm.batch_size for arm in arms])
        )

        for arm in choices:
            if arm_dict[arm].num_observations < num_exploration:
                logger.info("[%s] Explore arm %s.", self.name, str(arm))
                return arm

        # Thomopson Sampling phase.
        # Sample the expected reward for each arm.
        # Assumes that each arm has been explored at least once. Otherwise,
        # a value will be sampled from the prior.

        expectations = {}  # A mapping from every arm to their sampled expected reward.
        for arm in arms:
            if arm.param_precision == prior_precision:
                logger.warning(
                    "predict_expectations called when arm '%d' is cold.",
                    arm.batch_size,
                    stacklevel=1,
                )
            expectations[arm.batch_size] = self.service.get_normal(
                GetNormal(
                    job_id=job_id,
                    loc=arm.param_mean,
                    scale=np.sqrt(np.reciprocal(arm.param_precision)),
                )
            )

        logger.info("[%s] Sampled mean rewards:", self.name)
        for arm, sample in expectations.items():
            logger.info(
                "[%s] Arm %d: mu ~ N(%.2f, %.2f) -> %.2f",
                self.name,
                arm,
                arm_dict[arm].param_mean,
                1 / arm_dict[arm].param_precision,
                sample,
            )

        bs = max(expectations, key=expectations.get)
        logger.info("%s in Thompson Sampling stage -> BS = %d", job_id, bs)
        return bs

    async def construct_mab(
        self, job: JobState, evidence: ConstructMAB
    ) -> list[GaussianTsArmState]:
        """Construct arms and initialize them.

        Args:
            job: state of job.
            evidence: Completed explorations. We create arms based on the explorations we have done during pruning stage.

        Returns:
            list of arms that we created

        Raises:
            `ValueError`: If exploration states is invalid (ex. number of pruning rounds doesn't corresponds)
            `ZeusBSOValueError`: No converged batch sizes from pruning stage.
        """
        # list of converged batch sizes from last pruning rounds
        good_bs: list[ExplorationState] = []

        for _, exps_per_bs in evidence.explorations_per_bs.items():
            for exp in exps_per_bs.explorations:
                if (
                    exp.round_number == job.num_pruning_rounds
                    and exp.state == State.Converged
                ):
                    good_bs.append(exp)
                    break

        if len(good_bs) == 0:
            raise ZeusBSOValueError("While creating arms, no batch size is selected")

        logger.info(
            "Construct MAB for %s with arms %s",
            job.job_id,
            str([exp.batch_size for exp in good_bs]),
        )

        new_arms: list[GaussianTsArmState] = []

        # Fit the arm for each good batch size.
        for _, exp in enumerate(good_bs):
            # Get windowed measurements
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
                # create an arm
                self._fit_arm(
                    BatchSizeBase(job_id=exp.job_id, batch_size=exp.batch_size),
                    job.mab_prior_mean,
                    job.mab_prior_precision,
                    np.array(rewards),
                )
            )

        # submit new arms to db
        self.service.create_arms(new_arms)
        # update job stage from pruning to mab since we created arms
        self.service.update_job_stage(
            UpdateJobStage(job_id=job.job_id, stage=Stage.MAB)
        )
        return new_arms

    async def report(self, job: JobState, current_meausurement: Measurement) -> None:
        """Based on the measurement, update the arm state.

        Args:
            job: state of the job
            current_meausurement: result of training (job id, batch_size)

        Raises:
            `ZeusBSOValueError`: When the arm (job id, batch_size) doesn't exist
        """
        # Since we're learning the reward precision, we need to
        # 1. re-compute the precision of this arm based on the reward history,
        # 2. update the arm's reward precision
        # 3. and `fit` the new MAB instance on all the reward history.
        # Note that `arm_rewards` always has more than one entry (and hence a
        # non-zero variance) because we've been through pruning exploration.
        batch_size_key = BatchSizeBase(
            job_id=job.job_id, batch_size=current_meausurement.batch_size
        )

        # Get measurements of this bs in descending order. At most window_size length
        history = await self.service.get_measurements_of_bs(batch_size_key)

        if len(history.measurements) >= job.window_size and job.window_size > 0:
            # if the history is already above the window size, pop the last one to leave the spot for the current measurement.
            history.measurements.pop()
            history.measurements.reverse()  # Now ascending order.

        # Add current measurement to the window
        history.measurements.append(current_meausurement)

        arm_rewards = np.array(
            [
                -zeus_cost(m.energy, m.time, job.eta_knob, job.max_power)
                for m in history.measurements
            ]
        )
        logger.info("Arm_rewards: %s", str(arm_rewards))
        arm = await self.service.get_arm(batch_size_key)

        if arm is None:
            raise ZeusBSOValueError(
                f"MAB stage but Arm for batch size({current_meausurement.batch_size}) is not found."
            )

        # Get a new arm state based on observation
        new_arm = self._fit_arm(
            batch_size_key, job.mab_prior_mean, job.mab_prior_precision, arm_rewards
        )

        # update the new arm state in db
        await self.service.update_arm_state(current_meausurement, new_arm)

        arm_rewards_repr = ", ".join([f"{r:.2f}" for r in arm_rewards])
        logger.info(
            "%s @ %d: arm_rewards = [%s], reward_prec = %.2f",
            job.job_id,
            current_meausurement.batch_size,
            arm_rewards_repr,
            new_arm.reward_precision,
        )
