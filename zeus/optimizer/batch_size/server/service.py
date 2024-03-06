from uuid import UUID
from build.lib.zeus.util.metric import zeus_cost
import numpy as np
from zeus.optimizer.batch_size.common import ZeusBSOServiceError, ZeusBSOValueError
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpsertGaussianTsArmState,
    UpdateExploration,
    UpdateGaussianTsArmState,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationOfBs,
    GaussianTsArmStateModel,
    MeasurementOfBs,
    MeasurementsPerBs,
)
from zeus.optimizer.batch_size.server.batch_size_state.repository import (
    BatchSizeStateRepository,
)
from zeus.optimizer.batch_size.server.commands import CreateArms
from zeus.optimizer.batch_size.server.job.commands import (
    UpdateExpDefaultBs,
    UpdateJobMinCost,
)
from zeus.optimizer.batch_size.server.job.models import JobState, Stage
from zeus.optimizer.batch_size.server.job.repository import JobStateRepository
from zeus.optimizer.batch_size.server.mab import GaussianTS


class ZeusService:
    def __init__(self, bs_repo: BatchSizeStateRepository, job_repo: JobStateRepository):
        self.bs_repo = bs_repo
        self.job_repo = job_repo

    async def update_exploration(
        self,
        measurement: MeasurementOfBs,
        updated_exp: UpdateExploration,
        job: JobState,
    ) -> None:
        """
        1. add measurement
        2. update that exploration
        ----
        3. Update min cost if it is needed
        """
        self.bs_repo.add_measurement(measurement)
        self.bs_repo.update_exploration(updated_exp)
        await self._update_min_if_needed(measurement, job)

    async def update_arm_state(
        self,
        measurement: MeasurementOfBs,
        updated_arm: UpdateGaussianTsArmState,
        job: JobState,
    ):
        """
        1. add measurement
        2. update arm_state
        -----
        3. Update min cost if it is needed
        """
        self.bs_repo.add_measurement(measurement)
        self.bs_repo.update_arm_state(updated_arm)
        await self._update_min_if_needed(measurement, job)

    async def report_concurrent_job(self, measurement: MeasurementOfBs, job: JobState):
        """
        1. add measurement
        -----
        2. update min cost
        """
        self.bs_repo.add_measurement(measurement)
        await self._update_min_if_needed(measurement, job)

    async def create_arms(self, arms: CreateArms):
        """
        1. From Explorations,
        2. Get converged bs (good_bs)
        3. get measurement of each of good_bs
        4. create arms
        -------
        5. update stage to MAB!
        """
        if arms.job.stage == Stage.MAB:
            raise ZeusBSOServiceError(
                "Trying to create arms when we are already in a MAB stage"
            )
        good_bs: list[ExplorationOfBs] = []

        for bs, exps_per_bs in arms.explorations.items():
            for exp in exps_per_bs.explorations:
                if (
                    exp.trial_number == arms.job.num_pruning_rounds
                    and exp.state == ExplorationOfBs.State.Converged
                ):
                    good_bs.append(exp)
                    break

        if len(good_bs) == 0:
            raise ZeusBSOValueError("While creating arms, no batch size is selected")

        print(
            f"Construct MAB for {arms.job.job_id} with arms {[exp.batch_size for exp in good_bs]}"
        )

        new_arms: list[GaussianTsArmStateModel] = []

        # Fit the arm for each good batch size.
        for i, exp in enumerate(good_bs):
            history = await self.bs_repo.get_measurements_of_bs(
                BatchSizeBase(job_id=arms.job.job_id, batch_size=exp.batch_size),
                arms.job.window_size,
            )

            rewards = []
            # Collect rewards starting from the most recent ones and backwards.
            for m in history.measurements:
                rewards.append(
                    -zeus_cost(m.energy, m.time, arms.job.eta_knob, arms.job.max_power)
                )

            new_arms.append(
                GaussianTS.fit_arm(
                    BatchSizeBase(job_id=exp.job_id, batch_size=exp.batch_size),
                    arms.job.mab_prior_mean,
                    arms.job.mab_prior_precision,
                    np.array(rewards),
                )
            )

        self.bs_repo.create_arms(new_arms)

    async def update_pruning_rounds(
        self, exp: CreateExploration, updated_default_bs: UpdateExpDefaultBs
    ):
        """
        1. add exploration
        --------
        2. Update exp_default bs
        """
        self.bs_repo.add_exploration(exp)
        await self.job_repo.update_exp_default_bs(updated_default_bs)

    def add_exploration(self, exp: CreateExploration):
        """
        add exploration
        """
        self.bs_repo.add_exploration(exp)

    # JOBSTATE
    async def get_random_choices(self):
        """
        If seed is not none,
        1. get generator state
        2. Get the sequence
        3. Update state
        """

    async def _update_min_if_needed(
        self,
        measurement: MeasurementOfBs,
        job: JobState,
    ):
        cur_cost = zeus_cost(
            measurement.energy, measurement.time, job.eta_knob, job.max_power
        )
        if job.min_cost > cur_cost:
            await self.job_repo.update_min(
                UpdateJobMinCost(
                    job_id=job.job_id,
                    min_cost=cur_cost,
                    min_batch_size=MeasurementOfBs.batch_size,
                )
            )
