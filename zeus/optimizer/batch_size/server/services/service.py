"""Zeus batch size optimizer service layer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Tuple
from uuid import UUID

import numpy as np
from numpy.random import Generator as np_Generator
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpdateExploration,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerBs,
    ExplorationsPerJob,
    GaussianTsArmStateModel,
    MeasurementOfBs,
    MeasurementsPerBs,
)
from zeus.optimizer.batch_size.server.batch_size_state.repository import (
    BatchSizeStateRepository,
)
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServiceBadOperationError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.job.commands import (
    CreateJob,
    UpdateExpDefaultBs,
    UpdateGeneratorState,
    UpdateJobMinCost,
    UpdateJobStage,
)
from zeus.optimizer.batch_size.server.job.models import JobState
from zeus.optimizer.batch_size.server.job.repository import JobStateRepository
from zeus.optimizer.batch_size.server.services.commands import (
    GetNormal,
    GetRandomChoices,
)
from zeus.util import zeus_cost


class ZeusService:
    """Zeus Service that interacts with database using repository.

    Provides application layer methods to communicate with database.
    Each method is one or more number of db operations that have to be done at the same time.
    """

    def __init__(self, db_session: AsyncSession):
        """Set up repositories to use to talk to database"""
        self.bs_repo = BatchSizeStateRepository(db_session)
        self.job_repo = JobStateRepository(db_session)

    async def get_arms(self, job_id: UUID) -> list[GaussianTsArmStateModel]:
        """Get GaussianTs arm states for all arms(job_id, batch size)

        Args:
            job_id: Job id

        Return:
            list of arms
        """
        return await self.bs_repo.get_arms(job_id)

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmStateModel | None:
        """Get arm state for one arm

        Args:
            bs: (job_id, batch size) pair that represents one arm

        Return:
            Result arm state or None if we cannot find that arm
        """
        return await self.bs_repo.get_arm(bs)

    async def get_explorations_of_job(self, job_id: UUID) -> ExplorationsPerJob:
        """Get all explorations we have done for that job

        Args:
            job_id: Job id

        Return:
            list of explorations per each batch size
        """
        return await self.bs_repo.get_explorations_of_job(job_id)

    async def get_explorations_of_bs(self, bs: BatchSizeBase) -> ExplorationsPerBs:
        """Get explorations for one batch size

        Args:
            bs: (job_id, batch size) that represents one arm

        Return:
            List of explorations for that batch size
        """
        return await self.bs_repo.get_explorations_of_bs(bs)

    async def update_exploration(
        self,
        measurement: MeasurementOfBs,
        updated_exp: UpdateExploration,
    ) -> None:
        """Update exploration state. (1) add measurement which is an evidence of updating the exploration,
        (2) update the exploration, (3) we update the min training cost observed so far if we have to.

        Args:
            measurement: Result of training that batch size
            updated_exp: Updated state of exploration

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        job = self._get_job(measurement.job_id)
        self.bs_repo.add_measurement(measurement)
        await self.bs_repo.update_exploration(updated_exp)
        self._update_min_if_needed(measurement, job)

    async def update_arm_state(
        self,
        measurement: MeasurementOfBs,
        updated_arm: GaussianTsArmStateModel,
    ) -> None:
        """Update arm state. (1) add measurement which is the evidence of updating the arm state, (2) update arm state
        (3) update the min training cost observed so far if we have to.

        Args:
            measurement: Result of training that batch size
            updated_arm: Updated arm state

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        job = self._get_job(measurement.job_id)
        self.bs_repo.add_measurement(measurement)
        await self.bs_repo.update_arm_state(updated_arm)
        self._update_min_if_needed(measurement, job)

    def report_concurrent_job(self, measurement: MeasurementOfBs) -> None:
        """Report concurrent job submission. (1) add measurement and (2) update the min training cost observed so far

        Args:
            measurement: Result of training that batch size

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        job = self._get_job(measurement.job_id)
        self.bs_repo.add_measurement(measurement)
        self._update_min_if_needed(measurement, job)

    def update_exp_default_bs(self, updated_default_bs: UpdateExpDefaultBs) -> None:
        """Update the default batch size for exploration

        Args:
            updated_default_bs: Job Id and new default batch size

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """

        self._check_job_fetched(updated_default_bs.job_id)
        self.job_repo.update_exp_default_bs(updated_default_bs)

    def add_exploration(self, exp: CreateExploration) -> None:
        """Add new exploration

        Args:
            exp: New exploration state to create

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        self._check_job_fetched(exp.job_id)
        self.bs_repo.add_exploration(exp)

    def get_random_choices(self, choice: GetRandomChoices) -> np.ndarray[Any, Any]:
        """Get randome choices based on job's seed. If seed is not None (set by the user) we get the random choices from the
        generator that is stored in the database. Otherwise, we get random choices based on random seed.

        Args:
            choice: Job id and list of choices

        Return:
            reuslt random choices

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """

        arr = np.array(choice.choices)
        ret = self._get_generator(choice.job_id)
        should_update = ret[1]
        res = ret[0].choice(arr, len(arr), replace=False)

        if should_update:
            # If we used the generator from database, should update the generator state after using it
            self.job_repo.update_generator_state(
                UpdateGeneratorState(
                    job_id=choice.job_id, state=json.dumps(ret[0].__getstate__())
                )
            )

        return res

    def get_normal(self, arg: GetNormal) -> float:
        """Sample from normal distribution and update the generator state if seed was set

        Args:
            arg: args for `numpy.random.normal`, which is loc(mean of distribution) and scale(stdev of distribution)

        Return:
            Drawn sample.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """

        ret = self._get_generator(arg.job_id)
        res = ret[0].normal(arg.loc, arg.scale)
        should_update = ret[1]

        if should_update:
            # If we used the generator from database, should update the generator state after using it
            self.job_repo.update_generator_state(
                UpdateGeneratorState(
                    job_id=arg.job_id, state=json.dumps(ret[0].__getstate__())
                )
            )

        return res

    async def get_job(self, job_id: UUID) -> JobState | None:
        """Get job from database

        Args:
            job_id: Job Id

        Return:
            JobState if we found one, None if we couldn't find a job matching the job id.
        """
        return await self.job_repo.get_job(job_id)

    def create_job(self, new_job: CreateJob) -> None:
        """Create a new job.

        Args:
            new_job: Configuration of a new job
        """
        return self.job_repo.create_job(new_job)

    async def get_measurements_of_bs(self, bs: BatchSizeBase) -> MeasurementsPerBs:
        """Get a windowed list of measurement for that batch size. If job's window size is not set(= 0),
        get all measurements

        Args:
            bs: (job_id, batch size) pair.

        Return:
            list of windowed measurements in descending order for that (job_id, batch size)

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        job = self._get_job(bs.job_id)
        return await self.bs_repo.get_measurements_of_bs(
            BatchSizeBase(job_id=bs.job_id, batch_size=bs.batch_size),
            job.window_size,
        )

    def create_arms(self, new_arms: list[GaussianTsArmStateModel]) -> None:
        """Create GuassianTs arms for the job

        Args:
            new_arms: List of new arm states

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        if len(new_arms) != 0:
            self._check_job_fetched(new_arms[0].job_id)
            self.bs_repo.create_arms(new_arms)

    def update_job_stage(self, updated_stage: UpdateJobStage) -> None:
        """Update the job stage (Pruning -> MAB).

        Args:
            updated_stage: Updated stage.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        self._check_job_fetched(updated_stage.job_id)
        self.job_repo.update_stage(updated_stage)

    def _update_min_if_needed(
        self,
        measurement: MeasurementOfBs,
        job: JobState,
    ):
        """Update the min training cost and corresponding batch size based on the trianing result."""
        cur_cost = zeus_cost(
            measurement.energy, measurement.time, job.eta_knob, job.max_power
        )
        if job.min_cost == None or job.min_cost > cur_cost:
            self.job_repo.update_min(
                UpdateJobMinCost(
                    job_id=job.job_id,
                    min_cost=cur_cost,
                    min_batch_size=measurement.batch_size,
                )
            )

    def _get_generator(self, job_id: UUID) -> Tuple[np_Generator, bool]:
        """Get generator based on job_id. If mab_seed is not none, we should update the state after using generator

        Return:
            Tuple of [Generator, if we should update state]
        """
        jobState = self._get_job(job_id)

        rng = np.random.default_rng(int(datetime.now().timestamp()))

        should_update = jobState.mab_seed != None
        if jobState.mab_seed != None:
            if jobState.mab_random_generator_state == None:
                raise ZeusBSOValueError(
                    f"Seed is set but generator state is none. Should be impossible"
                )

            state = json.loads(jobState.mab_random_generator_state)
            rng.__setstate__(state)

        return (rng, should_update)

    def _get_job(self, job_id: UUID) -> JobState:
        """Get the job from the session. If we couldn't find the job, raise a `ZeusBSOServiceBadOperationError`."""
        res = self.job_repo.get_job_from_session(job_id)
        if res == None:
            raise ZeusBSOServiceBadOperationError(
                f"Should have fetched the job first or job does not exist(job_id = {job_id})"
            )
        return res

    def _check_job_fetched(self, job_id: UUID) -> None:
        """Check if we fetched the job in the current session. If we didn't raise a `ZeusBSOServiceBadOperationError`."""
        if not self.job_repo.check_job_fetched(job_id):
            raise ZeusBSOServiceBadOperationError(
                f"check_job_fetched: {job_id} is not currently in the session"
            )
