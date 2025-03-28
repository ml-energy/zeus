"""Zeus batch size optimizer service layer."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
from numpy.random import Generator as np_Generator
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateConcurrentTrial,
    CreateExplorationTrial,
    CreateMabTrial,
    CreateTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    GaussianTsArmState,
    Trial,
    TrialResultsPerBs,
)
from zeus.optimizer.batch_size.server.batch_size_state.repository import (
    BatchSizeStateRepository,
)
from zeus.optimizer.batch_size.server.database.schema import TrialStatus, TrialType
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
    UpdateArm,
)
from zeus.utils.metric import zeus_cost


class ZeusService:
    """Zeus Service that interacts with database using repository.

    Provides application layer methods to communicate with database.
    Each method is one or more number of db operations that have to be done at the same time.
    """

    def __init__(self, db_session: AsyncSession):
        """Set up repositories to use to talk to database."""
        self.bs_repo = BatchSizeStateRepository(db_session)
        self.job_repo = JobStateRepository(db_session)

    async def get_arms(self, job_id: str) -> list[GaussianTsArmState]:
        """Get GaussianTs arm states for all arms(job_id, batch size).

        Args:
            job_id: Job id

        Returns:
            list of arms
        """
        return await self.bs_repo.get_arms(job_id)

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmState | None:
        """Get arm state for one arm.

        Args:
            bs: (job_id, batch size) pair that represents one arm

        Returns:
            Result arm state or None if we cannot find that arm
        """
        return await self.bs_repo.get_arm(bs)

    async def get_explorations_of_job(self, job_id: str) -> ExplorationsPerJob:
        """Get all explorations we have done for that job.

        Args:
            job_id: Job id

        Returns:
            list of explorations per each batch size
        """
        return await self.bs_repo.get_explorations_of_job(job_id)

    def update_trial(self, updated_trial: UpdateTrial) -> None:
        """Update trial.

        (1) update the corresponding trial.
        (2) we update the min training cost observed so far if we have to.

        Args:
            updated_trial: Result of training that batch size

        Raises:
            [`ZeusBSOServiceBadOperationError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOServiceBadOperationError]: When we didn't fetch the job or trial during this session. This operation should have
                    fetched the job and trial first. Also, check if trial type is matching with fetched trial's type.
        """
        trial = self._get_trial(
            ReadTrial(
                job_id=updated_trial.job_id,
                batch_size=updated_trial.batch_size,
                trial_number=updated_trial.trial_number,
            )
        )
        if trial.status != TrialStatus.Dispatched:
            raise ZeusBSOServiceBadOperationError("Trial already has a result.")

        self.bs_repo.updated_current_trial(updated_trial)

        # Update the corresponding batch size's min cost if needed.
        if updated_trial.status != TrialStatus.Failed:
            job = self._get_job(updated_trial.job_id)
            if updated_trial.energy is None or updated_trial.time is None:
                raise ZeusBSOValueError(
                    "Energy and time should be set if the trial is not failed."
                )
            cur_cost = zeus_cost(
                updated_trial.energy, updated_trial.time, job.eta_knob, job.max_power
            )
            if job.min_cost is None or job.min_cost > cur_cost:
                self.job_repo.update_min(
                    UpdateJobMinCost(
                        job_id=job.job_id,
                        min_cost=cur_cost,
                        min_cost_batch_size=updated_trial.batch_size,
                    )
                )

    def update_arm_state(
        self,
        arm: UpdateArm,
    ) -> None:
        """Update arm state.

        Args:
            arm: Updated arm state.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job or trial during this session. This operation should have
                    fetched the job and trial first. Also, check if trial type is matching with fetched trial's type.
        """
        self._check_job_fetched(arm.trial.job_id)
        trial = self._get_trial(
            ReadTrial(
                job_id=arm.trial.job_id,
                batch_size=arm.trial.batch_size,
                trial_number=arm.trial.trial_number,
            )
        )
        if trial.type != TrialType.MAB:
            raise ZeusBSOServiceBadOperationError(
                "Cannot update an arm since this trial is not issued from MAB stage."
            )
        self.bs_repo.update_arm_state(arm.updated_arm)

    def update_exp_default_bs(self, updated_default_bs: UpdateExpDefaultBs) -> None:
        """Update the default batch size for exploration.

        Args:
            updated_default_bs: Job Id and new default batch size

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        self._check_job_fetched(updated_default_bs.job_id)
        self.job_repo.update_exp_default_bs(updated_default_bs)

    async def create_trial(
        self, trial: CreateExplorationTrial | CreateMabTrial | CreateConcurrentTrial
    ) -> ReadTrial:
        """Create a new trial.

        Args:
            trial: New trial to create.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        self._check_job_fetched(trial.job_id)
        trial_number = await self.bs_repo.get_next_trial_number(trial.job_id)
        self.bs_repo.create_trial(
            CreateTrial(**trial.dict(), trial_number=trial_number)
        )
        return ReadTrial(
            job_id=trial.job_id, batch_size=trial.batch_size, trial_number=trial_number
        )

    def get_random_choices(self, choice: GetRandomChoices) -> np.ndarray[Any, Any]:
        """Get randome choices based on job's seed.

        If seed is not None (set by the user) we get the random choices from the generator that is stored in the database.
        Otherwise, we get random choices based on random seed.

        Args:
            choice: Job id and list of choices

        Returns:
            reuslt random choices

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        arr = np.array(choice.choices)
        rng, should_update = self._get_generator(choice.job_id)
        res = rng.choice(arr, len(arr), replace=False)

        if should_update:
            # If we used the generator from database, should update the generator state after using it
            self.job_repo.update_generator_state(
                UpdateGeneratorState(
                    job_id=choice.job_id, state=json.dumps(rng.__getstate__())
                )
            )

        return res

    def get_normal(self, arg: GetNormal) -> float:
        """Sample from normal distribution and update the generator state if seed was set.

        Args:
            arg: args for `numpy.random.normal`, which is loc(mean of distribution) and scale(stdev of distribution)

        Returns:
            Drawn sample.

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        rng, should_update = self._get_generator(arg.job_id)
        res = rng.normal(arg.loc, arg.scale)

        if should_update:
            # If we used the generator from database, should update the generator state after using it
            self.job_repo.update_generator_state(
                UpdateGeneratorState(
                    job_id=arg.job_id, state=json.dumps(rng.__getstate__())
                )
            )

        return res

    async def get_job(self, job_id: str) -> JobState | None:
        """Get job from database.

        Args:
            job_id: Job Id

        Returns:
            JobState if we found one, None if we couldn't find a job matching the job id.
        """
        return await self.job_repo.get_job(job_id)

    async def get_trial(self, trial: ReadTrial) -> Trial | None:
        """Get a trial from database.

        Args:
            trial: (Job Id, batch size, trial_number) triplet.

        Returns:
            Trial if we found one, None if we couldn't find a job matching trial.
        """
        return await self.bs_repo.get_trial(trial)

    def create_job(self, new_job: CreateJob) -> None:
        """Create a new job.

        Args:
            new_job: Configuration of a new job
        """
        return self.job_repo.create_job(new_job)

    async def get_trial_results_of_bs(self, bs: BatchSizeBase) -> TrialResultsPerBs:
        """Load window size amount of results for a given batch size. If window size <= 0, load all of them.

        Args:
            bs: (job_id, batch size) pair.

        Returns:
            list of windowed measurements in descending order for that (job_id, batch size)

        Raises:
            `ZeusBSOServiceBadOperationError`: When we didn't fetch the job during this session. This operation should have
                    fetched the job first.
        """
        job = self._get_job(bs.job_id)
        return await self.bs_repo.get_trial_results_of_bs(
            BatchSizeBase(job_id=bs.job_id, batch_size=bs.batch_size),
            job.window_size,
        )

    def create_arms(self, new_arms: list[GaussianTsArmState]) -> None:
        """Create GuassianTs arms for the job.

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

    async def delete_job(self, job_id: str) -> bool:
        """Delete the job.

        Args:
            job_id: ID of the job.

        Returns:
            True if the job is deleted. False if none was deleted
        """
        return await self.job_repo.delete_job(job_id)

    def _get_generator(self, job_id: str) -> tuple[np_Generator, bool]:
        """Get generator based on job_id. If mab_seed is not none, we should update the state after using generator.

        Returns:
            Tuple of [Generator, if we should update state]
        """
        job_state = self._get_job(job_id)

        rng = np.random.default_rng(int(datetime.now().timestamp()))

        should_update = job_state.mab_seed is not None
        if job_state.mab_seed is not None:
            if job_state.mab_random_generator_state is None:
                raise ZeusBSOValueError(
                    "Seed is set but generator state is none. Should be impossible"
                )

            state = json.loads(job_state.mab_random_generator_state)
            rng.__setstate__(state)

        return (rng, should_update)

    def _get_job(self, job_id: str) -> JobState:
        """Get the job from the session. If we couldn't find the job, raise a `ZeusBSOServiceBadOperationError`."""
        res = self.job_repo.get_job_from_session(job_id)
        if res is None:
            raise ZeusBSOServiceBadOperationError(
                f"Should have fetched the job first or job does not exist(job_id = {job_id})"
            )
        return res

    def _get_trial(self, trial: ReadTrial) -> Trial:
        """Get the job from the session. If we couldn't find the trial, raise a `ZeusBSOServiceBadOperationError`."""
        res = self.bs_repo.get_trial_from_session(trial)
        if res is None:
            raise ZeusBSOServiceBadOperationError(
                f"Should have fetched the trial first or trial does not exist(trial = {trial})"
            )
        return res

    def _check_job_fetched(self, job_id: str) -> None:
        """Check if we fetched the job in the current session. If we didn't raise a `ZeusBSOServiceBadOperationError`."""
        if not self.job_repo.check_job_fetched(job_id):
            raise ZeusBSOServiceBadOperationError(
                f"check_job_fetched: {job_id} is not currently in the session"
            )
