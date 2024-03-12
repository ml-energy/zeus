"""
Pydantic model(JobModel) -> DB operation(Job) -> result pydantic model(JobModel). 
"""

from uuid import UUID
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.orm.util import identity_key
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import Job
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOServiceBadRequestError
from zeus.optimizer.batch_size.server.job.commands import (
    CreateJob,
    UpdateExpDefaultBs,
    UpdateGeneratorState,
    UpdateJobMinCost,
    UpdateJobStage,
)
from zeus.optimizer.batch_size.server.job.models import JobState
from sqlalchemy import and_, select, update
from sqlalchemy.orm import joinedload


class JobStateRepository(DatabaseRepository):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_job(self, job_id: UUID) -> JobState | None:
        # Only parse jobSpec + batch_sizes: list[int], without specific states of each batch_size
        try:
            stmt = (
                select(Job)
                .where(Job.job_id == job_id)
                .options(joinedload(Job.batch_sizes))
            )
            job = await self.session.scalar(stmt)

            if job == None:
                self._log("get_job: NoResultFound")
                return None

            self.fetched_job = job
            return JobState.from_orm(job)
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_job: {str(err)}")
            raise err

    def get_job_from_session(self, job_id: UUID) -> JobState | None:
        # https://stackoverflow.com/questions/70441711/python-conditional-async-method
        # https://stackoverflow.com/questions/33958770/sqlalchemy-get-items-from-the-identity-map-not-only-by-primary-key
        # https://stackoverflow.com/questions/5869514/sqlalchemy-identity-map-question
        # job_key = identity_key(Job, job_id)
        # return self.session.identity_map.get(job_key)

        if self.fetched_job == None or self.fetched_job.job_id != job_id:
            return None
        return self.fetched_job

    def update_exp_default_bs(self, updated_bs: UpdateExpDefaultBs):
        self.fetched_job.exp_default_batch_size = updated_bs.exp_default_batch_size

    def update_stage(self, updated_stage: UpdateJobStage):
        self.fetched_job.stage = updated_stage.stage

    def update_min(self, updated_min: UpdateJobMinCost):
        self.fetched_job.min_cost = updated_min.min_cost
        self.fetched_job.min_batch_size = updated_min.min_batch_size

    def update_generator_state(self, updated_state: UpdateGeneratorState):
        self.fetched_job.mab_random_generator_state = updated_state.state

    def create_job(self, new_job: CreateJob) -> None:
        self.session.add(new_job.to_orm())

    def _log(self, msg: str):
        print(f"[JobStateRepository]: {msg}")

    def check_job_fetched(self, job_id: UUID) -> None:
        if self.fetched_job == None or self.fetched_job.job_id != job_id:
            raise ZeusBSOServiceBadRequestError(
                f"check_job_fetched: {job_id} is not currently in the session"
            )
