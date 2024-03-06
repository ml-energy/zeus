"""
Pydantic model(JobModel) -> DB operation(Job) -> result pydantic model(JobModel). 
"""

from uuid import UUID
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import Job
from zeus.optimizer.batch_size.server.job.commands import (
    CreateJob,
    UpdateExpDefaultBs,
    UpdateJobMinCost,
    UpdateJobStage,
)
from zeus.optimizer.batch_size.server.job.models import JobState
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import and_, select, update
from sqlalchemy.orm import joinedload


class JobStateRepository(DatabaseRepository):
    async def get_job(
        self,
        job_id: UUID,
    ) -> (
        JobState | None
    ):  # Only parse jobSpec + batch_sizes: list[int], without specific states of each batch_size

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
            return JobState.from_orm(job)
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_job: {str(err)}")
            raise err

    def update_exp_default_bs(self, update: UpdateExpDefaultBs):
        pass

    def update_stage(self, update: UpdateJobStage):
        # self.session
        pass

    def update_min(self, update: UpdateJobMinCost):
        pass

    def create_job(self, new_job: CreateJob) -> None:
        self.session.add(new_job.to_orm())

    def _log(self, msg: str):
        print(f"[JobStateRepository]: {msg}")
