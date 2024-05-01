"""Repository for manipulating Job table."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import JobTable
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
from zeus.utils.logging import get_logger

logger = get_logger(__name__)


class JobStateRepository(DatabaseRepository):
    """Repository that provides basic interfaces to interact with Job table."""

    def __init__(self, session: AsyncSession):
        """Set db session and intialize job. We are working with only one job per session."""
        super().__init__(session)
        self.fetched_job: JobTable | None = None

    async def get_job(self, job_id: str) -> JobState | None:
        """Get job State, which includes jobSpec + batch_sizes(list[int]), without specific states of each batch_size.

        Args:
            job_id: Job id.

        Returns:
            set fetched_job and return `JobState` if we found a job, unless return None.
        """
        stmt = select(JobTable).where(JobTable.job_id == job_id)
        job = await self.session.scalar(stmt)

        if job is None:
            logger.info("get_job: NoResultFound")
            return None

        self.fetched_job = job
        return JobState.from_orm(job)

    def get_job_from_session(self, job_id: str) -> JobState | None:
        """Get a job that was fetched from this session.

        Args:
            job_id: Job id.

        Returns:
            Corresponding `JobState`. If none was found, return None.
        """
        if self.fetched_job is None or self.fetched_job.job_id != job_id:
            return None
        return JobState.from_orm(self.fetched_job)

    def update_exp_default_bs(self, updated_bs: UpdateExpDefaultBs) -> None:
        """Update exploration default batch size on fetched job.

        Args:
            updated_bs: Job Id and new batch size.
        """
        if self.fetched_job is None:
            raise ZeusBSOServiceBadOperationError("No job is fetched.")

        if updated_bs.job_id == self.fetched_job.job_id:
            self.fetched_job.exp_default_batch_size = updated_bs.exp_default_batch_size
        else:
            raise ZeusBSOValueError(
                f"Unknown job_id ({updated_bs.job_id}). Expecting {self.fetched_job.job_id}"
            )

    def update_stage(self, updated_stage: UpdateJobStage) -> None:
        """Update stage on fetched job.

        Args:
            updated_stage: Job Id and new stage.
        """
        if self.fetched_job is None:
            raise ZeusBSOServiceBadOperationError("No job is fetched.")

        if self.fetched_job.job_id == updated_stage.job_id:
            self.fetched_job.stage = updated_stage.stage
        else:
            raise ZeusBSOValueError(
                f"Unknown job_id ({updated_stage.job_id}). Expecting {self.fetched_job.job_id}"
            )

    def update_min(self, updated_min: UpdateJobMinCost) -> None:
        """Update exploration min training cost and corresponding batch size on fetched job.

        Args:
            updated_min: Job Id, new min cost and batch size.
        """
        if self.fetched_job is None:
            raise ZeusBSOServiceBadOperationError("No job is fetched.")

        if self.fetched_job.job_id == updated_min.job_id:
            self.fetched_job.min_cost = updated_min.min_cost
            self.fetched_job.min_cost_batch_size = updated_min.min_cost_batch_size
        else:
            raise ZeusBSOValueError(
                f"Unknown job_id ({updated_min.job_id}). Expecting {self.fetched_job.job_id}"
            )

    def update_generator_state(self, updated_state: UpdateGeneratorState) -> None:
        """Update generator state on fetched job.

        Args:
            updated_state: Job Id and new generator state.
        """
        if self.fetched_job is None:
            raise ZeusBSOServiceBadOperationError("No job is fetched.")

        if self.fetched_job.job_id == updated_state.job_id:
            self.fetched_job.mab_random_generator_state = updated_state.state
        else:
            raise ZeusBSOValueError(
                f"Unknown job_id ({updated_state.job_id}). Expecting {self.fetched_job.job_id}"
            )

    def create_job(self, new_job: CreateJob) -> None:
        """Create a new job by adding a new job to the session.

        Args:
            new_job: Job configuration for a new job.
        """
        self.session.add(new_job.to_orm())

    def check_job_fetched(self, job_id: str) -> bool:
        """Check if this job is already fetched before.

        Args:
            job_id: Job id.

        Returns:
            True if this job was fetched and in session. Otherwise, return false.
        """
        return not (self.fetched_job is None or self.fetched_job.job_id != job_id)

    async def delete_job(self, job_id: str) -> bool:
        """Delete the job of a given job_Id.

        Args:
            job_id: Job id.

        Returns:
            True if the job got deleted.
        """
        stmt = select(JobTable).where(JobTable.job_id == job_id)
        job = await self.session.scalar(stmt)

        if job is None:
            return False

        # We can't straight delete using a query, since some db such as sqlite
        # Foreign Key is default to OFF, so "on delete = cascade" will not be fired.
        await self.session.delete(job)
        return True
