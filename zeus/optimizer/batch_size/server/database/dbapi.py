from uuid import UUID
from build.lib.zeus.analyze import energy
from build.lib.zeus.optimizer import batch_size
from sqlalchemy import update, and_

from sqlalchemy.orm import session

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError

from zeus.optimizer.batch_size.common import JobSpec, TrainingResult
from zeus.optimizer.batch_size.server.database.models import (
    BatchSize,
    ExplorationState,
    GaussianTsArmState,
    Job,
    Measurement,
)

# https://github.com/ThomasAitken/demo-fastapi-async-sqlalchemy/blob/main/backend/app/crud/user.py


class DBapi(object):

    @staticmethod
    async def create_job(db: AsyncSession, jobSpec: JobSpec) -> Job:
        """Create job and return the job if successful. If job already exists, return None"""
        try:
            job_obj = Job()
            job_obj.populate_from_job_spec(jobSpec)
            db.add(job_obj)
            for bs in jobSpec.batch_sizes:
                bs_obj = BatchSize(job_id=jobSpec.job_id, batch_size=bs)
                db.add(bs_obj)  # Create each batch size
            await db.commit()
            await db.refresh(job_obj)
            return job_obj
        except Exception as err:
            await db.rollback()
            DBapi._log(f"create_job: {str(err)}")
            raise err

    @staticmethod
    async def get_job(db: AsyncSession, job_id: UUID) -> Job | None:
        try:
            return await db.get_one(Job, job_id)
        except NoResultFound:
            await db.rollback()
            DBapi._log("get_job: NoResultFound")
            return None
        except Exception as err:
            await db.rollback()
            DBapi._log(f"get_job: {str(err)}")
            raise err

    @staticmethod
    async def delete_job(db: AsyncSession, job: JobSpec):
        pass

    @staticmethod
    async def add_exploration(
        db: AsyncSession,
        job_id: UUID,
        batch_size: int,
        trial_number: int,
        state: ExplorationState.State,
    ) -> None:
        try:
            exp = ExplorationState(
                job_id=job_id,
                batch_size=batch_size,
                trial_number=trial_number,
                state=state,
            )
            db.add(exp)
            await db.commit()
        except Exception as err:
            await db.rollback()
            DBapi._log(
                f"add_exploration(job_id={job_id}, bs={batch_size}, trial_number={trial_number}, state={state}): {str(err)}"
            )
            raise err

    @staticmethod
    async def update_exploration(
        db: AsyncSession,
        job_id: UUID,
        batch_size: int,
        trial_number: int,
        state: ExplorationState.State,
        cost: float,
        commit: bool = True,
    ) -> None:
        try:
            stmt = (
                update(ExplorationState)
                .where(
                    and_(
                        ExplorationState.job_id == job_id,
                        ExplorationState.batch_size == batch_size,
                        ExplorationState.trial_number == trial_number,
                    )
                )
                .values(state=state, cost=cost)
            )
            await db.execute(stmt)
            if commit:
                await db.commit()
        except Exception as err:
            await db.rollback()
            DBapi._log(f"update_exploration: {str(err)}")
            raise err

    @staticmethod
    def add_arm(db: AsyncSession, job_id: UUID, batch_size: int) -> None:
        db.add(GaussianTsArmState(job_id=job_id, batch_size=batch_size))

    @staticmethod
    def add_measurement(
        db: AsyncSession, result: TrainingResult, converged: bool
    ) -> None:
        measurement = Measurement(
            job_id=result.job_id,
            batch_size=result.batch_size,
            time=result.time,
            energy=result.energy,
            converged=converged,
        )
        db.add(measurement)
        return

    @staticmethod
    async def update_exp_default_bs(
        db: AsyncSession, job_id: UUID, exp_default_bs: int, commit: bool = True
    ) -> None:
        try:
            stmt = (
                update(Job)
                .where(and_(Job.job_id == job_id))
                .values(exp_default_batch_size=exp_default_bs)
            )
            await db.execute(stmt)
            if commit:
                await db.commit()
        except Exception as err:
            await db.rollback()
            DBapi._log(f"update_exp_default_bs: {str(err)}")
            raise err

    @staticmethod
    async def get_history(db: AsyncSession, job_id: UUID):
        pass

    @staticmethod
    async def update_exploration_state(db: AsyncSession):
        pass

    @staticmethod
    async def upsert_arm_state(db: AsyncSession):
        pass

    @staticmethod
    async def get_arm_state(db: AsyncSession):
        pass

    @staticmethod
    async def get_measurement(db: AsyncSession):
        pass

    @staticmethod
    def _log(message: str):
        print(f"[DBapi] {message}")
