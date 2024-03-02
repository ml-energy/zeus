from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.exc import NoResultFound
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
    def add_job(db: AsyncSession, jobSpec: JobSpec) -> None:
        """Create job and return the job if successful. If job already exists, return None"""
        job_obj = Job()
        job_obj.populate_from_job_spec(jobSpec)
        for bs in jobSpec.batch_sizes:
            job_obj.batch_sizes.append(BatchSize(job_id=jobSpec.job_id, batch_size=bs))
        db.add(job_obj)
        return job_obj

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
    async def get_job_with_explorations(db: AsyncSession, job_id: UUID) -> Job | None:
        try:
            stmt = (
                select(Job)
                .where(Job.job_id == job_id)
                .options(joinedload(Job.batch_sizes).joinedload(BatchSize.explorations))
            )
            return await db.scalar(stmt)
        except Exception as err:
            await db.rollback()
            DBapi._log(f"get_job: {str(err)}")
            raise err

    @staticmethod
    async def get_measurements_of_bs(
        db: AsyncSession, job_id: UUID, window_size: int, bs: int
    ) -> list[Measurement]:
        try:
            if window_size == 0:
                return []
            stmt = (
                select(Measurement)
                .where(and_(Measurement.job_id == job_id, Measurement.batch_size == bs))
                .order_by(Measurement.timestamp.desc())
                .limit(window_size)
            )
            return (await db.scalars(stmt)).all()
        except Exception as err:
            await db.rollback()
            DBapi._log(f"get_measurements_of_bs: {str(err)}")
            raise err

    @staticmethod
    async def delete_job(db: AsyncSession, job: JobSpec):
        pass

    @staticmethod
    def add_exploration(
        db: AsyncSession,
        job_id: UUID,
        batch_size: int,
        trial_number: int,
        state: ExplorationState.State,
    ) -> None:
        exp = ExplorationState(
            job_id=job_id,
            batch_size=batch_size,
            trial_number=trial_number,
            state=state,
        )
        db.add(exp)

    @staticmethod
    async def update_exploration(
        db: AsyncSession,
        job_id: UUID,
        batch_size: int,
        trial_number: int,
        state: ExplorationState.State,
        cost: float,
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
        except Exception as err:
            await db.rollback()
            DBapi._log(f"update_exploration: {str(err)}")
            raise err

    @staticmethod
    def add_arms(db: AsyncSession, arms: list[GaussianTsArmState]) -> None:
        db.add_all(arms)

    @staticmethod
    def add_measurement(
        db: AsyncSession, result: TrainingResult, converged: bool
    ) -> None:
        db.add(
            Measurement(
                job_id=result.job_id,
                batch_size=result.batch_size,
                time=result.time,
                energy=result.energy,
                converged=converged,
            )
        )

    @staticmethod
    async def update_exp_default_bs(
        db: AsyncSession, job_id: UUID, exp_default_bs: int
    ) -> None:
        try:
            stmt = (
                update(Job)
                .where(Job.job_id == job_id)
                .values(exp_default_batch_size=exp_default_bs)
            )
            await db.execute(stmt)
        except Exception as err:
            await db.rollback()
            DBapi._log(f"update_exp_default_bs: {str(err)}")
            raise err

    @staticmethod
    async def update_arm_state(db: AsyncSession, arm: GaussianTsArmState) -> None:
        try:
            stmt = (
                update(GaussianTsArmState)
                .where(
                    and_(
                        GaussianTsArmState.job_id == arm.job_id,
                        GaussianTsArmState.batch_size == arm.batch_size,
                    )
                )
                .values(
                    num_observations=arm.num_observations,
                    param_mean=arm.param_mean,
                    param_precision=arm.param_precision,
                    reward_precision=arm.reward_precision,
                )
            )
            await db.execute(stmt)
        except Exception as err:
            await db.rollback()
            DBapi._log(f"update_exp_default_bs: {str(err)}")
            raise err

    @staticmethod
    def _log(message: str):
        print(f"[DBapi] {message}")
