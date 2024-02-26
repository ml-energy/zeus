from uuid import UUID
from sqlalchemy import select
import sqlalchemy
from sqlalchemy.orm import session

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError

from zeus.optimizer.batch_size.common import JobSpec, TrainingResult
from zeus.optimizer.batch_size.server.database.models import Job

# https://github.com/ThomasAitken/demo-fastapi-async-sqlalchemy/blob/main/backend/app/crud/user.py


class DBapi(object):

    @staticmethod
    async def create_job(db: AsyncSession, jobSpec: JobSpec) -> Job | None:
        """Create job and return the job if successful. If job already exists, return None"""
        try:
            job_obj = Job()
            job_obj.populate_from_job_spec(jobSpec)
            db.add(job_obj)
            await db.commit()
            return job_obj
        except IntegrityError as err:
            await db.rollback()
            if "UNIQUE constraint failed" in str(err):
                print("[DBAPI] UNIQUE CONSTRAINT FAILED")
                return None
            else:
                raise err
        except Exception as err:
            await db.rollback()
            raise err

    @staticmethod
    async def get_job(db: AsyncSession, job_id: UUID) -> Job | None:
        try:
            return await db.get_one(Job, job_id)
        except NoResultFound:
            await db.rollback()
            return None
        except Exception as err:
            await db.rollback()
            raise err

    @staticmethod
    async def delete_job(db: AsyncSession, job: JobSpec):
        pass

    @staticmethod
    async def create_history(db: AsyncSession, job_id: UUID):
        pass

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
    async def add_measurement(db: AsyncSession):
        pass

    @staticmethod
    async def get_measurement(db: AsyncSession):
        pass
