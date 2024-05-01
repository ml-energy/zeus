"""Zeus batch size optimizer server FAST API router."""

import asyncio
import logging
from collections import defaultdict

from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.common import (
    DELETE_JOB_URL,
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_END_URL,
    REPORT_RESULT_URL,
    CreatedJob,
    JobSpecFromClient,
    TrialId,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.config import settings
from zeus.optimizer.batch_size.server.database.db_connection import get_db_session
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServerBaseError,
)
from zeus.optimizer.batch_size.server.optimizer import ZeusBatchSizeOptimizer
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.logging import get_logger

app = FastAPI()
# Global variable across different requests: https://github.com/tiangolo/fastapi/issues/592
# We lock the job before we make any modification to prevent any concurrent bugs.
job_locks = defaultdict(asyncio.Lock)
prefix_locks = defaultdict(asyncio.Lock)

def get_job_locks() -> defaultdict[str, asyncio.Lock]:
    global job_locks
    return job_locks


def get_prefix_locks() -> defaultdict[str, asyncio.Lock]:
    global prefix_locks
    return prefix_locks


logger = get_logger(__name__)
logging.basicConfig(level=logging.getLevelName(settings.log_level))


@app.on_event("startup")
def startup_hook():
    """Startup hook."""
    pass


@app.post(
    REGISTER_JOB_URL,
    responses={
        200: {"description": "Job is already registered"},
        201: {"description": "Job is successfully registered"},
    },
    response_model=JobSpecFromClient,
)
async def register_job(
    job: JobSpecFromClient,
    response: Response,
    db_session: AsyncSession = Depends(get_db_session),
    prefix_locks: defaultdict[str, asyncio.Lock] = Depends(get_prefix_locks),
):
    """Endpoint for users to register a job or check if the job is registered and configuration is identical."""
    async with prefix_locks[job.job_id_prefix]:
        # One lock for registering a job. To prevent getting a same lock
        optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
        try:
            created = await optimizer.register_job(job)
            await db_session.commit()
            if created:
                # new job is created
                response.status_code = status.HTTP_201_CREATED
            else:
                # job already exists
                response.status_code = status.HTTP_200_OK
            return job
        except ZeusBSOServerBaseError as err:
            await db_session.rollback()
            return JSONResponse(
                status_code=err.status_code,
                content={"message": err.message},
            )
        except Exception as err:
            await db_session.rollback()
            logger.error("Commit Failed: %s", str(err))
            return JSONResponse(
                status_code=500,
                content={"message": str(err)},
            )


@app.delete(DELETE_JOB_URL)
async def delete_job(
    job_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to delete a job."""
    async with job_locks[job_id]:
        try:
            optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
            await optimizer.delete_job(job_id)
            await db_session.commit()
        except ZeusBSOServerBaseError as err:
            await db_session.rollback()
            return JSONResponse(
                status_code=err.status_code,
                content={"message": err.message},
            )
        except Exception as err:
            await db_session.rollback()
            logger.error("Commit Failed: %s", str(err))
            return JSONResponse(
                status_code=500,
                content={"message": str(err)},
            )
        finally:
            job_locks.pop(job_id)


@app.patch(REPORT_END_URL)
async def end_trial(
    trial: TrialId,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to end the trial."""
    async with job_locks[trial.job_id]:
        optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
        try:
            await optimizer.end_trial(trial)
            await db_session.commit()
        except ZeusBSOServerBaseError as err:
            await db_session.rollback()
            return JSONResponse(
                status_code=err.status_code,
                content={"message": err.message},
            )
        except Exception as err:
            await db_session.rollback()
            logger.error("Commit Failed: %s", str(err))
            return JSONResponse(
                status_code=500,
                content={"message": str(err)},
            )


@app.get(GET_NEXT_BATCH_SIZE_URL, response_model=TrialId)
async def predict(
    job_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to receive a batch size."""
    async with job_locks[job_id]:
        optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
        try:
            res = await optimizer.predict(job_id)
            await db_session.commit()
            return res
        except ZeusBSOServerBaseError as err:
            await db_session.rollback()
            return JSONResponse(
                status_code=err.status_code,
                content={"message": err.message},
            )
        except Exception as err:
            await db_session.rollback()
            logger.error("Commit Failed: %s", str(err))
            return JSONResponse(
                status_code=500,
                content={"message": str(err)},
            )


@app.post(REPORT_RESULT_URL, response_model=ReportResponse)
async def report(
    result: TrainingResult,
    db_session: AsyncSession = Depends(get_db_session),
    job_locks: defaultdict[str, asyncio.Lock] = Depends(get_job_locks),
):
    """Endpoint for users to report the training result."""
    async with job_locks[result.job_id]:
        optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
        try:
            logger.info("Report with result %s", str(result))
            res = await optimizer.report(result)
            await db_session.commit()
            return res
        except ZeusBSOServerBaseError as err:
            await db_session.rollback()
            return JSONResponse(
                status_code=err.status_code,
                content={"message": err.message},
            )
        except Exception as err:
            await db_session.rollback()
            logger.error("Commit Failed: %s", str(err))
            return JSONResponse(
                status_code=500,
                content={"message": str(err)},
            )
