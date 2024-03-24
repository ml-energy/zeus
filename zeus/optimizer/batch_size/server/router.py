"""Zeus batch size optimizer server FAST API router."""

import asyncio
from collections import defaultdict
import logging

from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobConfig,
    PredictResponse,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.config import settings
from zeus.optimizer.batch_size.server.database.db_connection import get_db_session
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOServerBaseError
from zeus.optimizer.batch_size.server.optimizer import ZeusBatchSizeOptimizer
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.util.logging import get_logger

app = FastAPI()
# Global variable across different requests: https://github.com/tiangolo/fastapi/issues/592
# We lock the job before we make any modification to prevent any concurrent bugs.
app.job_locks = defaultdict(asyncio.Lock)

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
    response_model=JobConfig,
)
async def register_job(
    job: JobConfig,
    response: Response,
    db_session: AsyncSession = Depends(get_db_session),
) -> JobConfig:
    """Endpoint for users to register a job or check if the job is registered and configuration is identical."""
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


# TODO: Add delete


@app.get(GET_NEXT_BATCH_SIZE_URL, response_model=PredictResponse)
async def predict(
    job_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> PredictResponse:
    """Endpoint for users to receive a batch size."""
    async with app.job_locks[job_id]:
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
) -> ReportResponse:
    """Endpoint for users to report the training result."""
    async with app.job_locks[result.job_id]:
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
