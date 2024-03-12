import logging
from typing import Annotated, Callable
from uuid import UUID

from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.exception import ZeusBaseError
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobSpec,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.config import ZeusBsoSettings
from zeus.util.logging import get_logger

from zeus.optimizer.batch_size.server.database.db_connection import get_db_session
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOServerBaseError
from zeus.optimizer.batch_size.server.optimizer import ZeusBatchSizeOptimizer
from zeus.optimizer.batch_size.server.services.service import ZeusService

app = FastAPI()

settings = ZeusBsoSettings()
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
    response_model=JobSpec,
)
async def register_job(
    job: JobSpec,
    response: Response,
    db_session: AsyncSession = Depends(get_db_session),
) -> JobSpec:
    """Endpoint for users to register a new job and receive batch size."""
    optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
    try:
        res = await optimizer.register_job(job)
        await db_session.commit()
        if res:
            response.status_code = status.HTTP_201_CREATED
        else:
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
        logger.error(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )


# TODO: Add delete


@app.get(GET_NEXT_BATCH_SIZE_URL)
async def predict(
    job_id: UUID,
    db_session: AsyncSession = Depends(get_db_session),
) -> int:
    """Endpoint for users to register a new job and receive batch size."""
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
        logger.error(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )


@app.post(REPORT_RESULT_URL, response_model=ReportResponse)
async def report(
    result: TrainingResult,
    db_session: AsyncSession = Depends(get_db_session),
) -> ReportResponse:
    """Endpoint for users to register a new job and receive batch size."""
    optimizer = ZeusBatchSizeOptimizer(ZeusService(db_session))
    try:
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
        logger.error(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )