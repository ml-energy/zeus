from typing import Annotated, Callable
from uuid import UUID

from fastapi import Depends, FastAPI, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobSpec,
    ReportResponse,
    TrainingResult,
    ZeusBSOJobSpecMismatch,
    ZeusBSOOperationOrderError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.database.db_connection import get_db_session
from zeus.optimizer.batch_size.server.job.repository import JobStateRepository
from zeus.optimizer.batch_size.server.optimizer import (
    ZeusBatchSizeOptimizer,
    get_global_zeus_batch_size_optimizer,
    init_global_zeus_batch_size_optimizer,
)

app = FastAPI()


@app.on_event("startup")
def startup_hook():
    """Startup hook."""
    print("START UP")
    # TODO: change setting
    init_global_zeus_batch_size_optimizer("setting")


@app.post(
    REGISTER_JOB_URL,
    responses={
        200: {"description": "Job is already registered"},
        201: {"description": "Job is successfully registered"},
    },
)
async def register_job(
    job: JobSpec,
    response: Response,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
    db_session: AsyncSession = Depends(get_db_session),
) -> JobSpec:
    """Endpoint for users to register a new job and receive batch size."""
    try:
        res = await zeus_server.register_job(job, JobStateRepository(db_session))
        await db_session.commit()
        if res:
            response.status_code = status.HTTP_201_CREATED
        else:
            response.status_code = status.HTTP_200_OK
        return job
    except ZeusBSOJobSpecMismatch as err:
        return JSONResponse(
            status_code=409,
            content={"message": err.message},
        )
    except (ZeusBSOValueError, ZeusBSOOperationOrderError) as err:
        return JSONResponse(
            status_code=400,
            content={"message": err.message},
        )
    except Exception as err:
        await db_session.rollback()
        print(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )


# TODO: Add delete


@app.get(GET_NEXT_BATCH_SIZE_URL)
async def predict(
    job_id: UUID,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
    db_session: AsyncSession = Depends(get_db_session),
) -> int:
    """Endpoint for users to register a new job and receive batch size."""
    try:
        res = await zeus_server.predict(db_session, job_id)
        await db_session.commit()
        return res
    except (ZeusBSOValueError, ZeusBSOOperationOrderError) as err:
        await db_session.rollback()
        return JSONResponse(
            status_code=400,
            content={"message": err.message},
        )
    except Exception as err:
        await db_session.rollback()
        print(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )


@app.post(REPORT_RESULT_URL, response_model=ReportResponse)
async def report(
    result: TrainingResult,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
    db_session: AsyncSession = Depends(get_db_session),
) -> ReportResponse:
    """Endpoint for users to register a new job and receive batch size."""
    try:
        res = await zeus_server.report(db_session, result)
        await db_session.commit()
        return res
    except (ZeusBSOValueError, ZeusBSOOperationOrderError) as err:
        await db_session.rollback()
        return JSONResponse(
            status_code=400,
            content={"message": err.message},
        )
    except Exception as err:
        await db_session.rollback()
        print(f"Commit Failed: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": str(err)},
        )


# TODO: Just for testing. Erase in the future
@app.get("/test")
async def test(
    job_id: UUID,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
    db_session: AsyncSession = Depends(get_db_session),
) -> None:
    """Endpoint for users to register a new job and receive batch size."""
    return await zeus_server.test(db_session, job_id)
