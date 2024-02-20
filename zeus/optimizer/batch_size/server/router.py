from uuid import UUID

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobSpec,
    ReportResponse,
    TrainingResult,
    ZeusBSOJobSpecMismatch,
    ZeusBSOValueError,
)
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
    init_global_zeus_batch_size_optimizer("setting")


@app.exception_handler(ZeusBSOJobSpecMismatch)
async def conflict_err_handler(request: Request, exc: ZeusBSOJobSpecMismatch):
    return JSONResponse(
        status_code=409,
        content={"message": exc.message},
    )


@app.exception_handler(ZeusBSOValueError)
async def value_err_handler(request: Request, exc: ZeusBSOValueError):
    return JSONResponse(
        status_code=400,
        content={"message": exc.message},
    )


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
) -> JobSpec:
    """Endpoint for users to register a new job and receive batch size."""
    if zeus_server.register_job(job):
        response.status_code = status.HTTP_201_CREATED
    else:
        response.status_code = status.HTTP_200_OK
    return job


# TODO: Add delete


@app.get(GET_NEXT_BATCH_SIZE_URL)
async def predict(
    job_id: UUID,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
) -> int:
    """Endpoint for users to register a new job and receive batch size."""
    return zeus_server.predict(job_id)


@app.post(REPORT_RESULT_URL)
async def report(
    result: TrainingResult,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_batch_size_optimizer),
) -> ReportResponse:
    """Endpoint for users to register a new job and receive batch size."""
    print(result)
    return zeus_server.report(result)
