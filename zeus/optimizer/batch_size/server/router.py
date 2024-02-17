from fastapi import Depends, FastAPI, Response, status
from uuid import uuid4, UUID

from zeus.optimizer.batch_size.server.models import (
    JobSpec,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.optimizer import (
    ZeusBatchSizeOptimizer,
    get_global_zeus_server,
    init_global_zeus_server,
)

app = FastAPI()


@app.on_event("startup")
def startup_hook():
    """Startup hook."""
    print("START UP")
    init_global_zeus_server("setting")


# app.add_event_handler("startup", startup_hook)


@app.post(
    "/jobs",
    responses={
        200: {"description": "Job is already registered"},
        201: {"description": "Job is successfully registered"},
    },
)
async def register_job(
    job: JobSpec,
    response: Response,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_server),
) -> JobSpec:
    """Endpoint for users to register a new job and receive batch size."""
    if zeus_server.register_job(job) == 0:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_201_CREATED
    return job


# TODO: Add delete


@app.get("/jobs/batch_size")
async def predict(
    job_id: UUID,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_server),
) -> int:
    """Endpoint for users to register a new job and receive batch size."""
    return zeus_server.predict(job_id)


@app.post("/jobs/report")
async def report(
    result: TrainingResult,
    zeus_server: ZeusBatchSizeOptimizer = Depends(get_global_zeus_server),
) -> ReportResponse:
    """Endpoint for users to register a new job and receive batch size."""
    print(result)
    return zeus_server.report(result)
