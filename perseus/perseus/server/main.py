"""Main Perseus server implementation."""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Depends, FastAPI, Response, Request
from fastapi.logger import logger
from fastapi.routing import APIRoute

from perseus.models import (
    JobInfo,
    PerseusSettings,
    ProfilingResult,
    RankInfo,
    PowerStateSchedule,
    ServerInfo,
)
from perseus.common import (
    GET_SERVER_INFO_URL,
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    GET_POWER_STATE_SCHEDULE_URL,
    REPORT_PROFILING_RESULT_URL,
)
from perseus.server.job import (
    JobManager,
    init_global_job_manager,
    get_global_job_manager,
)

app = FastAPI()


class LoggingRoute(APIRoute):
    """Route handler that logs out all requests and responses in DEBUG level."""

    def get_route_handler(self) -> Callable:
        """Wrap the original handler with debug messages."""
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response: Response = await original_route_handler(request)
            logger.debug(
                "%s %s: %s -> %s",
                request.method,
                request.url,
                await request.json() if await request.body() else "None",
                response.body.decode(response.charset),
            )
            return response

        return custom_route_handler


settings = PerseusSettings()
logging.basicConfig(level=logging.getLevelName(settings.log_level))
if logging.getLevelName(settings.log_level) >= logging.DEBUG:
    app.router.route_class = LoggingRoute
server_info = ServerInfo.parse_obj(settings)


@app.on_event("startup")
async def startup_hook():
    """Startup hook."""
    # pylint: disable=no-member
    logger.info(
        "Using %s scheduler `%s`", settings.scheduler.mode, settings.scheduler.__name__
    )
    init_global_job_manager(settings)


@app.get(GET_SERVER_INFO_URL, response_model=ServerInfo)
async def get_server_info() -> ServerInfo:
    """Return information about the Perseus server."""
    return server_info


@app.post(REGISTER_JOB_URL, response_model=str)
async def register_job(
    job_info: JobInfo, job_manager: JobManager = Depends(get_global_job_manager)
) -> str:
    """Register the training job's information in the server."""
    job_info.set_job_id(server_info)
    job_manager.register_job(job_info)
    return job_info.job_id


@app.post(REGISTER_RANK_URL)
async def register_rank(
    rank_info: RankInfo, job_manager: JobManager = Depends(get_global_job_manager)
) -> None:
    """Register each rank's information in the server."""
    job_manager.register_rank(rank_info)


@app.get(GET_POWER_STATE_SCHEDULE_URL, response_model=PowerStateSchedule)
async def get_power_state_schedule(
    job_id: str,
    rank: int,
    job_manager: JobManager = Depends(get_global_job_manager),
) -> PowerStateSchedule:
    """Return the next power state schedule for the rank."""
    return await job_manager.get_power_state_schedule(job_id, rank)


@app.post(REPORT_PROFILING_RESULT_URL)
async def report_profiling_result(
    job_id: str,
    profiling_result: ProfilingResult,
    job_manager: JobManager = Depends(get_global_job_manager),
) -> None:
    """Report the profiling result for the most recent power state schedule."""
    job_manager.report_profiling_result(job_id, profiling_result)
