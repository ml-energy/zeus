# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Perseus server FastAPI router."""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Depends, FastAPI, Response, Request
from fastapi.routing import APIRoute

from zeus.util.logging import get_logger
from zeus.optimizer.perseus.common import (
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    GET_FREQUENCY_SCHEDULE_URL,
    REPORT_PROFILING_RESULT_URL,
    JobInfo,
    RankInfo,
    PerseusSettings,
    ProfilingResult,
    FrequencySchedule,
)
from zeus.optimizer.perseus.server.job_manager import (
    JobManager,
    init_global_job_manager,
    get_global_job_manager,
)

logger = get_logger(__name__)
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
if logging.getLevelName(settings.log_level) <= logging.DEBUG:
    app.router.route_class = LoggingRoute


@app.on_event("startup")
async def startup_hook():
    """Startup hook."""
    logger.info("Using scheduler `%s`", settings.scheduler.__name__)
    init_global_job_manager(settings)


@app.post(REGISTER_JOB_URL, response_model=str)
async def register_job(
    job_info: JobInfo, job_manager: JobManager = Depends(get_global_job_manager)
) -> str:
    """Register the training job's information in the server."""
    job_info.set_job_id(scheduler_name=settings.scheduler.__name__)
    job_manager.register_job(job_info)
    return job_info.job_id


@app.post(REGISTER_RANK_URL)
async def register_rank(
    job_id: str,
    rank_info: RankInfo,
    job_manager: JobManager = Depends(get_global_job_manager),
) -> None:
    """Register each rank's information in the server."""
    job_manager.register_rank(job_id, rank_info)


@app.get(GET_FREQUENCY_SCHEDULE_URL, response_model=FrequencySchedule)
async def get_frequency_schedule(
    job_id: str,
    rank: int,
    job_manager: JobManager = Depends(get_global_job_manager),
) -> FrequencySchedule:
    """Return the next frequency schedule for the rank."""
    return await job_manager.get_frequency_schedule(job_id, rank)


@app.post(REPORT_PROFILING_RESULT_URL)
async def report_profiling_result(
    job_id: str,
    profiling_result: ProfilingResult,
    job_manager: JobManager = Depends(get_global_job_manager),
) -> None:
    """Report the profiling result for the most recent frequency schedule."""
    job_manager.report_profiling_result(job_id, profiling_result)
