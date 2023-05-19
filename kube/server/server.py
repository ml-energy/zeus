
from __future__ import annotations

import asyncio
from typing import Union

from models import (
    JobSpec,
    JobInfo,
    TrialInfo,
    TrialResult,
    ProfilingInfo,
    ProfilingResult,
)
from dbapis import DBAPI

GLOBAL_ZEUS_SERVER: ZeusServer | None = None

class ZeusServer:
    """A singleton class that manages training jobs."""

    def __init__(
        self,
        setting,
    ) -> None:
        """Initialize the server."""
        # Each job will have a `Task` running that serves MAB.
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._job_batch_size_channels: dict[str, asyncio.Queue[int]] = {}
        self._job_results_channels: dict[str, asyncio.Queue[TrialResult]] = {}
        self._job_profiling_channels: dict[str, asyncio.Queue[ProfilingResult]] = {}

    async def register_job(
        self,
        job: JobSpec,
    ) -> None:
        """Register a user-submitted job."""
        # TODO: Append the job to Jobs table by POST to DBServer
        await DBAPI.insert_job(job)

        # Run the job
        await self.run_job(job)

    async def report_trial_result(
        self,
        result: TrialResult,
    ) -> None:
        """Update the completed Trial record in Trials table."""
        self._job_results_channels[result.job_id].put_nowait(result)

    async def report_profiling_result(
        self,
        result: ProfilingResult,
    ) -> None:
        """Update the completed Profiling record in Power table."""
        self._job_profiling_channels[result.job_id].put_nowait(result)

    async def get_job_info(
        self,
        job_id: str,
    ) -> JobInfo:
        """Get the job info."""
        return DBAPI.get_job_info(job_id)
    
    async def get_trial_info(
        self,
        job_id: str,
        rec_i: Union[int, None] = None,
        trial_i: Union[int, None] = None,
    ) -> TrialInfo:
        """Get the trial info."""
        return DBAPI.get_trial_info(job_id, rec_i, trial_i)
    
    async def get_profiling_info(
        self,
        job_id: str,
    ) -> ProfilingInfo:
        """Get the profiling info."""
        return DBAPI.get_profiling_info(job_id)
    
    async def run_job(
        self,
        job: JobSpec,
    ) -> None:
        """Run a job."""
        # TODO: Looping to serves batch size optimizer, wait for profiling/trial results.
        pass


def init_global_zeus_server(setting) -> ZeusServer:
    """Initialize the global singleton `ZeusServer`."""
    global GLOBAL_ZEUS_SERVER
    GLOBAL_ZEUS_SERVER = ZeusServer(setting)
    return GLOBAL_ZEUS_SERVER

def get_global_zeus_server() -> ZeusServer:
    """Fetch the global singleton `ZeusServer`."""
    return GLOBAL_ZEUS_SERVER
