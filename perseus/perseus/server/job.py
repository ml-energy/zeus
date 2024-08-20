"""Main class that manages the profiling and power state scheduling of jobs."""

from __future__ import annotations

import asyncio
import traceback

from fastapi import HTTPException

from perseus.models import (
    JobInfo,
    PerseusSettings,
    PowerStateSchedule,
    ProfilingResult,
    RankInfo,
    ServerInfo,
)
from perseus.utils.state import (
    save_prof,
    save_sched,
    save_ranks,
    save_info,
)
from perseus.utils.task import create_task

GLOBAL_JOB_MANAGER: JobManager | None = None


class JobManager:
    """A singleton class that manages all states."""

    def __init__(self, perseus_settings: PerseusSettings) -> None:
        """Initialize the job manager."""
        self.perseus_settings = perseus_settings
        self.server_info = ServerInfo.parse_obj(perseus_settings)
        self.scheduler_type = perseus_settings.scheduler
        self.scheduler_args = perseus_settings.scheduler_args
        self.dump_data = perseus_settings.dump_data
        self.dump_dir = perseus_settings.dump_dir

        self._job_infos: dict[str, JobInfo] = {}
        self._job_rank_infos: dict[str, list[RankInfo]] = {}

        # TODO: Have a way to remove jobs (LRU) that haven't been around.
        #       Have a async task that periodically wakes up to evict jobs.
        #       The cleanup task can be cancelled using a shutdown event handler.
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._job_result_channels: dict[str, asyncio.Queue[ProfilingResult]] = {}
        self._job_sched_request_channels: dict[str, asyncio.Queue] = {}
        self._job_sched_response_channels: dict[str, list[asyncio.Queue]] = {}

    def register_job(self, job_info: JobInfo) -> None:
        """Prepare internal state for a new incoming job.

        This method will be called exactly once by the global rank 0 (master) process.
        """
        job_id = job_info.job_id
        world_size = job_info.world_size
        self._job_infos[job_info.job_id] = job_info
        self._job_rank_infos[job_id] = []
        self._job_result_channels[job_id] = asyncio.Queue(maxsize=world_size)
        self._job_sched_request_channels[job_id] = asyncio.Queue(maxsize=world_size)
        self._job_sched_response_channels[job_id] = [
            asyncio.Queue(maxsize=1) for _ in range(world_size)
        ]
        self._job_tasks[job_id] = create_task(self._job_task(job_id, self.dump_data))

    def register_rank(self, rank_info: RankInfo) -> None:
        """Register rank-specific information for an already registered job.

        This method will be called `world_size` number of times (once per rank).
        """
        self._job_rank_infos[rank_info.job_id].append(rank_info)

    async def get_power_state_schedule(
        self, job_id: str, rank: int
    ) -> PowerStateSchedule:
        """Get the next power state schedule for a rank.

        This method will be called `world_size` number of times (once per rank).
        All ranks will block on this method untill everyone reports their
        profiling results and calls this method.

        When an internal scheduler error happened at any point of servicing the
        job, clients will be notified through this API with a 500 Internal Error.
        """
        await self._job_sched_request_channels[job_id].put(rank)
        res = await self._job_sched_response_channels[job_id][rank].get()
        if isinstance(res, Exception):
            code = 400 if isinstance(res, ValueError) else 500
            raise HTTPException(
                status_code=code,
                detail="".join(
                    traceback.format_exception(type(res), res, res.__traceback__)
                ),
            )
        return res

    def report_profiling_result(self, job_id: str, result: ProfilingResult) -> None:
        """Send the profiling result to the job task and immediately return.

        This method will be called `world_size` number of times - one for each rank.
        """
        self._job_result_channels[job_id].put_nowait(result)

    async def _job_task(self, job_id: str, dump_data: bool) -> None:
        """Coalese requests and responses of each rank and interface with the scheduler."""
        result_chan = self._job_result_channels[job_id]
        sched_req_chan = self._job_sched_request_channels[job_id]
        sched_resp_chan = self._job_sched_response_channels[job_id]

        job_info = self._job_infos[job_id]

        try:
            # Wait until all ranks have reported their `RankInfo`s.
            rank_infos = self._job_rank_infos[job_id]
            while True:
                await asyncio.sleep(0.1)
                # Indexing the first element is always safe because this task is
                # created after putting the `RankInfo` of the first-connected rank
                # in `self.job_rank_infos[job_id]`.
                if len(rank_infos) == job_info.world_size:
                    break

            # Sort `RankInfo`s in rank order.
            rank_infos.sort(key=lambda r: r.rank)

            # Create directory to dump Perseus states.
            dump_dir = f"{self.dump_dir}/{job_id}"
            if dump_data:
                await save_info(self.server_info, dump_dir)
                await save_ranks(rank_infos, dump_dir)

            # Instantiate the power state scheduler.
            scheduler = self.scheduler_type(
                job_info,
                rank_infos,
                self.perseus_settings,
                **self.scheduler_args,
            )

            # Provide next schedules, observe profiling results, and repeat.
            schedule_num = 0
            while True:
                # Compute the next `PowerStateSchedule`s.
                schedules = scheduler.next_schedule()

                # Wait until all the ranks ask for the next schedule.
                await asyncio.gather(*[sched_req_chan.get() for _ in rank_infos])

                # Send out `PowerStateSchedule`s.
                await asyncio.gather(
                    *[sched_resp_chan[s.rank].put(s) for s in schedules]
                )

                # Gather profiling results from all ranks.
                results = await asyncio.gather(*[result_chan.get() for _ in rank_infos])
                results.sort(key=lambda r: r.rank)

                # Dump profiling results and schedules.
                if dump_data:
                    schedules.sort(key=lambda s: s.rank)
                    await save_prof(results, dump_dir, schedule_num)
                    await save_sched(schedules, dump_dir, schedule_num)

                # Send `ProfilingResult`s to the scheduler.
                scheduler.observe(results)

                # Increment schedule number.
                schedule_num += 1

        except Exception as exc:  # pylint: disable=broad-except
            # In case the scheduler errored, send out the exception to the clients.
            # The clients will receive the error when they ask for the next schedule.
            for chan in sched_resp_chan:
                chan.put_nowait(exc)
            raise


def init_global_job_manager(perseus_settings: PerseusSettings) -> None:
    """Instantiate the global singleton `JobManager`."""
    global GLOBAL_JOB_MANAGER
    GLOBAL_JOB_MANAGER = JobManager(perseus_settings=perseus_settings)


def get_global_job_manager() -> JobManager:
    """Fetch the global singleton `JobManager`."""
    assert GLOBAL_JOB_MANAGER is not None, "`init_global_job_manager` was not called."
    return GLOBAL_JOB_MANAGER
