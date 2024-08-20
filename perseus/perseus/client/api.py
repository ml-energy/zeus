"""Perseus client.

For a training job, each rank will register itself (`init`) to the Perseus server
with its rank information and pipeline training schedule. Then it will query the
server (`get_power_state_schedule`) for the power state schedule to execute. The client
reports back the profiling results (`report_profiling_result`) back to the server
for the most recent power state schedule.
"""

from __future__ import annotations

import httpx
import torch.distributed as dist

from perseus.models import (
    PipeInstruction,
    JobInfo,
    RankInfo,
    PowerStateSchedule,
    ProfilingResult,
    ServerInfo,
)
from perseus.common import (
    GET_SERVER_INFO_URL,
    GET_POWER_STATE_SCHEDULE_URL,
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    REPORT_PROFILING_RESULT_URL,
)

RANK: int | None = None
JOB_ID: str | None = None
SERVER_URL: str | None = None


def init(
    rank: int,
    dp_rank: int,
    pp_rank: int,
    tp_rank: int,
    pp_degree: int,
    dp_degree: int,
    tp_degree: int,
    world_size: int,
    pipe_schedule: list[PipeInstruction],
    server_url: str,
    power_state_range: list[int],
    framework: str,
    model_name: str,
    partition_method: str,
    microbatch_size: int,
    num_microbatches: int,
) -> str:
    """Initialize Perseus by registering information to the server.

    This function should be called after `init_process_group`.

    First, the master (rank 0) process will register job-level information
    at the Perseus server and retrieve the job ID for this training job.
    Then, each rank will register rank-specific information to the server.
    """
    # Check whether `torch.distributed` is initialized
    if not dist.is_initialized():
        raise RuntimeError("Call `torch.distributed.init_process_group` first.")

    # Set global variables that will be used to identify this training job.
    global JOB_ID, RANK, SERVER_URL
    RANK = rank
    SERVER_URL = server_url

    # Construct payload and register the job to the server.
    if rank == 0:
        job_info = JobInfo(
            pp_degree=pp_degree,
            dp_degree=dp_degree,
            tp_degree=tp_degree,
            world_size=world_size,
            framework=framework,
            model_name=model_name,
            partition_method=partition_method,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
        )
        response = httpx.post(f"{SERVER_URL}{REGISTER_JOB_URL}", json=job_info.dict())
        if (code := response.status_code) != 200:
            raise RuntimeError(
                f"Perseus server returned status code {code}: {response.text}"
            )
        JOB_ID = response.json()
        if not isinstance(JOB_ID, str):
            raise RuntimeError(f"Perseus server returned a strange job ID: {JOB_ID=}")

    # Rank 0 broadcasts the job ID across all ranks.
    objects = [JOB_ID]
    dist.broadcast_object_list(objects, src=0)
    JOB_ID = objects[0]
    if JOB_ID is None:
        raise RuntimeError("Failed to broadcast job ID")

    # Construct payload and register this rank to the server.
    rank_info = RankInfo(
        job_id=JOB_ID,
        rank=rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        pipe_schedule=pipe_schedule,
        power_state_range=power_state_range,
    )
    response = httpx.post(f"{SERVER_URL}{REGISTER_RANK_URL}", json=rank_info.dict())
    if (code := response.status_code) != 200:
        raise RuntimeError(
            f"Perseus server returned status code {code}: {response.text}"
        )

    return JOB_ID


def get_server_info(server_url: str) -> ServerInfo:
    """Fetch information about the running Perseus server."""
    response = httpx.get(f"{server_url}{GET_SERVER_INFO_URL}")
    if (code := response.status_code) != 200:
        raise RuntimeError(
            f"Perseus server returned status code {code}: {response.text}"
        )
    return ServerInfo.parse_raw(response.text)


def get_power_state_schedule() -> PowerStateSchedule:
    """Query the server for the power state schedule.

    This can act as a synchronization point before all ranks begin the next
    training iteration, because the server will only return the response once
    all processes have checked in.
    """
    if RANK is None or JOB_ID is None or SERVER_URL is None:
        raise RuntimeError("Call `perseus.client.init` first.")
    response = httpx.get(
        f"{SERVER_URL}{GET_POWER_STATE_SCHEDULE_URL.format(job_id=JOB_ID)}",
        params={"rank": RANK},
        timeout=None,
    )
    if (code := response.status_code) != 200:
        raise RuntimeError(
            f"Perseus server returned status code {code}: {response.text}"
        )
    schedule = PowerStateSchedule.parse_raw(response.text)
    if schedule.rank != RANK:
        raise RuntimeError(
            "Perseus server returned a power state schedule for another rank. "
            "This is a problem in Perseus. Check the scheduler implementation."
        )
    return schedule


def report_profiling_result(
    iter_time: list[float],
    iter_energy: list[float],
    time_breakdown: dict[PipeInstruction, list[list[float]]],
    energy_breakdown: dict[PipeInstruction, list[list[float]]],
) -> None:
    """Report the average iteration time and energy of the previous power state schedule.

    Args:
        iter_time: List of time for all iterations within the profiling window in seconds.
        iter_energy: List of energy consumption for all iterations within the profiling window in Joules.
        time_breakdown: Duration of each `PipeInstruction` across multiple iterations.
            e.g. `time_breakdown[PipeInstruction.FORWARD][i]` is the list of latencies
            of all forward computations in the `i`th iteration.
        energy_breakdown: Energy consumption of each `PipeInstruction` across multiple
            iterations. Value has the same structure as `time_breakdown`.
    """
    if RANK is None or JOB_ID is None or SERVER_URL is None:
        raise RuntimeError("Call `perseus.client.init` first.")
    if len(iter_time) != len(iter_energy):
        raise RuntimeError("Profiling result is invalid.")
    payload = ProfilingResult(
        rank=RANK,
        iter_time=iter_time,
        iter_energy=iter_energy,
        time_breakdown=time_breakdown,
        energy_breakdown=energy_breakdown,
    )
    response = httpx.post(
        f"{SERVER_URL}{REPORT_PROFILING_RESULT_URL.format(job_id=JOB_ID)}",
        json=payload.dict(),
    )
    if (code := response.status_code) != 200:
        raise RuntimeError(
            f"Perseus server returned status code {code}: {response.text}"
        )
