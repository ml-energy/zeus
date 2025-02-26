"""Shared model definitions for the server and client."""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, BaseSettings, Field, validator, PyObject


class PerseusSettings(BaseSettings):
    """Perseus settings, configurable via environment variables.

    For instance, setting `PERSEUS_SCHEDULER=AllMaxFrequency` when launching the
    server will automatically import `perseus.server.scheduler.AllMaxFrequency` and
    the `scheduler` variable will hold it.

    Attributes:
        scheduler: Name of the `PowerStateScheduler` to use.
        scheduler_args: Any extra arguments required by `scheduler.__init__`.
        log_level: Log level, e.g. "debug", "info".
        mode: Which type of power knob the scheduler controls.
        dump_data: Whether the scheduler should dump internal state to the filesystem
            (for future inspection purposes).
        dump_dir: Directory to dump state in (if enabled)
    """

    scheduler: PyObject = "AllMaxFrequency"  # type: ignore
    # TODO(Jae-Won): Check whether the scheduler type accepts the keys in `scheduler_args`.
    scheduler_args: dict[str, Any] = {}
    log_level: str = "DEBUG"
    mode: Literal["power limit", "frequency", "unspecified"] = "unspecified"
    dump_data: bool = True
    dump_dir: str = "dump"

    @validator("scheduler", pre=True)
    def _fix_scheduler_import_path(self, value):
        """Prepend `perseus.server.scheduler.` to the scheduler type name."""
        return f"perseus.server.scheduler.{value}"

    @validator("log_level")
    def _make_upper_case(self, value):
        return value.upper()

    @validator("mode")
    def _set_mode(self, mode, values):
        """Return the power control mode of the scheduler."""
        try:
            scheduler = values["scheduler"]
        except KeyError:
            raise ValueError(
                "Scheduler not set. Please make sure the environment variable "
                "`PERSEUS_SCHEDULER` is a valid scheduler in `perseus.server.scheduler`, "
                "e.g., `AllMaxFrequency`, `InstructionProfiler`.",
            ) from None

        # If the scheduler doesn't have a `mode` set, it means it supports both.
        # Then, the user must specify which mode to user through the `PERSEUS_MODE` env var.
        if not hasattr(scheduler, "mode"):
            if mode == "unspecified":
                raise ValueError("`PERSEUS_MODE` should be specified.")
            scheduler.mode = mode

        # If the scheduler supports only one mode, it's fine if `PERSEUS_MODE` is unspecified.
        # However, if `PERSEUS_MODE` is specified, it should not conflict with the scheduler.
        elif mode not in ["unspecified", scheduler.mode]:
            raise ValueError(
                f"`PERSEUS_MODE={mode}` and scheduler mode ({scheduler.mode}) conflicting.",
            )

        return scheduler.mode

    class Config:
        """Configuration class read by pydantic."""

        env_prefix = "perseus_"


class ServerInfo(BaseModel):
    """Information about the running Perseus server.

    See `PerseusSettings` for details on attributes.
    """

    scheduler: str
    scheduler_args: dict[str, Any]
    log_level: str
    mode: Literal["power limit", "frequency"]
    dump_data: bool
    dump_dir: str

    @validator("scheduler", pre=True)
    def _convert_class_to_str(self, value):
        if not isinstance(value, str):
            return value.__name__
        return value


class PipeInstruction(str, Enum):
    """Atomic operations in pipeline schedules."""

    LOAD = "load"
    FORWARD = "forward"
    BACKWARD = "backward"
    P2P = "p2p"
    CC = "cc"
    STEP = "step"
    LOW = "low"
    HIGH = "high"
    OTHER = "other"
    PURE_BACKWARD = "pure_backward"


class JobInfoPerseus(BaseModel):
    """Training job information reported to the server.

    Attributes:
        job_id: Globally unique ID of the training job, generated by the server.
        pp_degree: Pipeline parallel degree.
        dp_degree: Data parallel degree.
        tp_degree: Tensor parallel degree.
        world_size: World size of the training job.
        framework: Name of the DNN training farmework.
        model_name: Name of the DNN model being trained.
        partition_method: Name of the pipeline stage partition method used.
        microbatch_size: Microbatch size in pipeline parallel training.
        num_microbatches: Number of microbatches in pipeline parallel training.
    """

    job_id: str = ""
    # NOTE: Alpa would not have a fixed dp/pp/tp degree.
    pp_degree: int = Field(ge=1)
    dp_degree: int = Field(ge=1)
    tp_degree: int = Field(ge=1)
    world_size: int = Field(ge=1)
    framework: str
    model_name: str
    partition_method: str
    microbatch_size: int = Field(ge=1)
    num_microbatches: int = Field(ge=1)

    @validator("job_id")
    def _check_empty_job_id(self, job_id):
        assert not job_id
        return job_id

    @validator("world_size")
    def _check_world_size(self, world_size, values):
        """Product of PP, DP, and TP degree would be identical to the world size."""
        assert (
            values["pp_degree"] * values["dp_degree"] * values["tp_degree"]
            == world_size
        )
        return world_size

    def set_job_id(self, server_info: ServerInfo):
        """Generate and set the job ID."""
        self.job_id = "+".join(
            [
                datetime.now().strftime("%F-%H-%M-%S"),
                self.framework,
                self.model_name,
                self.partition_method,
                f"dp{self.dp_degree}",
                f"pp{self.pp_degree}",
                f"tp{self.tp_degree}",
                f"mbs{self.microbatch_size}",
                f"nmb{self.num_microbatches}",
                server_info.scheduler,
            ]
        )


class RankInfoPerseus(BaseModel):
    """Information passed to the server from each rank.

    Attributes:
        job_id: Globally unique ID of the training job, generated by the server.
        rank: Global rank of the reporting process.
        dp_rank: Data parallel rank of the reporting procees.
        pp_rank: Pipeline parallel rank of the reporting procees.
        tp_rank: Tensor parallel rank of the reporting procees.
        pipe_schedule: A list of `PipeInstruction`s for this rank.
        power_state_range: A list of power states to explore.
            Currently this be identical across all ranks of the job.
    """

    job_id: str
    rank: int = Field(ge=0)
    # NOTE: Alpa would not have a fixed dp/pp/tp rank.
    dp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    pipe_schedule: list[PipeInstruction]
    power_state_range: list[int]

    @validator("power_state_range")
    def _validate_power_state_and_sort(self, value):
        if any(ps <= 0 for ps in value):
            raise ValueError("Power state values must be positive integers.")
        if len(value) != len(set(value)):
            raise ValueError("List of power states must be unique.")
        return sorted(value, reverse=True)


class PowerStateSchedule(BaseModel):
    """Power state assignment for each `PipeInstruction` of a rank.

    `power_states` is ordered so that each number corresponds to an instruction in
    `RankInfo.pipe_schedule` in the same order.
    """

    rank: int = Field(ge=0)
    power_states: list[int]


class ProfilingResultPerseus(BaseModel):
    """Profiling results for a `PowerStateSchedule` of a rank.

    Attributes:
        rank: Global rank of the reporting client.
        iter_time: List of latency of all iterations within the profiling window in seconds.
        iter_energy: List of energy consumption of all iterations within the profiling window in Joules.
        time_breakdown: Duration of each `PipeInstruction` across multiple iterations.
            e.g. `time_breakdown[PipeInstruction.FORWARD][i]` is the list of latencies
            of all forward computations in the `i`th iteration.
        energy_breakdown: Energy consumption of each `PipeInstruction` across multiple
            iterations. Value has the same structure as `time_breakdown`.
    """

    rank: int = Field(ge=0)
    iter_time: list[float]
    iter_energy: list[float]
    time_breakdown: dict[PipeInstruction, list[list[float]]] = {}
    energy_breakdown: dict[PipeInstruction, list[list[float]]] = {}


class OfflineProfilingResult(BaseModel):
    """Profiling results generated from offline profiling each instruction.

    Attributes:
        rank: Global rank of the reporting client.
        dp_rank: Data parallel rank of the reporting procees.
        pp_rank: Pipeline parallel rank of the reporting procees.
        tp_rank: Tensor parallel rank of the reporting procees.
        forward_time: Dict that maps power state to average forward computation time.
        forward_energy: Dict that maps power state to average forward computation energy.
        backward_time: Dict that maps power state to average backward computation time.
        backward_energy: Dict that maps power state to average backward computation energy.
    """

    rank: int = Field(ge=0)
    dp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    forward_time: dict[int, float]
    forward_energy: dict[int, float]
    backward_time: dict[int, float]
    backward_energy: dict[int, float]


class InstructionProfilingResult(BaseModel):
    """Time and energy profiling results for each instruction in each stage."""

    __root__: list[OfflineProfilingResult]

    def to_csv(self, filepath: str) -> None:
        """Serialize and save this object into a CSV file.

        Columns: rank, dp_rank, pp_rank, tp_rank, stage, instruction, frequency, time, energy
        Notes
            - `rank` is the global rank of the process.
            - `pp_rank` and `stage` are always the same, for backwards compatibility.
            - All ranks and `stage` are zero-indexed.
            - `instruction` is either "forward" or "backward".
            - `time` and `energy` are already averaged over profiling iterations.
        """
        if not filepath.endswith(".csv"):
            raise ValueError("Filepath does not end with '.csv'")

        # fmt: off
        headers = ["rank", "dp_rank", "pp_rank", "tp_rank", "stage", "instruction", "frequency", "time", "energy"]
        records: list[tuple[int, int, int, int, int, str, int, float, float]] = []
        for res in self.__root__:
            prefix = (res.rank, res.dp_rank, res.pp_rank, res.tp_rank, res.pp_rank)
            for freq in res.forward_time:
                records.append((*prefix, "forward", freq, res.forward_time[freq], res.forward_energy[freq]))
            for freq in res.backward_time:
                records.append((*prefix, "backward", freq, res.backward_time[freq], res.backward_energy[freq]))
        # fmt: on

        df = pd.DataFrame.from_records(records, columns=headers)
        df.to_csv(filepath, index=False)
