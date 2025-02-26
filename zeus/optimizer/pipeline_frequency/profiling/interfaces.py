"""Interfaces for defining power state schedulers."""

from __future__ import annotations

import copy
from contextlib import suppress
from abc import ABC, abstractmethod
from typing import Callable, Literal, Generator, Sequence, Type

from zeus.utils.logging import get_logger

from zeus.optimizer.pipeline_frequency.profiling.models import (
    JobInfoPerseus,
    RankInfoPerseus,
    ProfilingResultPerseus,
    PowerStateSchedule,
    PerseusSettings,
    PipeInstruction,
)

logger = get_logger(__name__)


"""Interfaces for defining power state schedulers."""


class PowerStateScheduler(ABC):
    """Interface for classes that enclose power state scheduling policies."""

    mode: Literal["frequency", "power limit"]

    def __init__(
        self,
        job_info: JobInfoPerseus,
        rank_infos: list[RankInfoPerseus],
        perseus_settings: PerseusSettings,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
        """
        self.job_info = job_info
        self.rank_infos = sorted(rank_infos, key=lambda info: info.rank)
        # Currently we assume that all ranks have the same power states available.
        self.power_state_range = self.rank_infos[0].power_state_range
        self.world_size = self.job_info.world_size
        self.perseus_settings = perseus_settings

    def _config_to_schedule(self, config: list[list[int]]) -> list[PowerStateSchedule]:
        schedules = []
        for stage_id in range(self.job_info.pp_degree):
            schedule = []
            config_iter = iter(config[stage_id])
            power_state = config[stage_id][0]
            num_compute_insts = 0
            for inst in self.rank_infos[stage_id].pipe_schedule:
                if inst in [PipeInstruction.FORWARD, PipeInstruction.BACKWARD]:
                    power_state = next(config_iter)
                    num_compute_insts += 1
                schedule.append(power_state)
            if num_compute_insts != len(config[stage_id]):
                raise RuntimeError(
                    f"Number of compute instructions in the pipe schedule ({num_compute_insts}) "
                    f"does not match the number of power states in the config ({len(config[stage_id])})",
                )
            schedules.append(PowerStateSchedule(rank=stage_id, power_states=schedule))
        return schedules

    @abstractmethod
    def observe(self, profiling_results: list[ProfilingResultPerseus]) -> None:
        """Ingest the profiling results for the previous schedule.

        Args:
            profiling_results: Doesn't have to be sorted in rank order.
        """

    @abstractmethod
    def next_schedule(self) -> list[PowerStateSchedule]:
        """Return the schedules for the next round of iterations.

        Returns:
            A list of `PowerStateSchedule`s. May not be sorted in rank order.
        """


class PowerStateSchedulerV2(PowerStateScheduler):
    """Interface for classes that enclose power state scheduling policies."""

    def __init__(
        self,
        job_info: JobInfoPerseus,
        rank_infos: list[RankInfoPerseus],
        perseus_settings: PerseusSettings,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
        """
        super().__init__(job_info, rank_infos, perseus_settings)
        self._generator = self._run()
        self._next_schedule: list[PowerStateSchedule] | None = None

    def observe(self, profiling_results: list[ProfilingResultPerseus]) -> None:
        """Ingest the profiling results for the previous schedule.

        Args:
            profiling_results: Doesn't have to be sorted in rank order.
        """
        # When there are no more schedules left to yield, the generator will
        # raise `StopIteration`. We just ignore this, and later invocations of
        # `next_schedule()` will return the last schedule returned forever.
        with suppress(StopIteration):
            self._next_schedule = self._generator.send(profiling_results)

    def next_schedule(self) -> list[PowerStateSchedule]:
        """Return the schedules for the next round of iterations.

        Returns:
            A list of `PowerStateSchedule`s. May not be sorted in rank order.
        """
        if self._next_schedule is None:
            try:
                self._next_schedule = next(self._generator)
            except StopIteration as exc:
                raise RuntimeError(
                    "The _run generator raised StopIteration on its first next call.",
                ) from exc
        return self._next_schedule

    @abstractmethod
    def _run(
        self,
    ) -> Generator[list[PowerStateSchedule], list[ProfilingResultPerseus], None]:
        """Yield next schedules and receives profiling results in one place.

        This is an alternative way to write a power state scheduler. The advantage is
        that everything is enclosed inside this method. The downside is that you'll
        have to read this and understand how this generator works.

        The following implementation is a simple example of writing a scheduler using
        this class. `yield` the next power state schedule, and receive the profiling
        results corresponding to that schedule from the `yield`. `observe` and
        `next_schedule` will run the generator for you.

        In general, this generator should be designed to `yield` schedules infinitely.
        However, if this was written to write a finite number of next schedules and
        raise `StopIteration`, the last schedule cached inside `self._next_schedule`
        will infinitely be returned from the call to `next_schedule`. This can be
        useful when you converge to the optimal schedule and stop the generator, and
        the rest of training will run with the final optimal schedule indefinitely.
        """
        while True:
            # Generate the next power state schedule
            next_schedule: list[PowerStateSchedule] = []
            # Send the next schedule to client and receive the profiling result from client
            profiling_results = yield next_schedule
            # Ingest the profiling result
            logger.debug("%s", profiling_results)


class FrequencySchedulerPerseus(PowerStateScheduler):
    """Interface for classes that enclose frequency scheduling policies."""

    mode = "frequency"


class PowerLimitScheduler(PowerStateScheduler):
    """Interface for classes that enclose power limit scheduling policies."""

    mode = "power limit"


def make_3d_parallel_perseus(
    sched_cls: Type[PowerStateScheduler], name: str | None = None
) -> Type[PowerStateScheduler]:
    """Wrap `sched_cls` so that it is aware of 3D parallelism.

    Internally, this function subclasses `sched_cls` and overrides `observe` and
    `next_schedule`. `observe` will aggregate the profiling results from all ranks
    that share the same pp_rank and feed it to `super().observe`, while `next_schedule`
    will first retrieve the per-stage schedule from `super().next_schedule` and then
    copy-paste it to all ranks that share the same pp_rank. With this, the wrapped
    scheduler can operate under the illusion that it's only dealing with pure pipeline
    parallelism.

    Args:
        sched_cls: The scheduler class to wrap.
        name: Name of the scheduler. If None, use `sched_cls.__name__ + "3D"`.
    """

    class Wrapper(sched_cls):  # type: ignore[valid-type,misc]
        def __init__(
            self,
            job_info: JobInfoPerseus,
            rank_infos: list[RankInfoPerseus],
            perseus_settings: PerseusSettings,
            *args,
            **kwargs,
        ) -> None:
            self._orig_job_info = job_info
            self._orig_rank_infos = rank_infos

            # Give the wrapped scheduler a perfect illusion of pure pipeline parallelism
            # and no data or tensor parallelism. New rank is given by pp_rank.
            job_info = copy.deepcopy(job_info)
            job_info.dp_degree = 1
            job_info.tp_degree = 1
            job_info.world_size = job_info.pp_degree

            new_rank_infos = []
            for rank_info in rank_infos:
                if rank_info.dp_rank == 0 and rank_info.tp_rank == 0:
                    new_rank_info = copy.deepcopy(rank_info)
                    new_rank_info.rank = rank_info.pp_rank
                    new_rank_infos.append(new_rank_info)

            super().__init__(job_info, rank_infos, perseus_settings, *args, **kwargs)

        def observe(self, profiling_results: list[ProfilingResultPerseus]) -> None:
            """Aggregate results so that each pipeline stage has one result."""
            # Aggregate results from ranks that share the same pp_rank.
            rank_to_pp_rank = {
                rank_info.rank: rank_info.pp_rank for rank_info in self._orig_rank_infos
            }
            pp_results: list[list[ProfilingResultPerseus]] = [
                [] for _ in range(self._orig_job_info.pp_degree)
            ]
            for result in profiling_results:
                pp_results[rank_to_pp_rank[result.rank]].append(result)

            # For each stage, construct a new ProfilingResultPerseus that aggregates all ranks.
            # For iter_time and values in time_breakdown, take the max.
            # For iter_energy and values in energy_breakdown, take the sum.
            def agg_list(values: Sequence[list[float]], fun: Callable) -> list[float]:
                return [fun(vals) for vals in zip(*values)]

            def agg_list_of_list(
                values: Sequence[list[list[float]]], fun: Callable
            ) -> list[list[float]]:
                return [agg_list(vals, fun) for vals in zip(*values)]

            agg_results = []
            for pp_rank, results in enumerate(pp_results):
                agg_result = ProfilingResultPerseus(
                    rank=pp_rank,
                    iter_time=agg_list([result.iter_time for result in results], max),
                    iter_energy=agg_list(
                        [result.iter_energy for result in results], sum
                    ),
                    time_breakdown={
                        key: agg_list_of_list(
                            [result.time_breakdown[key] for result in results], max
                        )
                        for key in results[0].time_breakdown
                    },
                    energy_breakdown={
                        key: agg_list_of_list(
                            [result.energy_breakdown[key] for result in results], sum
                        )
                        for key in results[0].energy_breakdown
                    },
                )
                agg_results.append(agg_result)
                logger.info(
                    "Aggregated rank %s results for pp_rank %d: %s",
                    ", ".join([str(r.rank) for r in results]),
                    pp_rank,
                    agg_result,
                )

            # Finally, let the wrapped scheduler observe the aggregated results.
            super().observe(agg_results)

        def next_schedule(self) -> list[PowerStateSchedule]:
            """Copy and paste the schedule for each stage to all ranks in that stage."""
            # Retrive the next schedule for each stage.
            schedules = super().next_schedule()

            # Copy and paste the schedule for each stage to all ranks in that stage.
            rank_to_pp_rank = {
                rank_info.rank: rank_info.pp_rank for rank_info in self._orig_rank_infos
            }
            next_schedule = []
            for rank in range(self._orig_job_info.world_size):
                pp_rank = rank_to_pp_rank[rank]
                sched = copy.deepcopy(schedules[pp_rank])
                sched.rank = rank
                next_schedule.append(sched)
                logger.info(
                    "Copied schedule for pp_rank %d to rank %d: %s",
                    pp_rank,
                    rank,
                    sched,
                )
            return next_schedule

    Wrapper.__name__ = name or (sched_cls.__name__ + "3D")
    if sched_cls.__doc__ is not None:
        Wrapper.__doc__ = "[Wrapped for 3D parallelism] " + sched_cls.__doc__

    return Wrapper
