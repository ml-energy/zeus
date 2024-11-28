"""Interfaces for defining frequency schedulers."""

from __future__ import annotations

import copy
from pathlib import Path
from contextlib import suppress
from abc import ABC, abstractmethod
from typing import Callable, Generator, Sequence, Type

from zeus.optimizer.pipeline_frequency.common import (
    PFOServerSettings,
    JobInfo,
    RankInfo,
    FrequencySchedule,
    ProfilingResult,
)
from zeus.utils.logging import get_logger

logger = get_logger(__name__)


class FrequencyScheduler(ABC):
    """Interface for classes that enclose frequency scheduling policies."""

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        pfo_settings: PFOServerSettings,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            pfo_settings: PFOServerSettings object.
        """
        self.job_info = job_info
        self.rank_infos = sorted(rank_infos, key=lambda info: info.rank)
        self.world_size = self.job_info.world_size
        self.pfo_settings = pfo_settings

        self._generator = self._run()
        self._next_schedule: list[FrequencySchedule] | None = None

    def observe(self, profiling_results: list[ProfilingResult]) -> None:
        """Ingest the profiling results for the previous schedule.

        Args:
            profiling_results: Doesn't have to be sorted in rank order.
        """
        # When there are no more schedules left to yield, the generator will
        # raise `StopIteration`. We just ignore this, and later invocations of
        # `next_schedule()` will return the last schedule returned forever.
        with suppress(StopIteration):
            self._next_schedule = self._generator.send(profiling_results)

    def next_schedule(self) -> list[FrequencySchedule]:
        """Return the schedules for the next round of iterations.

        Returns:
            A list of `FrequencySchedule`s. May not be sorted in rank order.
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
    def _run(self) -> Generator[list[FrequencySchedule], list[ProfilingResult], None]:
        """Yield next schedules and receives profiling results in one place.

        This is an alternative way to write a frequency scheduler. The advantage is
        that everything is enclosed inside this method. The downside is that you'll
        have to read this and understand how this generator works.

        The following implementation is a simple example of writing a scheduler using
        this class. `yield` the next frequency schedule, and receive the profiling
        results corresponding to that schedule from the `yield`. `observe` and
        `next_schedule` will run the generator for you.

        In general, this generator should be designed to `yield` schedules infinitely.
        However, if this was written to write a finite number of next schedules and
        raise `StopIteration`, the last schedule cached inside `self._next_schedule`
        will infinitely be returned from the call to `next_schedule`. This can be
        useful when you converge to the optimal schedule and stop the generator, and
        the rest of training will run with the final optimal schedule indefinitely.
        """
        # This is an example implementation.
        while True:
            # Generate the next frequency schedule
            next_schedule: list[FrequencySchedule] = []
            # Send the next schedule to client and receive the profiling result from client
            profiling_results = yield next_schedule
            # Ingest the profiling result
            logger.debug("%s", profiling_results)


def make_3d_parallel(
    sched_cls: Type[FrequencyScheduler], name: str | None = None
) -> Type[FrequencyScheduler]:
    """Wrap `sched_cls` so that it is aware of 3D parallelism.

    Internally, this function subclasses `sched_cls` and overrides `observe` and
    `next_schedule`. `observe` will aggregate the profiling results from all ranks
    that share the same pp_rank and feed it to `super().observe`, while `next_schedule`
    will first retrieve the per-stage schedule from `super().next_schedule` and then
    copy-paste it to all ranks that share the same pp_rank. With this, the wrapped
    scheduler can operate under the illusion that it's only deadling with pure pipeline
    parallelism.

    Args:
        sched_cls: The scheduler class to wrap.
        name: Name of the scheduler. If None, use `sched_cls.__name__ + "3D"`.
    """

    class Wrapper(sched_cls):  # type: ignore[valid-type,misc]
        def __init__(
            self,
            job_info: JobInfo,
            rank_infos: list[RankInfo],
            pfo_settings: PFOServerSettings,
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

            super().__init__(job_info, rank_infos, pfo_settings, *args, **kwargs)

        def observe(self, profiling_results: list[ProfilingResult]) -> None:
            """Aggregate results so that each pipeline stage has one result."""
            # Aggregate results from ranks that share the same pp_rank.
            rank_to_pp_rank = {
                rank_info.rank: rank_info.pp_rank for rank_info in self._orig_rank_infos
            }
            pp_results: list[list[ProfilingResult]] = [
                [] for _ in range(self._orig_job_info.pp_degree)
            ]
            for result in profiling_results:
                pp_results[rank_to_pp_rank[result.rank]].append(result)

            # For each stage, construct a new ProfilingResult that aggregates all ranks.
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
                agg_result = ProfilingResult(
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
                logger.debug(
                    "Aggregated rank %s results for pp_rank %d: %s",
                    ", ".join([str(r.rank) for r in results]),
                    pp_rank,
                    agg_result,
                )

            # Finally, let the wrapped scheduler observe the aggregated results.
            super().observe(agg_results)

        def next_schedule(self) -> list[FrequencySchedule]:
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
                logger.debug(
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


class PointSolution(FrequencyScheduler):
    """Runs the given frequency schedule."""

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        pfo_settings: PFOServerSettings,
        solution_path: str,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            pfo_settings: PFOServerSettings object.
            solution_path: Path to the frequency Python file generated by lowtime.
        """
        super().__init__(job_info, rank_infos, pfo_settings)

        self.solution_path = Path(solution_path)
        if not self.solution_path.is_file():
            raise RuntimeError(f"Solution file not found: {solution_path}")
        if self.solution_path.suffix != ".py":
            raise RuntimeError(f"Solution file is not a Python file: {solution_path}")

        with open(self.solution_path, encoding="utf-8") as f:
            schedule: list[list[tuple[str, int]]] = eval(f.read())
            if len(schedule) != self.world_size:
                raise RuntimeError(
                    f"Solution file assumes {len(schedule)} ranks, but "
                    f"the job has {self.world_size} ranks."
                )

            self.schedule = []
            for rank, freqs in enumerate(schedule):
                self.schedule.append(FrequencySchedule(rank=rank, frequencies=freqs))

    def _run(self) -> Generator[list[FrequencySchedule], list[ProfilingResult], None]:
        """Yield the schedule given by the solution path."""
        yield self.schedule


PointSolution3D = make_3d_parallel(PointSolution)
