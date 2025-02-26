"""Schedulers that are really profilers."""

from __future__ import annotations

from typing import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.logger import logger

from zeus.optimizer.pipeline_frequency.profiling.models import (
    PerseusSettings,
    JobInfoPerseus,
    RankInfoPerseus,
    PowerStateSchedule,
    ProfilingResultPerseus,
    PipeInstruction,
)
from zeus.optimizer.pipeline_frequency.profiling.interfaces import (
    PowerStateSchedulerV2,
    make_3d_parallel_perseus,
)


class InstructionProfiler(PowerStateSchedulerV2):
    """Profiles the time and energy of each instruction.

    Average time and energy consumption of each instruction on each stage for each power state.
    """

    def __init__(
        self,
        job_info: JobInfoPerseus,
        rank_infos: list[RankInfoPerseus],
        perseus_settings: PerseusSettings,
        warmup_step_ratio: float = 0.1,
        minimum_power_state: int = 850,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
            warmup_step_ratio: The ratio of warmup steps over the total number of steps within one
                profiling window, ranging from 0 to 1. (Default: `0.1`)
            minimum_power_state: The minimum power state to profile. (Default: `850`)
        """
        super().__init__(job_info, rank_infos, perseus_settings)

        assert (
            self.perseus_settings.dump_data
        ), "InstructionProfiler is designed to dump profiling data."

        self.warmup_step_ratio = warmup_step_ratio
        self.minimum_power_state = minimum_power_state
        self._header = ["stage", "instruction", self.mode, "time", "energy"]
        self._records: list[tuple[int, PipeInstruction, int, float, float]] = []

    def _run(
        self,
    ) -> Generator[list[PowerStateSchedule], list[ProfilingResultPerseus], None]:
        # Warm up the GPUs.
        max_power_state = max(self.power_state_range)
        for _ in range(2):
            _ = yield [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[max_power_state for _ in rank_info.pipe_schedule],
                )
                for rank_info in self.rank_infos
            ]
        # Run each power state one at a time.
        for power_state in sorted(self.power_state_range, reverse=True):
            if power_state < self.minimum_power_state:
                break
            profiling_results = yield [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[power_state for _ in rank_info.pipe_schedule],
                )
                for rank_info in self.rank_infos
            ]
            for stage, prof in enumerate(profiling_results):
                # for inst in prof.time_breakdown:
                for inst in [PipeInstruction.FORWARD, PipeInstruction.BACKWARD]:
                    inst_time = prof.time_breakdown[inst]
                    inst_energy = prof.energy_breakdown[inst]
                    assert len(inst_time) == len(
                        inst_energy
                    ), "Number of iterations for time and energy should be the same"
                    warmup_steps = max(int(len(inst_time) * self.warmup_step_ratio), 1)
                    time_arr = np.array(inst_time[warmup_steps:]).flatten()
                    energy_arr = np.array(inst_energy[warmup_steps:]).flatten()
                    self._records.append(
                        (
                            stage,
                            inst,
                            power_state,
                            time_arr.mean().item(),
                            energy_arr.mean().item(),
                        )
                    )

            # After collecting the result of each power state, save results.
            # This will keep overwriting the same file.
            df = pd.DataFrame.from_records(data=self._records, columns=self._header)
            info = self.job_info
            filepath = (
                f"{self.perseus_settings.dump_dir}/{info.job_id}/"
                f"{info.framework}+{info.model_name}+{info.partition_method}"
                f"+dp{info.dp_degree}"
                f"+tp{info.tp_degree}"
                f"+pp{info.pp_degree}"
                f"+mbs{info.microbatch_size}"
                f"+nmb{info.num_microbatches}"
                ".csv"
            )
            df.to_csv(filepath, index=False)
            logger.info("Saved profiling result to %s", filepath)

        # Training should stop after profiling is done. Raise an error.
        raise RuntimeError("[profiling-done] Profiling is done, so stop training!")


class PointSolutionPerseus(PowerStateSchedulerV2):
    """Reads in point solutions and just executes them."""

    def __init__(
        self,
        job_info: JobInfoPerseus,
        rank_infos: list[RankInfoPerseus],
        perseus_settings: PerseusSettings,
        warmup_iters: int = 2,
        solution_path: str = "frequencies.py",
        reverse: bool = False,
    ) -> None:
        """Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
            warmup_iters: Number of iterations to warm up the GPUs with all max power states.
            solution_path: Path to the solution file or directory.
                If a directory is given, all .py files inside will be run in alphabetical order.
            reverse: If True, reverse the order of the solutions.
        """
        super().__init__(job_info, rank_infos, perseus_settings)

        self.warmup_iters = warmup_iters
        self.solution_path = Path(solution_path)
        self.solutions: list[list[PowerStateSchedule]] = []

        # If the solution path is a directory, we want to run everything in that directory.
        # If not, we just run the one file.
        if self.solution_path.is_dir():
            for file in sorted(self.solution_path.iterdir(), reverse=reverse):
                if file.suffix == ".py":
                    self.solutions.append(self._read_solution(file))
        elif self.solution_path.is_file() and self.solution_path.suffix == ".py":
            self.solutions.append(self._read_solution(self.solution_path))
        else:
            raise ValueError(
                f"Solution path '{self.solution_path}' is not a directory or a .py file."
            )

    def _read_solution(self, solution_path: Path) -> list[PowerStateSchedule]:
        """Read in a solution from a file and convert to a power state schedule."""
        # pylint: disable=eval-used
        with open(solution_path, encoding="utf-8") as f:
            solution = eval(f.read())

        # Convert the solution to a list of power states.
        return self._config_to_schedule(solution)

    def _run(
        self,
    ) -> Generator[list[PowerStateSchedule], list[ProfilingResultPerseus], None]:
        # Warm up the GPUs.
        max_power_state = max(self.power_state_range)
        for _ in range(self.warmup_iters):
            _ = yield [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[max_power_state for _ in rank_info.pipe_schedule],
                )
                for rank_info in self.rank_infos
            ]

        # Run each solution one at a time.
        for solution in self.solutions:
            _ = yield solution

        # Stop running after all solutions are done.
        raise RuntimeError("[profiling-done] All solutions are done, so stop training!")


PointSolution3_perseus = make_3d_parallel_perseus(PointSolutionPerseus)
