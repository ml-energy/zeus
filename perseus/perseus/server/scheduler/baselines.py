"""Schedulers that implement baselines."""

from __future__ import annotations

import json
from typing import Generator, Literal
from functools import lru_cache

import numpy as np
from fastapi.logger import logger

from perseus.models import (
    PerseusSettings,
    JobInfo,
    RankInfo,
    PowerStateSchedule,
    ProfilingResult,
    PipeInstruction,
)
from perseus.server.scheduler.interface import (
    PowerStateScheduler,
    PowerStateSchedulerV2,
    FrequencyScheduler,
    PowerLimitScheduler,
    make_3d_parallel,
)


class AllMax(PowerStateScheduler):
    """Run everything with the maximum power state."""

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        perseus_settings: PerseusSettings,
        num_iters: int = 1,
    ) -> None:
        """Initialize the scheduler."""
        super().__init__(job_info, rank_infos, perseus_settings)

        self.num_instructions = [len(ri.pipe_schedule) for ri in self.rank_infos]
        self.max_power_state = max(self.power_state_range)
        self.num_iters = num_iters
        logger.debug("self.num_instructions=%s", self.num_instructions)
        logger.debug("self.max_power_state=%s", self.max_power_state)
        logger.debug("self.num_iters=%s", self.num_iters)

    def observe(self, _: list[ProfilingResult]) -> None:
        """Observe nothing."""
        if self.num_iters == 0:
            raise RuntimeError("[profiling-done] Profiling done for AllMax!")

    def next_schedule(self) -> list[PowerStateSchedule]:
        """Return the next schedules."""
        self.num_iters -= 1
        return [
            PowerStateSchedule(
                rank=rank,
                power_states=[self.max_power_state] * self.num_instructions[rank],
            )
            for rank in range(len(self.num_instructions))
        ]


class AllMaxFrequency(AllMax):
    """Run everything with the maximum frequency."""

    mode = "frequency"


class AllMaxPowerLimit(AllMax):
    """Run everything with the maximum power limit.

    This is the "default" execution mode of training jobs that are not energy-aware.
    """

    mode = "power limit"


AllMaxPowerLimit3D = make_3d_parallel(AllMaxPowerLimit)


class TwoFrequencies(FrequencyScheduler):
    """Runs forward with one frequency and backward with another.

    Frequency will be changed only when a forward or backward instruction takes place.
    """

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        perseus_settings: PerseusSettings,
        forward_freq: int,
        backward_freq: int,
    ) -> None:
        """Initialize the scheduler."""
        super().__init__(job_info, rank_infos, perseus_settings)

        # Sanity check.
        if forward_freq not in self.rank_infos[0].power_state_range:
            raise ValueError(f"Forward frequency {forward_freq} is not supported.")
        if backward_freq not in self.rank_infos[0].power_state_range:
            raise ValueError(f"Backward frequency {backward_freq} is not supported.")

        self.forward_freq = forward_freq
        self.backward_freq = backward_freq
        logger.debug("Forward running with frequency %s", self.forward_freq)
        logger.debug("Backward running with frequency %s", self.backward_freq)

    def observe(self, _: list[ProfilingResult]) -> None:
        """Observe nothing."""

    @lru_cache(maxsize=1)
    def next_schedule(self) -> list[PowerStateSchedule]:
        """Return the next schedule."""
        schedules = []
        for rank_info in self.rank_infos:
            rank_schedule = []
            freq = self.forward_freq
            for inst in rank_info.pipe_schedule:
                if inst is PipeInstruction.FORWARD:
                    freq = self.forward_freq
                elif inst is PipeInstruction.BACKWARD:
                    freq = self.backward_freq
                rank_schedule.append(freq)
            schedules.append(
                PowerStateSchedule(rank=rank_info.rank, power_states=rank_schedule)
            )
        logger.debug("Schedules are:\n%s", schedules)
        return schedules


class ZeusGlobalPowerLimit(PowerLimitScheduler):
    """Finds a global power limit for all GPUs using Zeus's JIT profiling algorithm."""

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        perseus_settings: PerseusSettings,
        eta_knob: float = 0.5,
        warmup_step_ratio: float = 0.1,
    ) -> None:
        r"""Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
            eta_knob: $\eta$ used in the cost metric defined in Zeus, which represents the relative
                importance between performance and energy efficiency. (Default: `0.5`)
            warmup_step_ratio: The ratio of warmup steps over the total number of steps within one
                profiling window, ranging from 0 to 1. (Default: `0.1`)
        """
        super().__init__(job_info, rank_infos, perseus_settings)
        self.eta_knob = eta_knob
        self.warmup_step_ratio = warmup_step_ratio

        self.next_pl_index = 0
        self.optimal_pl = 0
        self.train_tput_result: dict[int, float] = {}
        self.train_power_result: dict[int, float] = {}
        self.train_energy_result: dict[int, float] = {}
        self.max_pl = max(self.power_state_range) * self.world_size

        # HACK: Cutoff the lowest power state.
        self.power_state_range = sorted(self.power_state_range, reverse=True)[:-1]

        self.warmup_iters = 1

        logger.debug("Power limit range: %s", self.power_state_range)
        logger.debug("MaxPower: %sW", self.max_pl)

    def observe(self, profiling_results: list[ProfilingResult]) -> None:
        """Ingest the profiling results for the previous schedule."""
        if self.warmup_iters > 0:
            self.warmup_iters -= 1
            return

        # Profiling
        if self.next_pl_index < len(self.power_state_range):
            pl = self.power_state_range[self.next_pl_index]

            # Slicing off warmup steps
            assert all(
                len(result.iter_time) == len(result.iter_energy)
                for result in profiling_results
            ), "Number of iterations for time and energy should be the same"
            warmup_steps = [
                max(int(len(result.iter_time) * self.warmup_step_ratio), 1)
                for result in profiling_results
            ]
            # Compute the average throughput within the profiling window.
            tput = 1.0 / max(
                np.mean(result.iter_time[step:]).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            # Compute the average power within the profiling window.
            power = sum(
                np.mean(
                    np.array(result.iter_energy[step:])
                    / np.array(result.iter_time[step:])
                ).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            # Compute the average energy consumption for one iteration within the profiling window.
            energy = sum(
                np.mean(result.iter_energy[step:]).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            self.train_tput_result[pl] = tput
            self.train_power_result[pl] = power
            self.train_energy_result[pl] = energy
            logger.debug(
                "Profiling results for PL %sW: tput=%s, power=%s, energy=%s",
                pl,
                tput,
                power,
                energy,
            )
            self.next_pl_index += 1

        # Profiling done
        if self.next_pl_index == len(self.power_state_range):
            tput_ = self.train_tput_result
            power_ = self.train_power_result
            cost_map = {
                pl: (
                    self.eta_knob * power_[pl]
                    + (1 - self.eta_knob) * self.max_pl * self.world_size
                )
                / tput_[pl]
                for pl in self.power_state_range
            }
            self.optimal_pl = min(cost_map.keys(), key=lambda k: cost_map[k])
            logger.debug("Optimal PL is %sW", self.optimal_pl)
            self.next_pl_index += 1  # So that we only compute the optimal PL once.

            if self.perseus_settings.dump_data:
                results = {
                    pl: {
                        "time": 1.0 / self.train_tput_result[pl],
                        "energy": self.train_energy_result[pl],
                    }
                    for pl in self.power_state_range
                }
                with open(
                    f"{self.perseus_settings.dump_dir}/{self.rank_infos[0].job_id}/ZeusGlobalPowerLimit.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(results, f)

    def next_schedule(self) -> list[PowerStateSchedule]:
        """Return the next schedules for all ranks in order."""
        if self.warmup_iters > 0:
            return [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[max(self.power_state_range) for _ in rank_info.pipe_schedule],
                )
                for rank_info in self.rank_infos
            ]

        # Profiling
        if self.next_pl_index < len(self.power_state_range):
            next_pl = self.power_state_range[self.next_pl_index]
            return [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[next_pl for _ in rank_info.pipe_schedule],
                )
                for rank_info in self.rank_infos
            ]

        # HACK: Stop training when profiling is done!
        raise RuntimeError("[profiling-done] Profiling done for ZeusGlobal")

        # Profiling done
        # HACK: Restore it!
        # assert self.optimal_pl > 0
        # return [
        #     PowerStateSchedule(
        #         rank=rank_info.rank,
        #         power_states=[self.optimal_pl for _ in rank_info.pipe_schedule],
        #     )
        #     for rank_info in self.rank_infos
        # ]


class ZeusLocalPowerState(PowerStateSchedulerV2):
    """Zeus+ baseline.

    Zeus+ will balance computation time over either forward or backward phase across all the stages.
    """

    def __init__(
        self,
        job_info: JobInfo,
        rank_infos: list[RankInfo],
        perseus_settings: PerseusSettings,
        eta_knob: float = 0.5,
        balanced_phase: Literal["forward", "backward"] = "forward",
        warmup_step_ratio: float = 0.1,
    ):
        r"""Initialize the scheduler.

        Args:
            job_info: Info about the training job.
            rank_infos: Info about all ranks. May not be sorted in rank order.
            perseus_settings: PerseusSettings object.
            eta_knob: $\eta$ used in the cost metric defined in Zeus, which represents the relative
                importance between performance and energy efficiency. (Default: `0.5`)
            balanced_phase: On which phase the computation time will be balanced across all the stage,
                either `"forward"` or `"backward"`. (Default: `"forward"`)
            warmup_step_ratio: The ratio of warmup steps over the total number of steps within one
                profiling window, ranging from 0 to 1. (Default: `0.1`)
        """
        super().__init__(job_info, rank_infos, perseus_settings)

        # Sanity check.
        # Pipeline parallelism only.
        assert all(
            rank_info.dp_rank == 0 and rank_info.tp_rank == 0
            for rank_info in rank_infos
        )

        self.eta_knob = eta_knob
        self.balanced_phase = PipeInstruction(balanced_phase)
        self.warmup_step_ratio = warmup_step_ratio

    # ruff: noqa: PLR0915, PLR2004
    def _run(self) -> Generator[list[PowerStateSchedule], list[ProfilingResult], None]:
        """Run the scheduler."""
        # Warmup one iteration with all max power states.
        _ = yield [
            PowerStateSchedule(
                rank=rank_info.rank,
                power_states=[max(self.power_state_range) for _ in rank_info.pipe_schedule],
            )
            for rank_info in self.rank_infos
        ]

        # The heaviest stage, i.e., the stage that consumes the longest computation time
        # on the balanced phase (either forward or backward).
        heaviest_stage: int | None = None

        # Sort the power states to prepare for searching.
        ordered_power_state_range: list[int] = sorted(
            self.power_state_range, reverse=True
        )

        # Save the mapping from each power state to throughput/power after balancing the computation time.
        train_tput_result: dict[int, float] = {}
        train_power_result: dict[int, float] = {}
        train_energy_result: dict[int, float] = {}

        # For each power state of the heaviest stage, save the balanced power state of each stage.
        # The value type is a list of the balanced power state on each stage.
        balanced_power_states: dict[int, list[int]] = {}

        # The cost-optimal set of power states found after profiling.
        optimal_power_states: list[int] = []

        # The next power state to profile for each rank
        next_power_state_indices: list[int] = [0 for _ in range(self.world_size)]

        # We profile the heaviest stage while balancing the computation time of other stages. We define the balanced
        # power state as the one that gives the minimum distance, where distance is the difference in the computation
        # time of the balanced phase between each stage and the heaviest stage.
        # Track the previous balanced power state indices for each rank to save the search time.
        for heaviest_stage_power_index, heaviest_stage_power_state in enumerate(
            ordered_power_state_range
        ):
            # The index of the power state that searching should start from. This will be the greater index
            # (i.e. lower power state) between the previous balanced power state on that stage and the power state
            # of the heaviest stage.
            next_power_state_indices = [
                max(index, heaviest_stage_power_index)
                for index in next_power_state_indices
            ]
            next_power_states: list[int] = [
                ordered_power_state_range[index] for index in next_power_state_indices
            ]
            profiling_results: list[ProfilingResult] = yield [
                PowerStateSchedule(
                    rank=rank_info.rank,
                    power_states=[
                        next_power_states[rank_info.rank]
                        for _ in rank_info.pipe_schedule
                    ],
                )
                for rank_info in self.rank_infos
            ]
            assert all(
                len(result.iter_time) >= 3 for result in profiling_results
            ), "Need more than three samples so that computation is sufficiently warmed up"
            warmup_steps = [
                max(int(len(result.iter_time) * self.warmup_step_ratio), 1)
                for result in profiling_results
            ]
            curr_balanced_phase_avg_time: list[float] = [
                np.mean(result.time_breakdown[self.balanced_phase][step:]).item()  # type: ignore
                for result, step in zip(profiling_results, warmup_steps)
            ]
            # Locate the heaviest stage
            if heaviest_stage is None:
                heaviest_stage = np.array(curr_balanced_phase_avg_time).argmax().item()

            # Record the average instruction time for the heaviest stage
            heaviest_stage_balanced_phase_avg_time: float = (
                curr_balanced_phase_avg_time[heaviest_stage]
            )

            # Record the average time of the balanced phase from the previous profiling window, to check
            # the monotonicity of time.
            prev_balanced_phase_avg_time: list[float] = [
                0 for _ in range(self.world_size)
            ]

            # The distance from the previous profiling window. We found the power state that gives us the
            # local minimum distance, i.e., we stop searching once we find the distance starts to increase.
            prev_distances: list[float] = [float("inf") for _ in range(self.world_size)]
            # Whether the balanced power state for each non-heaviest stage has been found.
            balanced_power_state_found: list[bool] = [
                stage == heaviest_stage for stage in range(self.world_size)
            ]

            # Searching for the balanced power state for all non-heaviest rank.
            while not all(balanced_power_state_found):
                # Calculate the current distance.
                curr_distances: list[float] = [
                    abs(
                        curr_balanced_phase_avg_time[stage]
                        - heaviest_stage_balanced_phase_avg_time
                    )
                    for stage in range(self.world_size)
                ]
                # Tune the non-heaviest stages, whose balanced power state has not been found
                for stage, found in enumerate(balanced_power_state_found):
                    if found:
                        continue
                    # Keep searching until both of the two conditions hold:
                    #   1) The distance starts to increase, and
                    #   2) The monotonicity of time still holds
                    if (
                        curr_distances[stage] <= prev_distances[stage]
                        or curr_balanced_phase_avg_time[stage]
                        < prev_balanced_phase_avg_time[stage]
                    ):
                        # Distance still decreases, tune down the power state if possible.
                        if next_power_state_indices[stage] + 1 < len(
                            ordered_power_state_range
                        ):
                            next_power_state_indices[stage] += 1
                        else:
                            # If the current power state is the last one in the reversed power state list
                            # (i.e., the lowest power state), stick to this power state and stop searching.
                            balanced_power_state_found[stage] = True
                    else:
                        # Distance starts to increase, stop searching and save the balanced power state.
                        # Set the power state to previous power state, which is the one that gives the
                        # minimum distance.
                        next_power_state_indices[stage] -= 1
                        balanced_power_state_found[stage] = True
                # Trace the avg time and distance from previous profiling window.
                prev_balanced_phase_avg_time = curr_balanced_phase_avg_time
                prev_distances = curr_distances

                # Explore the next power states
                next_power_states = [
                    ordered_power_state_range[index]
                    for index in next_power_state_indices
                ]
                profiling_results = yield [
                    PowerStateSchedule(
                        rank=rank_info.rank,
                        power_states=[
                            next_power_states[rank_info.rank]
                            for _ in rank_info.pipe_schedule
                        ],
                    )
                    for rank_info in self.rank_infos
                ]
                # Calculate the average of time on balanced phase (either forward or backword)
                # over one iteration when running full power.
                assert all(
                    len(result.iter_time) >= 3 for result in profiling_results
                ), "Need more than three samples so that computation is sufficiently warmed up"
                warmup_steps = [
                    max(int(len(result.iter_time) * self.warmup_step_ratio), 1)
                    for result in profiling_results
                ]
                curr_balanced_phase_avg_time = [  # type: ignore
                    np.mean(result.time_breakdown[self.balanced_phase][step:]).item()
                    for result, step in zip(profiling_results, warmup_steps)
                ]

            # Balanced power states for each stage found.
            # Save the balanced power state for the current power state of the heaviest stage
            power_states = [
                ordered_power_state_range[index] for index in next_power_state_indices
            ]
            balanced_power_states[heaviest_stage_power_state] = power_states
            # Save throughput/power/energy for the current power state of the heaviest stage
            assert all(
                len(result.iter_time) == len(result.iter_energy)
                for result in profiling_results
            ), "Number of iterations for time and energy should be the same"
            assert all(
                len(result.iter_time) >= 3 for result in profiling_results
            ), "Need more than three samples so that computation is sufficiently warmed up"
            warmup_steps = [
                max(int(len(result.iter_time) * self.warmup_step_ratio), 1)
                for result in profiling_results
            ]
            # Compute the average throughput within the profiling window.
            tput = 1.0 / max(
                np.mean(result.iter_time[step:]).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            # Compute the average power within the profiling window.
            power = sum(
                np.mean(
                    np.array(result.iter_energy[step:])
                    / np.array(result.iter_time[step:])
                ).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            # Compute the average energy consumption for one iteration within the profiling window.
            energy = sum(
                np.mean(result.iter_energy[step:]).item()
                for result, step in zip(profiling_results, warmup_steps)
            )
            train_tput_result[heaviest_stage_power_state] = tput
            train_power_result[heaviest_stage_power_state] = power
            train_energy_result[heaviest_stage_power_state] = energy
            logger.debug(
                "Balanced power states found. %s:",
                self._power_states_info(power_states, tput, power, energy),
            )

            # If power state on any rank touches the lower bound, no more balanced power state could be found. Break.
            if any(
                power_state == ordered_power_state_range[-1]
                for power_state in power_states
            ):
                break

        # Profiling done.
        # Sanity checks.
        assert heaviest_stage is not None, "Heaviest stage not found."
        # Compute the optimal power state.
        max_power_state = ordered_power_state_range[0]
        cost_map = {
            power_state: (
                self.eta_knob * train_power_result[power_state]
                + (1 - self.eta_knob) * max_power_state * self.world_size
            )
            / train_tput_result[power_state]
            for power_state in balanced_power_states
        }
        optimal_heaviest_stage_power_state: int = min(
            cost_map.keys(), key=lambda k: cost_map[k]
        )
        optimal_power_states = balanced_power_states[optimal_heaviest_stage_power_state]
        tput = train_tput_result[optimal_heaviest_stage_power_state]
        power = train_power_result[optimal_heaviest_stage_power_state]
        energy = train_energy_result[optimal_heaviest_stage_power_state]
        logger.debug(
            "Profiling done. Optimal power state found. %s",
            self._power_states_info(optimal_power_states, tput, power, energy),
        )
        if self.perseus_settings.dump_data:
            results = {
                "heaviest_stage": heaviest_stage,
                "optimal_heaviest_stage_power_state": optimal_power_states[
                    heaviest_stage
                ],
                "profiling_results": {
                    heaviest_stage_power_state: {
                        "balanced_power_states": power_states,
                        "throughput": train_tput_result[heaviest_stage_power_state],
                        "power": train_power_result[heaviest_stage_power_state],
                        "time": 1.0 / train_tput_result[heaviest_stage_power_state],
                        "energy": train_energy_result[heaviest_stage_power_state],
                    }
                    for heaviest_stage_power_state, power_states in balanced_power_states.items()
                },
            }
            job_id = self.rank_infos[0].job_id
            with open(
                f"{self.perseus_settings.dump_dir}/{job_id}/ZeusLocalPowerState.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results, f)

        # HACK: Stop training when profiling is done!
        raise RuntimeError("[profiling-done] Profiling done for ZeusLocal!")

        # The last yield that sets to the optimal power state
        # HACK: Restore it!
        # yield [
        #     PowerStateSchedule(
        #         rank=rank_info.rank,
        #         power_states=[
        #             optimal_power_states[rank_info.rank]
        #             for _ in rank_info.pipe_schedule
        #         ],
        #     )
        #     for rank_info in self.rank_infos
        # ]

    def _power_states_info(
        self, power_states: list[int], tput: float, power: float, energy: float
    ) -> str:
        """Return the power states with the profiling info."""
        states = ",".join([str(power_state) for power_state in power_states])
        unit = "W" if self.mode == "power limit" else "MHz"
        return f"Power state is ({states}){unit}: throughput={tput}, power={power}, energy={energy}."


ZeusLocalPowerState3D = make_3d_parallel(ZeusLocalPowerState)
