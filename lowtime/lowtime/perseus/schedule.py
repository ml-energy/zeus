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

"""Collection of schedules that describe the ordering of pipeline instructions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Type

from lowtime.operation import OperationSpec
from lowtime.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    Recomputation,
    ForwardBackward,
)


class PipelineSchedule(ABC):
    """Abstract class that defines a pipeline schedule.

    Designed to look like DeepSpeed's PipeSchedule class.
    """

    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int,
        operation_spec_map: dict[Type[Instruction], OperationSpec],
    ) -> None:
        """Instantiate the pipeline schedule.

        Arguments:
            num_stages: The number of pipeline stages.
            num_micro_batches: The number of micro batches in the pipeline.
            stage_id: Zero-indexed pipeline stage to yield instructions for.
            operation_spec_map: A map from instruction type to operation spec.
        """
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages
        self.op_spec_map = operation_spec_map
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage.

        `Instruction`s just need their stage ID and microbatch ID.

        This method also corresponds to DeepSpeed's PipeSchedule.steps method.
        However, in Lowtime, one step doesn't have much meaning. We just exhaust the
        generator immediately to get a list of all instructions.
        """


class Synchronous1F1B(PipelineSchedule):
    """Describes the synchronous 1F1B schedule.

    Adapted from DeepSpeed's TrainSchedule class.
    """

    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage."""
        f_spec = self.op_spec_map[Forward]
        b_spec = self.op_spec_map[Backward]

        total_steps = 2 * (self.num_micro_batches + self.num_stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    yield Forward(
                        spec=f_spec,
                        stage_id=self.stage_id,
                        micro_batch_id=micro_batch_id,
                    )
                else:
                    yield Backward(
                        spec=b_spec,
                        stage_id=self.stage_id,
                        micro_batch_id=micro_batch_id,
                    )

    def _valid_stage(self, stage_id) -> bool:
        return 0 <= stage_id < self.num_stages

    def _valid_micro_batch(self, micro_batch_id) -> bool:
        return 0 <= micro_batch_id < self.num_micro_batches

    def _step_to_micro_batch(self, step_id):
        def _is_even(x: int) -> bool:
            return x % 2 == 0

        def _is_odd(x: int) -> bool:
            return x % 2 != 0

        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            raise AssertionError()

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.num_stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.num_stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class EarlyRecomputation1F1B(Synchronous1F1B):
    """Early recomputation 1F1B schedule from Merak."""

    def __iter__(self) -> Generator[Instruction, None, None]:
        """Return a generator that yields `Instruction`s for one stage."""
        f_spec = self.op_spec_map[Forward]
        b_spec = self.op_spec_map[Backward]
        r_spec = self.op_spec_map[Recomputation]
        fb_spec = self.op_spec_map[ForwardBackward]

        prev_micro_batch_id = -1
        total_steps = 2 * (self.num_micro_batches + self.num_stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            cmds: list[Instruction] = []

            # Recomputation right before Backwards.
            if not is_forward and (
                not self._valid_micro_batch(prev_micro_batch_id)
                and self._valid_micro_batch(micro_batch_id)
                and self._valid_stage(self.next_stage)
            ):
                cmds.append(
                    Recomputation(
                        spec=r_spec,
                        stage_id=self.stage_id,
                        micro_batch_id=micro_batch_id,
                    )
                )

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if (
                        self.num_micro_batches - micro_batch_id + self.stage_id
                        < self.num_stages
                    ):
                        # Used to be PreCheckpointForward
                        cmds.append(
                            Forward(
                                spec=f_spec,
                                stage_id=self.stage_id,
                                micro_batch_id=micro_batch_id,
                            )
                        )
                    else:
                        cmds.append(
                            Forward(
                                spec=f_spec,
                                stage_id=self.stage_id,
                                micro_batch_id=micro_batch_id,
                            )
                        )
                else:  # noqa: PLR5501
                    if (
                        micro_batch_id
                        <= self.num_micro_batches - self.num_stages + self.stage_id
                    ):
                        cmds.append(
                            ForwardBackward(
                                spec=fb_spec,
                                stage_id=self.stage_id,
                                micro_batch_id=micro_batch_id,
                            )
                        )
                    else:
                        cmds.append(
                            Backward(
                                spec=b_spec,
                                stage_id=self.stage_id,
                                micro_batch_id=micro_batch_id,
                            )
                        )

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield from cmds
