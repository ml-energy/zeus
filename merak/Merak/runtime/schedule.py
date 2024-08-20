# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/schedule.py

from .utils import call_to_str
from ..utils.merak_args import get_args

from queue import Queue
from abc import ABC, abstractmethod


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are generators that yield sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """
    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipeInstruction` for each step in the schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ForwardPass(recv_buf))

            yield cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class ProfileSchedule(PipeSchedule):
    """A schedule for profiling the time and energy consumption of each instruction."""

    def steps(self):
        """Dummy implementation to avoid ABC instantiation error."""
        yield []

    def num_pipe_buffers(self):
        return 1

    def buffer_fill_steps(self):
        cmds = []

        if self.stage_id != 0:
            cmds.append(RecvActivation(0))
        if self.stage_id == 0 or self.stage_id == self.stages - 1:
            cmds.append(LoadMicroBatch(0))
        cmds.append(ForwardPass(0))
        if self.stage_id != self.stages - 1:
            cmds.append(SendActivation(0))

        if self.stage_id != self.stages - 1:
            cmds.append(RecvGrad(0))
        cmds.append(BackwardPass(0))
        if self.stage_id != 0:
            cmds.append(SendGrad(0))

        cmds.append(ReduceTiedGrads())
        cmds.append(ReduceGrads())
        cmds.append(OptimizerStep())

        return [cmds]

    def forward_steps(self, num_steps: int):
        return [[ForwardPass(0)] * num_steps]

    def backward_steps(self, num_steps: int):
        return [[BackwardPass(0)] * num_steps]


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(SendActivation(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
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
            assert False

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
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class EnvPipeTrainSchedule(TrainSchedule):
    """EnvPipe schedule."""

    def __init__(
        self,
        micro_batches,
        stages,
        stage_id,
        reschedule_forward_cnt=None,
    ):
        super().__init__(micro_batches, stages, stage_id)

        if reschedule_forward_cnt is None:
            self.reschedule_forward_cnt = get_args().envpipe_reschedule_cnt
        else:
            self.reschedule_forward_cnt = reschedule_forward_cnt

        print("Reschedule forward cnt:", self.reschedule_forward_cnt)

    def steps(self):
        prev_micro_batch_id = -1
        total_steps = self.num_total_steps()
        send_activation_queue = Queue()

        # How many microbatches worth of activations to defer sending.
        # The last stage is never going to send any activations, and this variable will never be read.
        if self.stage_id != self.stages - 1:
            num_send_activation_defer = self._num_upfront_forwards(self.stage_id) - self._num_upfront_forwards(self.next_stage) - 1
        else:
            num_send_activation_defer = 0

        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # breakpoint()

            # Handle communication with the previous stage in forward steps.
            if is_forward:
                # The previous stages will always have the activations for the current forward
                # computation, because reschedule_forward_cnt was made to decrease.
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))

                # Sending the gradient is always for the previous microbatch, since backward
                # computations are not rescheduled.
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
            # Handle communication with the next stage in backward steps.
            else:
                # There should a next stage to which to send data to.
                if self._valid_stage(self.next_stage):
                    if (
                        self._valid_micro_batch(prev_micro_batch_id)
                        and prev_micro_batch_id < self._num_upfront_forwards(self.next_stage)
                    ):
                        cmds.append(SendActivation(prev_buffer))
                    elif self._valid_micro_batch(prev_micro_batch_id) and num_send_activation_defer:
                        send_activation_queue.put(prev_micro_batch_id)
                        num_send_activation_defer -= 1
                    else:
                        if self._valid_micro_batch(prev_micro_batch_id):
                            send_activation_queue.put(prev_buffer)
                        if not send_activation_queue.empty():
                            buffer_id_to_send = send_activation_queue.get()
                            cmds.append(SendActivation(buffer_id_to_send))

                # The next stage will always have just executed the backward pass. Always receive.
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_total_steps(self):
        return 2 * (self.micro_batches + self.stages - 1 + max(self.reschedule_forward_cnt))

    def _num_upfront_forwards(self, stage_id: int) -> int:
        """Return the number of forward computations during the pipeline ramp up phase."""
        return self.stages - stage_id + self.reschedule_forward_cnt[stage_id]

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id_ours(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id_ours(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id_ours(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id_ours(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id_ours(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id_ours(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id_ours(self, step_id):
        base = step_id // 2 - self.reschedule_forward_cnt[self.stage_id]
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id_ours(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1 - \
            self.reschedule_forward_cnt[self.stage_id]
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id

    def num_pipe_buffers(self):
        if self.is_last_stage:
            return 2
        else:
            return min(self.stages - self.stage_id + 1, self.micro_batches) + \
                self.reschedule_forward_cnt[self.stage_id]

        

class MergeP2PTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        cmds.append(SendGradRecvActivation((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                        cmds.append(SendActivationRecvGrad((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

class PreRecomputeTrainSchedule(TrainSchedule):
    """Early recomputation schedule."""

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        cmds.append(SendGradRecvActivation((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                        cmds.append(SendActivationRecvGrad((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id < self.stages:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class LastNoRecomputeTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        frist_sendrecv = True
        frist_fp = True
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        if self.stage_id == self.stages - 1 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        else:
                            cmds.append(SendGradRecvActivation((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                        if self.stage_id == self.stages - 2 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(SendActivation(prev_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendActivationRecvGrad((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self.stage_id == self.stages - 2:
                        cmds.append(RecvGrad(curr_buffer))
                    else:
                        cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id < self.stages \
                        and self.stage_id != self.stages - 2:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    elif self.stage_id == self.stages - 2 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

class FullCriticalPathTrainSchedule(TrainSchedule):
    """Shifted critical path schedule."""

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        frist_sendrecv = True
        second_sendrecv = True
        frist_fp = True
        second_fp = True
        frist_bp = 0
        b0_buffer = self._buffer_idx(0)
        f2_buffer = self._buffer_idx(2)

        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id):
                        if self.stage_id == self.stages - 1 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        elif self.stage_id == self.stages - 1 and second_sendrecv:
                            second_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        elif self.stage_id == self.stages - 2 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(RecomputeRecvGrad(b0_buffer))
                            cmds.append(SendActivation(f2_buffer-1))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendGradRecvActivation((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                            self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                elif frist_sendrecv and self.stages == 2 and self._valid_micro_batch(micro_batch_id) and self.stage_id == 0 and self._valid_micro_batch(prev_micro_batch_id):
                    frist_sendrecv = False
                    cmds.append(RecomputeRecvGrad(b0_buffer))
                    cmds.append(SendActivation(f2_buffer-1))
                    cmds.append(RestoreRecomputeStatus())
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                        if self.stage_id == self.stages - 2 and frist_sendrecv:
                            cmds.append(RecvActivation(f2_buffer))
                        elif self.stage_id == self.stages - 2 and second_sendrecv:
                            second_sendrecv = False
                            cmds.append(SendGrad(b0_buffer))
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(SendActivation(f2_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        elif self.stage_id == self.stages - 3 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendActivation(prev_buffer))
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendActivationRecvGrad((prev_buffer, curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    if self.stage_id == self.stages - 2:
                        cmds.append(RecvGrad(curr_buffer))
                    else:
                        cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id < self.stages \
                        and self.stage_id != self.stages - 2:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    elif self.stage_id == self.stages - 2 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    elif self.stage_id == self.stages - 2 and second_fp:
                        second_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    elif self.stage_id == self.stages - 2 and frist_bp == 1:
                        # use third forward for first backward 

                        frist_bp = 2
                        if isinstance(cmds[-1], LoadMicroBatch):
                            cmds[-1] = BackwardPass(b0_buffer)
                        else:
                            cmds.append(BackwardPass(b0_buffer))

                    elif self.stage_id == self.stages - 3 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    if self.stage_id == self.stages - 2 and frist_bp == 0:
                        # use first backward for third forward 
                        frist_bp = 1
                        if self.stage_id == 0:
                            cmds.append(LoadMicroBatch(f2_buffer))
                        cmds.append(ForwardPass(f2_buffer))
                    else:
                        cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with gradient
    accumulation.
    """
    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                LoadMicroBatch(buffer_id=0),
                ForwardPass(buffer_id=0),
                BackwardPass(buffer_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend([
                    ReduceGrads(),
                    OptimizerStep(),
                ])
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed.
        """
        return 1


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """
    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass


# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['ouputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['ouputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass


# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass

class SendActivationRecvGrad(BufferOpInstruction):
    pass

class SendGradRecvActivation(BufferOpInstruction):
    pass

class PreCheckpointForwardPass(BufferOpInstruction):
    pass

class RecomputeRecvGrad(BufferOpInstruction):
    pass

class RestoreRecomputeStatus(PipeInstruction):
    pass


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0


class EnvPipeTrainSchedule0(TrainSchedule):
    """Pipeline schedule used by EnvPipe. This doesn't work."""

    def __init__(
        self,
        micro_batches,
        stages,
        stage_id,
        cnt=None,
    ) -> None:
        super().__init__(micro_batches, stages, stage_id)
        if cnt is not None:
            self.reschedule_forward_cnt = cnt
        else:
            self.reschedule_forward_cnt = get_args().envpipe_reschedule_cnt
            assert self.reschedule_forward_cnt is not None
            assert len(self.reschedule_forward_cnt) == stages
        print("EnvPipe reschedule forward count:", self.reschedule_forward_cnt)

    def steps(self):
        prev_micro_batch_id = -1
        total_steps = self.num_total_steps()
        send_activation_queue = Queue()

        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))

                # TODO skip send gradient for last stage

                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                        self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
            else:
                if not send_activation_queue.empty() and self._valid_stage(
                        self.next_stage):
                    micro_batch_id_to_send = send_activation_queue.get()
                    cmds.append(SendActivation(
                        self._buffer_idx(micro_batch_id_to_send)))

                    # TODO send activation twice for second last stage

                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                        self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                    if self._valid_stage(self.next_stage):
                        send_activation_queue.put(micro_batch_id)
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


    def num_total_steps(self):
        return 2 * (self.micro_batches + self.stages - 1 + max(self.reschedule_forward_cnt))

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id_ours(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id_ours(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id_ours(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id_ours(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id_ours(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id_ours(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id_ours(self, step_id):
        base = step_id // 2 - self.reschedule_forward_cnt[self.stage_id]
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id_ours(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1 - \
            self.reschedule_forward_cnt[self.stage_id]
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id

    def num_pipe_buffers(self):
        if self.is_last_stage:
            return 2
        else:
            return min(self.stages - self.stage_id + 1, self.micro_batches) + \
                self.reschedule_forward_cnt[self.stage_id]


class MaxBufferTrainSchedule(TrainSchedule):
    """1F1B train schedule, but uses the maximum number of buffers."""

    def num_pipe_buffers(self):
        return self.micro_batches


class ManualSwapTrainSchedule(TrainSchedule):
    """Dumb train schedule class used to observe how good swapping is."""

    def __init__(self, micro_batches, stages, stage_id):
        assert micro_batches == 4 and stages == 2 and 0 <= stage_id < 2
        super().__init__(micro_batches, stages, stage_id)

        self.args = get_args()
        
    def num_pipe_buffers(self):
        return self.num_micro_batches

    def steps(self):
        if self.args.num_swaps == 0:
            if self.stage_id == 0:
                yield from [
                    [LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [SendActivation(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [SendActivation(buffer_id=1), RecvGrad(buffer_id=0), BackwardPass(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [SendActivation(buffer_id=2), RecvGrad(buffer_id=1), BackwardPass(buffer_id=1)],
                    [LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [SendActivation(buffer_id=3), RecvGrad(buffer_id=2), BackwardPass(buffer_id=2)],
                    [],
                    [RecvGrad(buffer_id=3), BackwardPass(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
            elif self.stage_id == 1:
                yield from [
                    [],
                    [RecvActivation(buffer_id=0), LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [BackwardPass(buffer_id=0)],
                    [RecvActivation(buffer_id=1), SendGrad(buffer_id=0), LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [BackwardPass(buffer_id=1)],
                    [RecvActivation(buffer_id=2), SendGrad(buffer_id=1), LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [BackwardPass(buffer_id=2)],
                    [RecvActivation(buffer_id=3), SendGrad(buffer_id=2), LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [BackwardPass(buffer_id=3)],
                    [SendGrad(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
        elif self.args.num_swaps == 1:
            if self.stage_id == 0:
                yield from [
                    [LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [SendActivation(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],  # SWAPPED
                    [SendActivation(buffer_id=1), RecvGrad(buffer_id=0), BackwardPass(buffer_id=0)],  # SWAPPED
                    [SendActivation(buffer_id=2), RecvGrad(buffer_id=1), BackwardPass(buffer_id=1)],
                    [LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [SendActivation(buffer_id=3), RecvGrad(buffer_id=2), BackwardPass(buffer_id=2)],
                    [],
                    [RecvGrad(buffer_id=3), BackwardPass(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
            elif self.stage_id == 1:
                yield from [
                    [],
                    [RecvActivation(buffer_id=0), LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [BackwardPass(buffer_id=0)],
                    [RecvActivation(buffer_id=1), SendGrad(buffer_id=0), LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [BackwardPass(buffer_id=1)],
                    [RecvActivation(buffer_id=2), SendGrad(buffer_id=1), LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [BackwardPass(buffer_id=2)],
                    [RecvActivation(buffer_id=3), SendGrad(buffer_id=2), LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [BackwardPass(buffer_id=3)],
                    [SendGrad(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
        elif self.args.num_swaps == 2:
            if self.stage_id == 0:
                yield from [
                    [LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [SendActivation(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [SendActivation(buffer_id=1), RecvGrad(buffer_id=0), BackwardPass(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],  # SWAPPED
                    [SendActivation(buffer_id=2), RecvGrad(buffer_id=1), BackwardPass(buffer_id=1)],  # SWAPPED
                    [SendActivation(buffer_id=3), RecvGrad(buffer_id=2), BackwardPass(buffer_id=2)],
                    [],
                    [RecvGrad(buffer_id=3), BackwardPass(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
            elif self.stage_id == 1:
                yield from [
                    [],
                    [RecvActivation(buffer_id=0), LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [BackwardPass(buffer_id=0)],
                    [RecvActivation(buffer_id=1), SendGrad(buffer_id=0), LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [BackwardPass(buffer_id=1)],
                    [RecvActivation(buffer_id=2), SendGrad(buffer_id=1), LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [BackwardPass(buffer_id=2)],
                    [RecvActivation(buffer_id=3), SendGrad(buffer_id=2), LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [BackwardPass(buffer_id=3)],
                    [SendGrad(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
        elif self.args.num_swaps == 3:
            if self.stage_id == 0:
                yield from [
                    [LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [SendActivation(buffer_id=0)],
                    [LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],
                    [SendActivation(buffer_id=1)],  # Added SendActivation(1)
                    [LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],
                    [RecvGrad(buffer_id=0), BackwardPass(buffer_id=0)],  # Removed SendActivation(1)
                    [SendActivation(buffer_id=2)],  # Added SendActivation(2)
                    [LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [RecvGrad(buffer_id=1), BackwardPass(buffer_id=1)],  # Removed SendActivation(2)
                    [SendActivation(buffer_id=3), RecvGrad(buffer_id=2), BackwardPass(buffer_id=2)],
                    [],
                    [RecvGrad(buffer_id=3), BackwardPass(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
            elif self.stage_id == 1:
                yield from [
                    [],
                    [RecvActivation(buffer_id=0), LoadMicroBatch(buffer_id=0), ForwardPass(buffer_id=0)],
                    [RecvActivation(buffer_id=1), LoadMicroBatch(buffer_id=1), ForwardPass(buffer_id=1)],  # SWAPPED, Removed SendGrad(0)
                    [BackwardPass(buffer_id=0)],  # SWAPPED
                    [SendGrad(buffer_id=0)],  # Added SendGrad(0)
                    [BackwardPass(buffer_id=1)],
                    [RecvActivation(buffer_id=2), LoadMicroBatch(buffer_id=2), ForwardPass(buffer_id=2)],  # Removed SendGrad(1)
                    [SendGrad(buffer_id=1)],   # Added SendGrad(1)
                    [BackwardPass(buffer_id=2)],
                    [RecvActivation(buffer_id=3), SendGrad(buffer_id=2), LoadMicroBatch(buffer_id=3), ForwardPass(buffer_id=3)],
                    [BackwardPass(buffer_id=3)],
                    [SendGrad(buffer_id=3), ReduceTiedGrads(), ReduceGrads(), OptimizerStep()]
                ]
