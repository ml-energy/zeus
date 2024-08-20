# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/engine.py

import time
import logging
import copy
import os

from types import MethodType
from inspect import isgenerator
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from perseus.client.power_state import get_power_controller

from ..utils.logging import logger
from ..utils.timer import SynchronizedWallClockTimer, ThroughputTimer, set_timer_log_rank
from ..utils.merak_args import get_args

from .engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE

from ..modules.module import PipelineModule, PipelineError
from .. import mpu, print_rank_0
from . import schedule
from ..mpu.p2p_communication import recv_forward, send_backward, recv_backward, send_forward, send_forward_recv_backward, send_backward_recv_forward

from .checkpointing import pre_checkpoint as pre_checkpoint_func
from .checkpointing import RNGManager


TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.
    """
    def __init__(self, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        if self.wall_clock_breakdown():
            if mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
                set_timer_log_rank([dist.get_rank()])
            else:
                set_timer_log_rank([])

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        # assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
        #     " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1


        if self.train_schedule == '1f1b':
            TrainScheduleClass = schedule.MergeP2PTrainSchedule
        elif self.train_schedule == 'ds_default':
            TrainScheduleClass = schedule.TrainSchedule
        elif self.train_schedule == "envpipe":
            TrainScheduleClass = schedule.EnvPipeTrainSchedule
        elif self.train_schedule == "swap":
            TrainScheduleClass = schedule.ManualSwapTrainSchedule
        elif self.train_schedule == "instruction_profiler":
            TrainScheduleClass = schedule.ProfileSchedule
        else:
            # if not self.is_last_stage():
            #     assert self.module.activation_checkpoint_interval > 0, 'should use checkpoint layer'
            if self.train_schedule == 'early_recompute_1f1b':
                TrainScheduleClass = schedule.PreRecomputeTrainSchedule
            elif self.train_schedule == 'last_no_recompute_1f1b':
                if self.is_last_stage():
                    self.module.activation_checkpoint_interval = 0
                TrainScheduleClass = schedule.LastNoRecomputeTrainSchedule
            elif self.train_schedule == 'full_critical_path_1f1b':
                if self.is_last_stage():
                    self.module.activation_checkpoint_interval = 0
                TrainScheduleClass = schedule.FullCriticalPathTrainSchedule
                if self.num_stages > 2:
                    self.module.activation_checkpoint_ratio[-2] = self.module.activation_checkpoint_ratio[-3]
                # else:
                #     ## full critical path schedule only support stage > 2 for
                #     TrainScheduleClass = schedule.LastNoRecomputeTrainSchedule
            else:
                raise NotImplementedError(f' train schedule {self.train_schedule} not supported.')

        self.train_sched = TrainScheduleClass(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self.train_power_state_schedule = [0 for _ in chain.from_iterable(self.train_sched)]

        self.activation_checkpoint_interval_backup = self.module.activation_checkpoint_interval

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            next_rank = None
            prev_rank = None
            if 0 <= self.next_stage < self.num_stages:
                next_rank = self.grid.stage_to_global(stage_id=self.next_stage)
            if 0 <= self.prev_stage < self.num_stages:
                prev_rank = self.grid.stage_to_global(stage_id=self.prev_stage)
            mpu.initialize.set_pipeline_model_parallel_next_rank(next_rank)
            mpu.initialize.set_pipeline_model_parallel_prev_rank(prev_rank)



        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'extra_inputs' : [],
            # 'output_tensors' : [], # tensor object to preserve backward graph
            'losses': [],  # Instead of overwriting `self.loss` in _exec_backward_pass for the last stage,
                           # save the losses in the pipe buffer. This way, the last stage can do multiple
                           # forwards before doing any backward. Previously, both DeepSpeed and Merak wrongly
                           # assumed that the last stage will only repeat one-forward-one-backward.
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        self.rng_manager = RNGManager()

        #stores the loss for the current micro batch being processed
        # self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)


        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        # Initialize pipeline communicators. Just send a 0.
        zero = torch.tensor(0.0).to(self.device)
        if is_even(self.stage_id):
            if not self.is_last_stage():
                send_forward(zero)
            if not self.is_first_stage():
                recv_forward(zero)
        else:
            if not self.is_first_stage():
                recv_forward(zero)
            if not self.is_last_stage():
                send_forward(zero)

        # When we're doing instruction profiling, we don't want to free buffers.
        self.is_profiling = get_args().profile

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_tied_allreduce').start()
            self.timers('backward_tied_allreduce').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        self.timers('backward_tied_allreduce').start()
        self.module.allreduce_tied_weight_gradients()
        self.timers('backward_tied_allreduce').stop()

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            self.timers('backward_allreduce').start()
            self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
            self.timers('backward_allreduce').stop()
        self._force_grad_boundary = False

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def train_batch(self, data_iter=None, return_loss: bool = True):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        if self.data_iterator is None and data_iter:
            self.set_dataiterator(data_iter)
        if self.training_dataloader is None and data_iter:
            self.set_dataloader(data_iter)

        self.module.train()
        self.do_train = True
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers('train_batch').start()

        
        self._exec_schedule(self.train_sched, self.train_power_state_schedule)
        if not return_loss:
            self.timers('train_batch').stop()
            return
        else:
            self.agg_train_loss = self._aggregate_total_loss()
            self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True)
                # print(self.train_batch_size(), elapsed)
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'total samples/sec: {tput:0.3f}')

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_train_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.print_wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0 and mpu.get_data_parallel_rank()==0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad', 
                'pipe_send_grad_recv_input',
                'pipe_send_output_recv_grad',
            ], normalizer=self.steps_per_print())

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self, data_iter, compute_loss=True, reduce_output='avg'):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()
        self.do_train = False

        eval_output = None
        if self.return_logits:
            self.logits = []
            self.labels = []

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)
        self.set_dataloader(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        power_state_schedule = [0 for _ in chain.from_iterable(sched)]
        with torch.no_grad():
            self._exec_schedule(sched, power_state_schedule)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/eval_loss',
                                        eval_output.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                self.summary_writer.flush()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)
        if self.return_logits and self.is_last_stage():
            return eval_output, self.logits, self.labels
        else:
            return eval_output, None, None

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        # if self.is_first_stage() or self.is_last_stage():
        self.training_dataloader = loader
        # self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        self.training_dataloader = None
        self.data_iterator = iterator

    def reset_dataiterator(self, iterator):
        del self.data_iterator
        self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        batch = None
        if self.data_iterator is not None:
            if not isgenerator(self.data_iterator):
                self.data_iterator = iter(self.data_iterator)

            batch = next(self.data_iterator)

        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward').start()

        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()


        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        if self.pipe_buffers['extra_inputs'][buffer_id] is not None:
            if isinstance(self.pipe_buffers['extra_inputs'][buffer_id], tuple):
                extra_inputs = tuple(t.clone() for t in self.pipe_buffers['extra_inputs'][buffer_id])
            else:
                extra_inputs = self.pipe_buffers['extra_inputs'][buffer_id].clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        self._zero_grads(inputs)

        outputs = super().forward(inputs)
        
        if isinstance(outputs, tuple) and len(outputs) == 1:
            outputs = outputs[0]
        self.pipe_buffers['outputs'][buffer_id] = outputs
        # torch.save(outputs, f"{self.stage_id}+activations.torch")

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.loss_model is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                # print(outputs.shape, labels.shape)
                # self.loss = self.loss_model(outputs, labels)
                self.pipe_buffers["losses"][buffer_id] = self.loss_model(outputs, labels)
                if self.return_logits and not self.do_train:
                    self.logits.append(outputs)
                    self.labels.append(labels)
            else:
                # Some models just return loss from forward()
                # self.loss = outputs
                self.pipe_buffers["losses"][buffer_id] = outputs

            loss = self.pipe_buffers["losses"][buffer_id]
            if isinstance(loss, torch.Tensor):
                self.fwd_outputs.append(loss.detach())

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(loss)
                self.total_loss += loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in loss]
                for idx, l in enumerate(loss):
                    self.total_loss[idx] += l.detach()

        if self.wall_clock_breakdown():
            self.timers('forward').stop()
            self.timers('forward_microstep').stop()

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.pipe_buffers["losses"][buffer_id])
            # Free up the output and loss buffers since we're done with backward.
            if not self.is_profiling:
                self.pipe_buffers["losses"][buffer_id] = None
                self.pipe_buffers["outputs"][buffer_id] = None
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()


        grad_tensors = self.grad_layer
 
        # try:
        #     print('output&grad', self.stage_id, outputs, grad_tensors.shape)
        # except:
        #     print('output&grad', self.stage_id, [i.shape for i in outputs], [i.shape for i in grad_tensors])
        if  self.wall_clock_breakdown():
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.requires_grad]
            if not len(out_tensors) == len(grad_tensors):
                grad_tensors = [grad_tensors[t] for t in range(len(outputs)) if outputs[t].requires_grad]
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors, retain_graph=self.is_profiling)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ), retain_graph=self.is_profiling)

        # Free up the memory from the output of forward()
        if not self.is_profiling:
            self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = []
            load_idx = len(self.input_to_stage_dic[0])
            for x in batch[:load_idx]:
                assert torch.is_tensor(x)
                mine = x.clone().detach().to(self.device)
                mine.requires_grad = mine.is_floating_point()
                loaded.append(mine)
            self.pipe_buffers['inputs'][buffer_id] = tuple(loaded)

        if self.is_last_stage():
            loaded = []

            # 计算input_to_stage_dic所需的输入数量
            load_idx = sum([len(self.input_to_stage_dic[i]) for i in self.input_to_stage_dic])
            # 如果batch的数据少于load_idx，则是使用了split input，就不需要再做划分
            if load_idx < len(batch):
                # 去掉last stage不需要的部分
                batch = batch[load_idx:]

            for x in batch:
                assert torch.is_tensor(x)
                x = x.to(self.device).detach()
                loaded.append(x)
            loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded


        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            send_forward(type_tensor)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            send_forward(send_ndims)
            send_forward(send_shape)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            send_forward(type_tensor)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            send_forward(count_tensor)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_forward(send_ndims)
                send_forward(send_shape)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        recv_forward(type_tensor)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            recv_forward(recv_ndims)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            recv_forward(recv_shape)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            recv_forward(count_tensor)
            num_tensors = count_tensor.item()
            recv_shapes = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                recv_forward(recv_ndims)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                recv_forward(recv_shape)
                recv_shapes.append(recv_shape.tolist())

            buffers = self._allocate_buffers(recv_shapes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            if not self.is_last_stage() and outputs.dtype != self.communication_data_type():
                outputs = outputs.to(self.communication_data_type())
            send_forward(outputs)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                if not self.is_last_stage() and buffer.dtype != self.communication_data_type():
                    buffer = buffer.to(self.communication_data_type())
                send_forward(buffer)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')


        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # results = []

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            send_backward(inputs.grad)
            # results.append(inputs.grad)
        else:
            for idx, buffer in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not buffer.is_floating_point():
                    assert buffer.grad is None
                    continue
                assert buffer.grad is not None
                send_backward(buffer.grad)
                # results.append(buffer.grad)
        
        # torch.save(results, f"{self.stage_id}+grads.torch")

        # We can free up the input buffer now
        if not self.is_profiling:
            self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        if not self.is_first_stage() and not self.is_last_stage():
            extra = []
            # Obtain previous stage, as input_to_stage_dic index
            # 获取之前stage，在input_to_stage_dic中的输入数
            load_idx = sum([len(self.input_to_stage_dic[i]) for i in range(self.stage_id)])
            for k, v in self.input_to_stage_dic.items():
                # Match the stages in the dic, to avoid empty list
                # 匹配dic中出现的stage，并防止出现空list的情况
                if k == self.stage_id and len(v):
                    o_input = self._next_batch()
                    if len(v) == len(o_input):
                        load_idx = 0
                    start_idx = load_idx
                    end_idx = load_idx + len(v)
                    for idx in range(start_idx, end_idx):
                        x = o_input[idx]
                        assert torch.is_tensor(x)
                        mine = x.clone().detach().to(self.device)
                        mine.requires_grad = mine.is_floating_point()
                        extra.append(mine)
                    self.pipe_buffers['extra_inputs'][buffer_id] = tuple(extra)
        
        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()
            self.timers('pipe_recv_input').start()

        recvd = None

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            recv_forward(self.pipe_recv_buf)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                recv_forward(buffer)
                recvd[idx] = buffer.clone().detach()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd 

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes, num_buffers=1)[0]
        # print('before', self.global_rank, self.grad_layer)
        if isinstance(self.grad_layer, torch.Tensor):
            recv_backward(self.grad_layer)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                recv_backward(buffer)
        # print('after', self.global_rank, self.grad_layer)
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()

        if self.print_wall_clock_breakdown():
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'step_microstep'
                ], normalizer=self.steps_per_print())
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'backward_tied_allreduce',
                    'step'
                ], normalizer=self.steps_per_print())

    def _exec_send_activations_recv_grads(self, buffer_id):

        send_buffer, recv_buffer=buffer_id

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output_recv_grad').start()

        activations = self.pipe_buffers['outputs'][send_buffer]

        outputs = self.pipe_buffers['outputs'][recv_buffer]

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes, num_buffers=1)[0]
        # print('before', self.global_rank, self.grad_layer)
        if isinstance(self.grad_layer, torch.Tensor):
            if not self.is_last_stage() and activations.dtype != self.communication_data_type():
                activations = activations.to(self.communication_data_type())
            send_forward_recv_backward(activations, self.grad_layer)
        else:
            assert isinstance(outputs, tuple)
            activations_list = []
            for idx, values in enumerate(activations):
                activations_list.append(values.to(self.communication_data_type()))
            activations = tuple(activations_list)
            for idx, buffer in enumerate(self.grad_layer):
                send_forward_recv_backward(activations[idx], buffer)

        # print('after', self.global_rank, self.grad_layer)
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output_recv_grad').stop()
    
    def _exec_send_grads_recv_activations(self, buffer_id):
        
        send_buffer, recv_buffer = buffer_id
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad_recv_input').start()

        inputs = self.pipe_buffers['inputs'][send_buffer]
        recvd = None
    
        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
        if isinstance(self.pipe_recv_buf, torch.Tensor):
            send_backward_recv_forward(inputs.grad, self.pipe_recv_buf)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            # print(len(inputs), len(self.pipe_recv_buf))
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                send_backward_recv_forward(inputs[idx].grad, buffer)
                recvd[idx] = buffer.clone().detach()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][send_buffer] = None
        self.pipe_buffers['inputs'][recv_buffer] = recvd

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad_recv_input').stop()
            self.timers('batch_input').start()

        if not self.is_first_stage() and not self.is_last_stage():
            extra = []
            load_idx = sum([len(self.input_to_stage_dic[i]) for i in range(self.stage_id)])
            for k, v in self.input_to_stage_dic.items():
                if k == self.stage_id and len(v):
                    o_input = self._next_batch()
                    if len(v) == len(o_input):
                        load_idx = 0
                    start_idx = load_idx
                    end_idx = load_idx + len(v)
                    for idx in range(start_idx, end_idx):
                        x = o_input[idx]
                        assert torch.is_tensor(x)
                        mine = x.clone().detach().to(self.device)
                        mine.requires_grad = mine.is_floating_point()
                        extra.append(mine)
                    self.pipe_buffers['extra_inputs'][recv_buffer] = tuple(extra)

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()
 

    def _exec_precheckpoint_forward_pass(self, buffer_id):

        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if self.module.activation_checkpoint_func is not pre_checkpoint_func:
            self.checkpoint_func_bak = self.module.activation_checkpoint_func
            self.module.activation_checkpoint_func = pre_checkpoint_func

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        assert not self.is_last_stage(), 'last stage should not pre checkpoint recompute'


        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        if self.pipe_buffers['extra_inputs'][buffer_id] is not None:
            if isinstance(self.pipe_buffers['extra_inputs'][buffer_id], tuple):
                extra_inputs = tuple(t.clone() for t in self.pipe_buffers['extra_inputs'][buffer_id])
            else:
                extra_inputs = self.pipe_buffers['extra_inputs'][buffer_id].clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        self._zero_grads(inputs)
    
        self.rng_manager.store_fwd_rng_state(buffer_id)
        outputs = super().forward(inputs)        
        if isinstance(outputs, tuple) and len(outputs) == 1:
            outputs = outputs[0]
        self.pipe_buffers['outputs'][buffer_id] = outputs

    def _exec_recompute_recv_grads(self, buffer_id):


        if self.module.activation_checkpoint_func is pre_checkpoint_func:
            self.module.activation_checkpoint_func = self.checkpoint_func_bak
        if self.module.activation_checkpoint_interval != 0:
            self.module.activation_checkpoint_interval = 0


        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        if self.pipe_buffers['extra_inputs'][buffer_id] is not None:
            if isinstance(self.pipe_buffers['extra_inputs'][buffer_id], tuple):
                extra_inputs = tuple(t.clone() for t in self.pipe_buffers['extra_inputs'][buffer_id])
            else:
                extra_inputs = self.pipe_buffers['extra_inputs'][buffer_id].clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        self.rng_manager.set_recompute_rng_state(buffer_id)
        # do recomputation
        outputs = super().forward(inputs)

        if isinstance(outputs, tuple) and len(outputs) == 1:
            outputs = outputs[0]
        self.pipe_buffers['outputs'][buffer_id] = outputs

        self.rng_manager.restore_bwd_rng_state(buffer_id)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, num_buffers=1)[0]
            else:
                sizes = [list(t.size()) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes, num_buffers=1)[0]
        if isinstance(self.grad_layer, torch.Tensor):
            recv_backward(self.grad_layer)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                recv_backward(buffer)
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_restore_recompute_status(self):
        if self.module.activation_checkpoint_func is pre_checkpoint_func:
            self.module.activation_checkpoint_func = self.checkpoint_func_bak
        if self.module.activation_checkpoint_interval == 0:
            self.module.activation_checkpoint_interval = self.activation_checkpoint_interval_backup

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            fp16 (bool): whether to use FP16. default: defer to self.fp16_enabled()
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs and self.fp16_enabled():
            kwargs["dtype"] = torch.half

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape in shapes:
                buffer.append(self._allocate_zeros(shape, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            #return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        torch.cuda.synchronize()

        if reset_max:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS',
            msg,
            f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
            f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
        )


    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        schedule.SendActivationRecvGrad: _exec_send_activations_recv_grads,
        schedule.SendGradRecvActivation: _exec_send_grads_recv_activations,
                
        schedule.PreCheckpointForwardPass: _exec_precheckpoint_forward_pass,
        schedule.RecomputeRecvGrad: _exec_recompute_recv_grads,
        schedule.RestoreRecomputeStatus: _exec_restore_recompute_status,
    }

    def _exec_schedule(
        self,
        pipe_schedule: schedule.PipeSchedule,
        power_state_schedule: list[int],
    ):
        # Reserve and reset buffers.
        if hasattr(pipe_schedule, "num_pipe_buffers"):
            self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        # Profiling mode with `ProfileSchedule`
        else:
            self._reserve_pipe_buffers(1)
        self.fwd_outputs = []

        # For each step in the schedule
        power_controller = get_power_controller()
        for cmd, power_state in zip(chain.from_iterable(pipe_schedule), power_state_schedule):
            # print(self.global_rank, cmd, power_state)
            # For each instruction in the step
            # for cmd in step_cmds:
                # if type(cmd) not in self._INSTRUCTION_MAP:
                #     raise RuntimeError(
                #         f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                #     )

            # Set GPU power state
            power_controller.set(power_state)

            # Equivalent to: self._exec_forward_pass(buffer_id=0)
            # self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
            # self._exec_instr(**cmd.kwargs)
            self._INSTRUCTION_MAP[type(cmd)](self, **cmd.kwargs)

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn
