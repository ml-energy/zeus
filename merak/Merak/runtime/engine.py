'''
Copyright 2019 The Microsoft DeepSpeed Team
'''
# https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/engine.py

import os
import stat
import math
import torch
import warnings
import hashlib
import torch.distributed as dist
from collections import OrderedDict
from shutil import copyfile

from torch.nn.modules import Module
from torch.distributed.distributed_c10d import _get_global_rank
from tensorboardX import SummaryWriter

from .. import mpu
from .utils import see_memory_usage, clip_grad_norm_
from .config import DeepSpeedConfig
from ..utils import logger, log_dist
from ..utils.timer import ThroughputTimer, SynchronizedWallClockTimer
from ..utils.merak_args import get_args
from ..modules.module import PipelineModule
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from ..utils.fp16_optimizer import FP16_Optimizer

try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # will using torch.cuda.amp
    APEX_INSTALLED = False
# from torch.cuda.amp import autocast
# from torch.cuda.amp.grad_scaler import GradScaler

version = "0.0.0"

MEMORY_OPT_ALLREDUCE_SIZE = 500000000


def split_half_float_double(tensors):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


def print_configuration(args, name):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(args)):
        dots = '.' * (29 - len(arg))
        logger.info('  {} {} {}'.format(arg, dots, getattr(args, arg)))

class DeepSpeedEngine(Module):
    r"""DeepSpeed engine for training.
    """
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_params=None,
                 dont_change_device=False,
                 train_schedule='1f1b',
                 return_logits=False):
        super(DeepSpeedEngine, self).__init__()
        self.args = args
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_model_parameters = model_parameters
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        self.enable_backward_allreduce = True
        self.train_schedule = train_schedule
        self.gas_boundary_ctr = 0
        self.dist_backend = "nccl"
        self.return_logits = return_logits


        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}

        # Set config using config_params for backwards compat
        if self.config is None and config_params is not None:
            self.config = config_params

        if dist_init_required is None:
            dist_init_required = not dist.is_initialized()

        if dist_init_required is False:
            assert dist.is_initialized() is True, "Torch distributed not initialized. Please set dist_init_required to True or initialize before calling deepspeed.initialize()"
        
        see_memory_usage(f"DeepSpeed Engine: Before args sanity test")
        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)

        self._set_distributed_vars()

        if self.tensorboard_enabled() and self.global_rank == 0:
            self.summary_writer = self.get_summary_writer()

        see_memory_usage(f"DeepSpeed Engine: Before configure distributed model")

        # Configure distributed model
        self._configure_distributed_model(model)

        self.pipeline_parallelism = isinstance(self.module, PipelineModule)

        see_memory_usage(f"DeepSpeed Engine: After configure distributed model")

        # Configure wall clock timer
        self.timers = SynchronizedWallClockTimer(self.local_rank)

        # Throughput timer
        self.tput_timer = ThroughputTimer(
            batch_size=self.train_micro_batch_size_per_gpu(),
            num_workers=self.dp_world_size,
            steps_per_output=self.steps_per_print(),
            monitor_memory=False)

        self.training_dataloader = None

        # Configure optimizer and scheduler
        if self.amp_enabled() or self.fp16_enabled():
            # if self.args.half_precision_backend == "amp":
            #     assert mpu.get_pipe_parallel_world_size() <= 1 and self.mp_world_size <= 1, "Currently not support model parallelism with native amp"
            #     self.scaler = GradScaler()
            self._configure_optimizer(optimizer, model_parameters)
            self.lr_scheduler = lr_scheduler
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        if self.global_rank == 0:
            self._config.print('DeepSpeedEngine configuration')
            if self.dump_state():
                print_configuration(self, 'DeepSpeedEngine')

        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

    def tensorboard_enabled(self):
        return self._config.tensorboard_enabled

    def tensorboard_output_path(self):
        return self._config.tensorboard_output_path

    def tensorboard_job_name(self):
        return self._config.tensorboard_job_name

    def get_summary_writer(self,
                           name="DeepSpeedJobName",
                           base=os.path.join(os.path.expanduser("~"),
                                             "tensorboard")):
        if self.tensorboard_output_path():
            base_dir = self.tensorboard_output_path()
            job_name = self.tensorboard_job_name()
            log_dir = os.path.join(base_dir, job_name)
        else:
            if self.tensorboard_job_name():
                name = self.tensorboard_job_name()

            # Infrastructure-specific job-id
            if 'DLWS_JOB_ID' in os.environ:
                infra_job_id = os.environ['DLWS_JOB_ID']
            elif 'DLTS_JOB_ID' in os.environ:
                infra_job_id = os.environ['DLTS_JOB_ID']
            else:
                infra_job_id = 'unknown-job-id'

            summary_writer_dir_name = os.path.join(infra_job_id, "logs")
            log_dir = os.path.join(base, summary_writer_dir_name, name)

        os.makedirs(log_dir, exist_ok=True)

        return SummaryWriter(log_dir=log_dir)

    def fp16_enabled(self):
        return self._config.fp16_enabled

    # def bfloat16_enabled(self):
    #     return self._config.bfloat16_enabled

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def communication_data_type(self):
        if self.fp16_enabled():
            return torch.float16

        return torch.float32

    def print_wall_clock_breakdown(self):
        return self._config.print_wall_clock_breakdown

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def dump_state(self):
        return self._config.dump_state

    def gradient_clipping(self):
        return self._config.gradient_clipping

    def dynamic_loss_scale(self):
        return self._config.loss_scale == 0

    def initial_dynamic_scale(self):
        return self._config.initial_dynamic_scale

    def dynamic_loss_scale_args(self):
        return self._config.dynamic_loss_scale_args

    def loss_scale(self):
        return self._config.loss_scale


    def _set_distributed_vars(self):
        if self.local_rank >= 0:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatability with scripts relying on [args|self].local_rank containing
        # the correct local rank info. _do_args_sanity_check will ensure this is the case.

        self.local_rank = int(os.environ['LOCAL_RANK'])
        if hasattr(args, 'local_rank'):
            args.local_rank = self.local_rank
        if self.config is None:
            self.config = args.deepspeed_config if hasattr(args,
                                                           'deepspeed_config') else None
        self._config = DeepSpeedConfig(self.config, mpu)

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        if hasattr(args, 'deepscale_config') and args.deepscale_config is not None:
            logger.warning(
                "************ --deepscale_config is deprecated, please use --deepspeed_config ************"
            )
            if hasattr(args, 'deepspeed_config'):
                assert args.deepspeed_config is None, "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
            args.deepspeed_config = args.deepscale_config

        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch.distributed launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank != None:
            assert isinstance(args.local_rank, int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert env_local_rank == args.local_rank, \
                    f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

        if self.config is None:
            assert hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None, \
                'DeepSpeed requires --deepspeed_config to specify configuration file'

            assert os.path.isfile(args.deepspeed_config), \
                'DeepSpeed configuration file: {} is not an existing file'.format(args.deepspeed_config)

    def _broadcast_model(self):

        def is_replicated(p):
            if hasattr(p, 'ds_status'):
                return False
            return True

        for p in self.module.parameters():
            if torch.is_tensor(p) and is_replicated(p):
                dist.broadcast(p,
                               self.broadcast_src_rank,
                               group=self.data_parallel_group)

    def _configure_distributed_model(self, model):
        self.module = model

        if self.fp16_enabled():
            self.module.half()
            if not all(
                    [param.dtype == torch.half for param in self.module.parameters()]):
                    names = [
                        n for n,
                        p in self.module.named_parameters() if p.dtype != torch.half
                    ]
                    raise ValueError(
                        f"fp16 is enabled but the following parameters have dtype that is not fp16: {', '.join(names)}"
                    )
        # elif self.bfloat16_enabled():
        #     self.module.bfloat16()
        else:
            if not all(
                [param.dtype == torch.float for param in self.module.parameters()]):
                names = [
                    n for n,
                    p in self.module.named_parameters() if p.dtype != torch.float
                ]
                raise ValueError(
                    f"fp32 is enabled but the following parameters have dtype that is not fp32: {', '.join(names)}"
                )

        if not self.dont_change_device:
            self.module.to(self.device)

        if self.mpu is None:
            assert False, 'only support pipeline engine'
        else:
            self.data_parallel_group = self.mpu.get_data_parallel_group()
            self.dp_world_size = self.mpu.get_data_parallel_world_size()
            self.mp_world_size = self.mpu.get_model_parallel_world_size()
            self.broadcast_src_rank = _get_global_rank(
                self.mpu.get_data_parallel_group(),
                0)
        if not self.args.half_precision_backend == "apex":
            self._broadcast_model()

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if isinstance(client_optimizer, torch.optim.Optimizer):
            client_optimizer.param_groups[:] = [
                pg for pg in client_optimizer.param_groups if len(pg["params"]) != 0
            ]
            if self.global_rank == 0:
                logger.info(
                    "Removing param_group that has no 'params' in the client Optimizer"
                )

            basic_optimizer = client_optimizer
            if self.global_rank == 0:
                logger.info('Using client Optimizer as basic optimizer')
        else:
            basic_optimizer = client_optimizer(model_parameters)
            if self.global_rank == 0:
                logger.info('Using client callable to create basic optimizer')

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        if self.global_rank == 0:
            logger.info("Basic Optimizer = {}".format(
                basic_optimizer.__class__.__name__))

        if self.amp_enabled() and self.args.half_precision_backend == "apex":
            assert not self.fp16_enabled(), "Cannot enable both amp with (legacy) fp16 mode"
            amp_params = self.amp_params()
            if self.global_rank == 0:
                logger.info(f"Initializing AMP with these params: {amp_params}")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError(
                    "Unable to import apex/amp, please make sure it is installed")
            self.module, self.optimizer = amp.initialize(
                self.module, basic_optimizer, **amp_params
            )
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif self.fp16_enabled():
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer
            log_dist("Final Optimizer = {}".format(self.client_optimizer.__class__.__name__),
                    ranks=[0])

        self.quantizer = None

    def _configure_fp16_optimizer(self, optimizer):
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()
        if self.dynamic_loss_scale():
            log_dist("Creating fp16 unfused optimizer with dynamic loss scale",
                        ranks=[0])
            optimizer = FP16_Optimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=optimizer.__class__.__name__ == "lamb",
            )

        return optimizer

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id)
                if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behaviour."



    @staticmethod
    def is_map_style_dataset(obj):
        return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

    @staticmethod
    def is_iterable_style_dataset(obj):
        return isinstance(obj,
                          torch.utils.data.IterableDataset
                          )  # hasattr(obj, "__iter__") should work as well

    def train(self, mode=True):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train(mode)

    def eval(self):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train(False)

    def _scale_loss_by_gas(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.gradient_accumulation_steps()
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.gradient_accumulation_steps())
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(
                    f'DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}'
                )
                self.warn_unscaled_loss = False

        return scaled_loss

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """

        if self.training_dataloader is None:
            self.tput_timer.start()

        # onnx graph node test
        # if mpu.get_data_parallel_rank() == 0:
        #     torch.onnx.export(model=self.module,
        #                       args=inputs,
        #                       f="/dat/txacs/merak-final/merak/examples/language-modeling/test.onnx",
        #                       verbose=True,
        #                       export_params=True,
        #                       do_constant_folding=False)
        # torch.distributed.barrier()
        # os._exit(0)

        # if self.amp_enabled() and self.args.half_precision_backend == "amp":
        #     with autocast(dtype=torch.float16):
        #         loss = self.module(*inputs, **kwargs)
        # else:
        loss = self.module(*inputs, **kwargs)


        return loss

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
 
        if self.is_gradient_accumulation_boundary():
            self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def backward(self, loss, allreduce_gradients=True, release_loss=False):
        r"""Execute backward pass on the loss

        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
        """

        if not allreduce_gradients:
            logger.warning(
                f'Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed'
            )

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1:
            loss = self._scale_loss_by_gas(loss.float())

        # Log training Loss
        if self.tensorboard_enabled():
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [
                        (f'Train/Samples/train_loss',
                         loss.mean().item() * self.gradient_accumulation_steps(),
                         self.global_samples)
                    ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        if self.wall_clock_breakdown():
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        if self.amp_enabled() and self.args.half_precision_backend == "apex":
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss,
                                self.optimizer,
                                delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
        # elif self.amp_enabled() and self.args.half_precision_backend == "amp":
        #     if not self.is_gradient_accumulation_boundary():
        #         self.scaler.scale(loss).backward()
        elif self.fp16_enabled():
            self.optimizer.backward(loss, retain_graph=get_args().profile)
        else:
            loss.backward(retain_graph=get_args().profile)

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()

        if self.wall_clock_breakdown():
            self.timers('backward_allreduce').start()

        if self.enable_backward_allreduce:
            self.allreduce_gradients()

        if self.wall_clock_breakdown():
            self.timers('backward_allreduce').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        if release_loss:
            # loss.data = None
            pass

        return loss

    def is_gradient_accumulation_boundary(self):
        """Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.
        """
        return (self.micro_steps + 1) % \
            self.gradient_accumulation_steps() == 0

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        torch.nn.utils.clip_grad_norm_(parameters=self.module.parameters(),
                                       max_norm=self.gradient_clipping())

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        if self.gradient_clipping() > 0.0:
            if not (self.fp16_enabled() or self.amp_enabled()):
                self.clip_fp32_gradients()
            elif self.amp_enabled() and self.args.half_precision_backend == "apex":
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params,
                                max_norm=self.gradient_clipping(),
                                mpu=self.mpu)
            # elif self.amp_enabled() and self.args.half_precision_backend == "amp":
            #     if hasattr(self.scaler, "_scale") and self.scaler._scale is not None:
            #         self.scaler.unscale_(self.optimizer)
            #         torch.nn.utils.clip_grad_norm_(parameters=self.module.parameters(),
            #                                 max_norm=self.args.max_grad_norm)
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #         self.lr_scheduler.step(**(lr_kwargs or {}))
        self.optimizer.step()

        # Modified 
        if (not self.fp16_enabled()
                and not self.amp_enabled()):
            self.zero_grad()
        else:
            self.optimizer.zero_grad()

        report_progress = self.global_rank == 0 if self.global_rank else True

        # Check overlow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, 'overflow'):
            overflow = self.optimizer.overflow

        if overflow:
            self.skipped_steps += 1
        else:
            if self.lr_scheduler is not None:
                    self.lr_scheduler.step(**(lr_kwargs or {}))

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

        if self.wall_clock_breakdown() and (self.global_steps + 1) % self.steps_per_print() == 0:
            should_log = (mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0)
            see_memory_usage('After {} iterations'.format(self.global_steps + 1), should_log, ranks=[dist.get_rank()])

        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def step(self, lr_kwargs=None):
        "rewrite in pipe_engine"
        pass

    def _get_optimizer_param(self, param_name):
        result = []
        if not self.optimizer:
            return result
        for group in self.optimizer.param_groups:
            if param_name in group:
                result.append(group[param_name])
            else:
                result.append(0.0)
        return result

    def get_lr(self):
        return self._get_optimizer_param('lr')


    def _report_progress(self, step):
        lr = self.get_lr()
        log_dist(f'step={step}, skipped={self.skipped_steps}, lr={lr}',
                 ranks=[0])

    def allreduce_bucket(self, bucket):
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if self.communication_data_type() != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type())

        if self.postscale_gradients():
            if self.gradient_predivide_factor() != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor())

            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

            if self.gradient_average:
                if self.gradient_predivide_factor() != self.dp_world_size:
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() /
                                             self.dp_world_size)
        else:
            tensor_to_allreduce.div_(self.dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

        if self.communication_data_type() != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket):
        allreduced = self.allreduce_bucket(small_bucket)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        grads = []
        for param_name, param in self.module.named_parameters():
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.device)
                grads.append(param.grad.data)
            else:
                grad_data = param.grad.data
                grads.append(grad_data)

        split_buckets = split_half_float_double(grads)

        for i, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer)


