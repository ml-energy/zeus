'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

# https://github.com/microsoft/DeepSpeed/blob/85ce85dd5f4b18c0019a5121b06900e3a2c3933b/deepspeed/utils/timer.py

import time
import pynvml
import multiprocessing as mp

import torch
import torch.distributed as dist
from torch.profiler import profile, record_function

from .logging import log_dist
from ..utils.merak_args import get_args

from . import logger

try:
    import psutil
    PSUTILS_INSTALLED = True
except ImportError:
    PSUTILS_INSTALLED = False
    pass


LOG_RANKS = [0]

def set_timer_log_rank(ranks=[0]):
    global LOG_RANKS
    LOG_RANKS = ranks

class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(
            self,
            name: str,
            device_idx: int,
            export_pytorch_trace: bool,
            export_timing_csv: bool,
            energy_breakdown: bool,
        ):
            self.name_ = name
            self.export_pytorch_trace = export_pytorch_trace
            self.export_timing_csv = export_timing_csv
            self.energy_breakdown = energy_breakdown
            self.elapsed_ = 0.0
            self.elapsed_single = []  # records the computation time of every instruction in all iterations
            self.started_ = False
            self.start_time = time.time()
            self.msg = f"{self.name_}-{dist.get_rank()}"
            self.start_timings = []
            self.end_timings = []

            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.start_energy = 0
            self.energy_consumed = []

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            if self.export_pytorch_trace:
                self.record = record_function(self.msg).__enter__()
            self.start_time = time.time()
            if self.export_timing_csv:
                self.start_timings.append(self.start_time)
            if self.energy_breakdown:
                if self.name_ in ["forward_microstep", "backward_microstep"]:
                    self.start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle) / 1000.0
            self.started_ = True

        def stop(self, reset=False):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            if self.export_pytorch_trace:
                self.record.__exit__(None, None, None)
            end_time = time.time()
            if self.export_timing_csv:
                self.end_timings.append(end_time)
            if self.energy_breakdown:
                if self.name_ in ["forward_microstep", "backward_microstep"]:
                    end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle) / 1000.0
                    self.energy_consumed.append(end_energy - self.start_energy)
            self.elapsed_single.append(end_time - self.start_time)
            if reset:
                self.elapsed_ = self.elapsed_single[-1]
            else:
                self.elapsed_ += self.elapsed_single[-1]
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.elapsed_single = []
            self.started_ = False
            self.energy_consumed = []

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self, device_idx: int):
        self.timers: dict[str, SynchronizedWallClockTimer.Timer] = {}
        self.device_idx = device_idx
        self.export_pytorch_trace = get_args().export_pytorch_trace
        self.export_timing_csv = get_args().export_timing_csv
        self.energy_breakdown = get_args().energy_breakdown
        if self.export_pytorch_trace:
            self.profiler = profile()
            self.profiler.__enter__()
        if self.export_timing_csv:
            self.channel = mp.SimpleQueue()
            self.proc = mp.Process(target=energy_polling_process, args=(device_idx, self.channel))
            self.proc.start()
            time.sleep(1)

    def end(self):
        if self.export_pytorch_trace:
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(f"{get_args().logging_dir}/{dist.get_rank()}.trace.json")
        if self.export_timing_csv:
            self.channel.put("end")
            with open(f"{get_args().logging_dir}/instructions-{dist.get_rank()}.csv", "w") as f:
                f.write("instruction,start,end\n")
                for timer in self.timers.values():
                    if not timer.export_timing_csv:
                        continue
                    name = timer.name_
                    for start, end in zip(timer.start_timings, timer.end_timings):
                        f.write(f"{name},{start},{end}\n")
            self.proc.join()

    def __call__(self, name):
        if name not in self.timers:
            export_timing_csv = self.export_timing_csv and name in ["batch_input", "forward_microstep", "backward_microstep"]
            self.timers[name] = self.Timer(
                name,
                self.device_idx,
                self.export_pytorch_trace,
                export_timing_csv,
                self.energy_breakdown,
            )
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(torch.cuda.memory_allocated() /
                                                  (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(torch.cuda.memory_cached() /
                                                    (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, ranks=None):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = f'rank={torch.distributed.get_rank()} time (ms)'
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(
                    reset=reset) * 1000.0 / normalizer
                string += ' | {}: {:.2f}'.format(name, elapsed_time)

        log_dist(string, ranks=ranks if ranks else LOG_RANKS)


def energy_polling_process(device_idx: int, channel: mp.SimpleQueue) -> None:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    timings = []
    energy_consumed = []
    prev_energy = -1.0
    i = 0
    while True:
        timing = time.time()
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle) / 1000.0
        if prev_energy != energy:
            timings.append(timing)
            energy_consumed.append(energy)
            prev_energy = energy

        i += 1
        if i % 100 == 0 and not channel.empty():
            break

    pynvml.nvmlShutdown()

    with open(f"{get_args().logging_dir}/time-energy-{dist.get_rank()}.csv", "w") as f:
        f.write("time,energy\n")
        for timing, energy in zip(timings, energy_consumed):
            f.write(f"{timing},{energy}\n")


class ThroughputTimer():
    def __init__(self,
                 batch_size,
                 num_workers,
                 start_step=2,
                 steps_per_output=50,
                 monitor_memory=False,
                 logging_fn=None):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = 1
        self.num_workers = num_workers
        self.start_step = start_step
        self.epoch_count = 0
        self.local_step_count = 0
        self.total_step_count = 0
        self.total_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            self.logging = logger.info
        self.initialized = False

        if self.monitor_memory and not PSUTILS_INSTALLED:
            raise ImportError("Unable to import 'psutils', please install package")

    def update_epoch_count(self):
        self.epoch_count += 1
        self.local_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        self._init_timer()
        self.started = True
        if self.total_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()

    def stop(self, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.total_step_count += 1
        self.local_step_count += 1
        if self.total_step_count > self.start_step:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            if self.local_step_count % self.steps_per_output == 0:
                if report_speed:
                    self.logging("{}/{}, SamplesPerSec={}".format(
                        self.epoch_count,
                        self.local_step_count,
                        self.avg_samples_per_sec()))
                if self.monitor_memory:
                    virt_mem = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    self.logging("{}/{}, vm percent: {}, swap percent: {}".format(
                        self.epoch_count,
                        self.local_step_count,
                        virt_mem.percent,
                        swap.percent))

    def avg_samples_per_sec(self):
        if self.total_step_count > 0:
            samples_per_step = self.batch_size * self.num_workers
            total_step_offset = self.total_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            # training samples per second
            return samples_per_step / avg_time_per_step
        return float("-inf")
