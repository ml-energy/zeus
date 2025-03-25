"""Pipeline frequency optimizer implementation.

The `PipelineFrequencyOptimizer` is to be integrated into the training framework.
It is responsible for communicating with the PFO server and managing
the `FrequencyController` instance, which is responsible for controlling
the frequency of the CPU of the current process.
"""

from __future__ import annotations

import httpx
import torch
import torch.distributed as dist

from zeus.callback import Callback
from zeus.device import get_gpus
from zeus.optimizer.pipeline_frequency.frequency_controller import FrequencyController
from zeus.optimizer.pipeline_frequency.common import (
    GET_FREQUENCY_SCHEDULE_URL,
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    REPORT_TIMING_URL,
    REPORT_ENERGY_URL,
    JobInfo,
    RankInfo,
    FrequencySchedule,
)
from zeus.utils.framework import sync_execution
from zeus.monitor import ZeusMonitor


class PipelineFrequencyOptimizer(Callback):
    """Pipeline frequency optimizer."""

    def __init__(
        self,
        rank: int,
        dp_rank: int,
        pp_rank: int,
        tp_rank: int,
        device_id: int,
        dp_degree: int,
        pp_degree: int,
        tp_degree: int,
        world_size: int,
        server_url: str,
        job_metadata: str | None = None,
    ) -> None:
        """Initialize the Pipeline frequency optimizer.

        Assumptions:
            - `torch.distributed` has been initialized.
            - `torch.cuda.set_device` has been called with `device_id`.
                This is needed to broadcast the job ID to all ranks.

        The master process (rank 0) will register the job with the Peresus
        server and retrieve the job ID of this job. Then, each rank will
        report itself to the PFO server with the job ID.

        Args:
            rank: Global rank of the current process.
            dp_rank: Rank in the data parallel group.
            pp_rank: Rank in the pipeline parallel group.
            tp_rank: Rank in the tensor parallel group.
            device_id: CUDA device ID that the current process manages.
            dp_degree: Size of the data parallel group.
            pp_degree: Size of the pipeline parallel group.
            tp_degree: Size of the tensor parallel group.
            world_size: Total number of ranks that participate in training.
            server_url: URL of the PFO server.
            job_metadata: An optional arbitrary string that describes the job. This will
                be appended to the job ID if given. Typically for logging purposes.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "Instantiate `PipelineFrequencyOptimizer` after `init_process_group`."
            )

        self.server_url = server_url
        self.rank = rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.device_id = device_id

        gpus = get_gpus()
        torch.cuda.set_device(device_id)

        # Rank 0 registers the job with the PFO server and retrieves the job ID.
        job_id = None
        if rank == 0:
            job_info = JobInfo(
                pp_degree=pp_degree,
                dp_degree=dp_degree,
                tp_degree=tp_degree,
                world_size=world_size,
                job_metadata=job_metadata,
            )
            response = httpx.post(
                self.server_url + REGISTER_JOB_URL, json=job_info.dict()
            )
            if (code := response.status_code) != 200:
                raise RuntimeError(
                    f"PFO server returned status code {code}: {response.text}"
                )
            job_id = response.json()
            if not isinstance(job_id, str):
                raise RuntimeError(f"PFO server returned a strange job ID: {job_id=}")

        # Rank 0 broadcasts the job ID across all ranks.
        objects = [job_id]
        dist.broadcast_object_list(objects, src=0)
        self.job_id = objects[0]
        if self.job_id is None:
            raise RuntimeError("Failed to broadcast job ID to all ranks")

        # Query the list of available frequencies of the GPU.
        max_mem_freq = max(gpus.getSupportedMemoryClocks(device_id))
        freqs = sorted(
            gpus.getSupportedGraphicsClocks(device_id, max_mem_freq),
            reverse=True,
        )

        # Each rank reports itself to the PFO server with the job ID.
        rank_info = RankInfo(
            rank=self.rank,
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            available_frequencies=freqs,
        )
        response = httpx.post(
            self.server_url + REGISTER_RANK_URL.format(job_id=self.job_id),
            json=rank_info.dict(),
        )
        if (code := response.status_code) != 200:
            raise RuntimeError(
                f"PFO server returned status code {code}: {response.text}"
            )

        # The frequency controller is responsible for controlling the frequency
        # of the GPU (device_id) asynchronously.
        self.frequency_controller = FrequencyController(device_id=device_id)

        # Fetch the frequency schedule from the PFO server.
        self.freq_schedule = self._get_frequency_schedule()
        self.freq_schedule_iter = iter(self.freq_schedule)

         # Containers for timing and energy data.
        self.timing_data = {"forward": [], "backward": []}
        self.energy_data = []

        # Spawn energy polling process.
        self.energy_polling_process = mp.Process(target=self._energy_polling_loop)
        self.energy_polling_process.daemon = True
        self.energy_polling_process.start()

    def _get_frequency_schedule(self) -> list[tuple[str, int]]:
        """Get the frequency schedule from the PFO server."""
        response = httpx.get(
            self.server_url + GET_FREQUENCY_SCHEDULE_URL.format(job_id=self.job_id),
            params={"rank": self.rank},
            timeout=None,
        )
        if (code := response.status_code) != 200:
            raise RuntimeError(
                f"PFO server returned status code {code}: {response.text}"
            )
        schedule = FrequencySchedule.parse_raw(response.text)
        if schedule.rank != self.rank:
            raise RuntimeError(
                f"PFO server returned a schedule for rank {schedule.rank} to rank {self.rank}"
            )
        return schedule.frequencies

    def on_step_begin(self) -> None:
        """Mark the beginning of a step."""
        pass

    def on_step_end(self) -> None:
        """Mark the end of a step.
        Also report the profiling result to the PFO server after N iterations.
        """
        # Frequency schedule holds one iteration-worth of frequencies, so at
        # the end of each iteration, the iterator should be exhausted.
        item = next(self.freq_schedule_iter, None)
        if item is not None:
            raise RuntimeError(
                "PFO server returned more frequencies than expected. "
                f"Next expected instruction and frequency is {item}"
            )
        self.freq_schedule_iter = iter(self.freq_schedule)

    def on_instruction_begin(self, name: str) -> None:
        """Mark the beginning of an instruction, like forward and backward.

        Retrieve the next frequency from the schedule, check whether the next
        expected instruction matches the name of the instruction, and set the
        frequency accordingly.
        """
        sync_execution([self.device_id], sync_with="torch")
        # Record the start time for latency measurement.
        self._instr_start_time = time.time()
        # Retrieve the next frequency from the schedule.
        item = next(self.freq_schedule_iter, None)
        if item is None:
            raise RuntimeError("PFO server returned fewer frequencies than expected")

        # Check whether the next expected instruction matches the name of the instruction.
        instruction, frequency = item
        if instruction != name:
            raise RuntimeError(
                f"The next expected instruction is not forward: {instruction}"
            )

        self.frequency_controller.set_frequency(frequency)

    def on_instruction_end(self, name: str) -> None:
        """Mark the end of an instruction, like forward and backward and report its latency."""
        end_time  = time.time()
        self.timing_data.setdefault(name, []).append((self._instr_start_time, end_time))
        # Report timing data to the server.
        payload = {
            "job_id": self.job_id,
            "rank": self.rank,
            "timing_breakdown": self.timing_data,
            
        }
        try:
            httpx.post(f"{self.server_url}/{REPORT_TIMING_URL}", json=payload, timeout=5)
        except Exception as e:
            pass

    def _energy_polling_loop(self):
        """Continuously measure energy and report measurements to the server."""

        # we are aggregating in the generate_profile_csv function, hence not appending or collecting here.
        # Please let me know if that should be changed
        polling_interval = 1.0
        gpus = get_gpus()
        while True:
            # Measure energy consumption over the polling interval.
            measurement = self.measure_energy(polling_interval, gpus)
            self.energy_data.append(measurement)
            payload = {
                "job_id": self.job_id,
                "rank": self.rank,
                "energy_measurements": [measurement],
            }
            try:
                httpx.post(f"{self.server_url}/{REPORT_ENERGY_URL}", json=payload, timeout=5)
            except Exception as e:
                pass

    def measure_energy(self, polling_interval: float, gpus) -> tuple[float, float]:
        """
        Measure GPU energy consumption over a polling interval.
        
        Args:
            polling_interval: Duration (in seconds) over which to measure energy.
            gpus: The GPU interface obtained from get_gpus().
        
        Returns:
            A tuple (timestamp, energy_delta) where:
              - timestamp: The time at the end of the measurement window.
              - energy_delta: The difference in energy consumption (in Joules) over the interval.
        """
        start_time = time.time()
        start_energy = gpus.getTotalEnergyConsumption(self.device_id) / 1000.0
        time.sleep(polling_interval)
        end_time = time.time()
        end_energy = gpus.getTotalEnergyConsumption(self.device_id) / 1000.0
        energy_delta = end_energy - start_energy
        return (end_time, energy_delta)

