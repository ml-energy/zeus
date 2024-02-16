from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

import pynvml
import httpx

from zeus.callback import Callback
from zeus.job import Job
from zeus.monitor import ZeusMonitor
from zeus.monitor.energy import Measurement
from zeus.optimizer.batch_size.server.models import JobSpec, TrainingResult
from zeus.util.logging import get_logger
from zeus.util.metric import zeus_cost

"""
TODO: Do we want one job -> one BSO Client? 
"""


class BatchSizeOptimizerClient(Callback):
    def __init__(self, monitor: ZeusMonitor, server_url: str, job: JobSpec) -> None:

        self.monitor = monitor
        self.server_url = server_url
        self.job = job
        self.cur_epoch = 0
        self.running_time = 0.0
        self.consumed_energy = 0.0

        # Get max PL
        pynvml.nvmlInit()
        pls = []
        self.max_power = 0
        for index in self.monitor.nvml_gpu_indices:
            device = pynvml.nvmlDeviceGetHandleByIndex(index)
            pls.append(pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device))
        if not all(pls[0] == pl for pl in pls):
            raise ValueError("Power limits ranges are not uniform across GPUs.")

        self.max_power = max(pls) * len(monitor.gpu_indices)
        self.current_batch_size = 0

        # Register job
        res = httpx.post(self.server_url + "/jobs", content=job.json())
        self._handle_response(res)

    def get_batch_size(self) -> int:
        """Get batch size to use from the BSO server"""
        self.cur_epoch = 0
        res = httpx.get(
            self.server_url + "/jobs/batch_size", params={"job_id": self.job.job_id}
        )
        self._handle_response(res)

        batch_size = res.json()
        if not isinstance(batch_size, int) or batch_size not in self.job.batch_sizes:
            raise RuntimeError(
                f"Zeus server returned a strange batch_size: {batch_size}"
            )

        self.current_batch_size = batch_size
        return batch_size

    def on_train_begin(self) -> None:
        self.monitor.begin_window("BatciSizeOptimizerClient")

    def on_evaluate(
        self,
        metric: float,
    ) -> None:
        """If converged or max_epoch is reached, report the result to BSO server"""

        if self.current_batch_size == 0:
            raise ValueError("Call get_batch_size to set the batch size first")

        self.cur_epoch += 1
        converged = False

        if (self.job.high_is_better_metric and self.job.target_metric <= metric) or (
            not self.job.high_is_better_metric and self.job.target_metric >= metric
        ):
            converged = True

        measurement = self.monitor.end_window("BatciSizeOptimizerClient")

        self.running_time += measurement.time
        self.consumed_energy += measurement.total_energy

        training_result = TrainingResult(
            job_id=self.job.job_id,
            batch_size=self.current_batch_size,
            time=self.running_time,
            energy=self.consumed_energy,
            max_power=self.max_power,
            converged=converged,
            current_epoch=self.cur_epoch,
        )

        # report to the server about the result of this training
        res = httpx.post(
            self.server_url + "/jobs/report", content=training_result.json()
        )
        self._handle_response(res)

        if not converged and self.cur_epoch < self.job.max_epochs:
            self.monitor.begin_window("BatciSizeOptimizerClient")

        if converged:
            # TODO: If training is done and converged, anything BSO should do? (stop training? -> User should do this probably)
            pass

    def _handle_response(self, res: httpx.Response) -> None:
        if not (200 <= (code := res.status_code) < 300):
            raise RuntimeError(f"Zeus server returned status code {code}: {res.text}")
