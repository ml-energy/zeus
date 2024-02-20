from __future__ import annotations

import httpx
import pynvml
from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.optimizer.batch_size.common import JobSpec, ReportResponse, TrainingResult


class BatchSizeOptimizer(Callback):
    def __init__(self, monitor: ZeusMonitor, server_url: str, job: JobSpec) -> None:

        self.monitor = monitor
        self.server_url = server_url
        self.job = job
        self.cur_epoch = 0
        self.running_time = 0.0
        self.consumed_energy = 0.0
        self.train_end = False

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

        if self.train_end == True:
            return self.current_batch_size

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
        self.train_end = False
        self.monitor.begin_window("BatciSizeOptimizerClient")

    def on_evaluate(
        self,
        metric: float,
    ) -> None:
        """If converged or max_epoch is reached, report the result to BSO server"""

        if self.current_batch_size == 0:
            raise ValueError("Call get_batch_size to set the batch size first")

        if self.train_end == True:
            return

        self.cur_epoch += 1
        measurement = self.monitor.end_window("BatciSizeOptimizerClient")

        self.running_time += measurement.time
        self.consumed_energy += measurement.total_energy

        training_result = TrainingResult(
            job_id=self.job.job_id,
            batch_size=self.current_batch_size,
            time=self.running_time,
            energy=self.consumed_energy,
            max_power=self.max_power,
            metric=metric,
            current_epoch=self.cur_epoch,
        )

        # report to the server about the result of this training
        res = httpx.post(
            self.server_url + "/jobs/report", content=training_result.json()
        )
        self._handle_response(res)

        parsedResposne = ReportResponse.parse_obj(res.json())

        print(f"Result: {parsedResposne}")
        if parsedResposne.stop_train == False:
            self.monitor.begin_window("BatciSizeOptimizerClient")
        else:
            self.train_end = True
            if parsedResposne.converged == False:
                raise RuntimeError(f"Train failed: {parsedResposne.message}")

    def _handle_response(self, res: httpx.Response) -> None:
        if not (200 <= (code := res.status_code) < 300):
            raise RuntimeError(f"Zeus server returned status code {code}: {res.text}")
