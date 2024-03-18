from __future__ import annotations

import httpx
import pynvml

from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    JobConfig,
    JobSpec,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.exceptions import (
    ZeusBSOConfigError,
    ZeusBSOOperationOrderError,
    ZeusBSORuntimError,
    ZeusBSOTrainFailError,
)
from zeus.util.logging import get_logger

logger = get_logger(__name__)


class BatchSizeOptimizer(Callback):
    """Batch size optimizer client that talks to server. One batch size optimizer per one training session of the job."""

    def __init__(self, monitor: ZeusMonitor, server_url: str, job: JobSpec) -> None:
        """Initialize the optimizer, and register the job to the server.
        If job is already registered, check if the job configuration is identical with previously registered config.

        Args:
            monitor: zeus monitor
            server_url: url of batch size optimizer server
            job: job specification. Refer to `JobSpec` for job specifcatio parameters.
        """

        self.monitor = monitor
        self.server_url = server_url
        self.cur_epoch = 0  # 0-indexed
        self.running_time = 0.0
        self.consumed_energy = 0.0
        self.train_end = False

        # Get max PL
        pynvml.nvmlInit()
        pls = []
        for index in self.monitor.nvml_gpu_indices:
            device = pynvml.nvmlDeviceGetHandleByIndex(index)
            # name = pynvml.nvmlDeviceGetName(device)
            # TODO: CHECK DEVICE ALL EQUAL AND SET JOB.GPU_MODEL
            pls.append(pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device))
        if not all(pls[0] == pl for pl in pls):
            raise ZeusBSOConfigError("Power limits ranges are not uniform across GPUs.")

        # set gpu configurations(max_power, number of gpus, and gpu model)
        self.job = JobConfig(
            **job.dict(),
            max_power=(pls[0][1] // 1000) * len(monitor.gpu_indices),
            number_of_gpus=len(monitor.gpu_indices),
        )

        # Track the batch size of current job
        self.current_batch_size = 0

        # Register job
        res = httpx.post(self.server_url + REGISTER_JOB_URL, content=self.job.json())
        self._handle_response(res)

        logger.info(f"Job is registered: {self.job}")

    def get_batch_size(self) -> int:
        """Get batch size to use from the BSO server

        Return:
            return a batch size to use for current job

        Raises:
            `ZeusBSORuntimError`: if the batch size we receive is invalid
        """

        # If train is already over, should not re-send the request to the server. Typically, re-launch the script for another training
        if self.train_end == True:
            return self.current_batch_size

        self.cur_epoch = 0
        res = httpx.get(
            self.server_url + GET_NEXT_BATCH_SIZE_URL,
            params={"job_id": self.job.job_id},
        )
        self._handle_response(res)

        batch_size = res.json()
        if not isinstance(batch_size, int) or batch_size not in self.job.batch_sizes:
            raise ZeusBSORuntimError(
                f"Zeus server returned a strange batch_size: {batch_size}"
            )

        self.current_batch_size = batch_size
        logger.info(f"[BatchSizeOptimizer] Chosen batch size: {batch_size}")
        return batch_size

    def on_train_begin(self) -> None:
        """Start the monitor window and mark training is started"""
        self.train_end = False
        self.monitor.begin_window("BatciSizeOptimizerClient")

    def on_evaluate(
        self,
        metric: float,
    ) -> None:
        """Determine whether or not to stop training after evaluation.

        Training stops when
        - `max_epochs` was reached, or
        - the target metric was reached. or
        - Cost exceeded the early stop threshold

        Args:
            metric: Validation metric of this epoch. See also `higher_metric_is_better` in
            [`JobSpec`][zeus.optimizer.batch_size.common.JobSpec].

        Raises:
            `ZeusBSOOperationOrderError`: When `get_batch_size` was not called first.
            `ZeusBSOTrainFailError`: When train failed for a chosen batch size and should be stopped.
                                    This batch size will not be tried again. To proceed training, re-launch the training then bso will select another batch size
            `ZeusBSORuntimError`: When the server returns an error
        """

        if self.current_batch_size == 0:
            raise ZeusBSOOperationOrderError(
                "Call get_batch_size to set the batch size first"
            )

        if self.train_end == True:
            return

        self.cur_epoch += 1
        measurement = self.monitor.end_window("BatciSizeOptimizerClient")

        # Accumulate time and energy
        self.running_time += measurement.time
        self.consumed_energy += measurement.total_energy

        training_result = TrainingResult(
            job_id=self.job.job_id,
            batch_size=self.current_batch_size,
            time=self.running_time,
            energy=self.consumed_energy,
            metric=metric,
            current_epoch=self.cur_epoch,
        )

        # report to the server about the result of this training
        res = httpx.post(
            self.server_url + REPORT_RESULT_URL, content=training_result.json()
        )
        self._handle_response(res)

        parsedResposne = ReportResponse.parse_obj(res.json())

        if not parsedResposne.stop_train:
            # Should keep training. Re-open the window
            self.monitor.begin_window("BatciSizeOptimizerClient")
        else:
            # Train is over. If not converged, raise an error
            self.train_end = True
            if not parsedResposne.converged:
                raise ZeusBSOTrainFailError(
                    f"Train failed: {parsedResposne.message}. This batch size will not be selected again. Please re-launch the training"
                )

    def _handle_response(self, res: httpx.Response) -> None:
        """Check if the response is success. Otherwise raise an error with error message from the server

        Args:
            res: response from the server
        """
        if not (200 <= (code := res.status_code) < 300):
            raise ZeusBSORuntimError(
                f"Zeus server returned status code {code}: {res.text}"
            )
