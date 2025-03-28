"""Zeus batch size optimizer client that communicates with server."""

from __future__ import annotations
import atexit

import httpx
from zeus.callback import Callback
from zeus.monitor import ZeusMonitor
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_END_URL,
    REPORT_RESULT_URL,
    CreatedJob,
    JobSpecFromClient,
    JobSpec,
    TrialId,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.exceptions import (
    ZeusBSOBadOperationError,
    ZeusBSOConfigError,
    ZeusBSOOperationOrderError,
    ZeusBSORuntimError,
    ZeusBSOTrainFailError,
)
from zeus.utils.logging import get_logger
from zeus.device import get_gpus

logger = get_logger(__name__)


class BatchSizeOptimizer(Callback):
    """Batch size optimizer client that talks to server.

    The following methods must be called in order inside the training script:

    - `get_batch_size`: At the beginning of the script.
    - `on_train_begin`: Before running any epochs.
    - `on_evaluate`: After each epoch when the validation metric is available.

    One batch size optimizer per one training session of the job.
    The set of GPUs to be used for training should be homogeneous, and will be inferred
    from the `ZeusMonitor` instance passed into the constructor.
    """

    def __init__(
        self, monitor: ZeusMonitor, server_url: str, job: JobSpec, rank: int = 0
    ) -> None:
        """Initialize the optimizer, and register the job to the server.

        If job is already registered, check if the job configuration is identical with previously registered config.

        Args:
            monitor: `ZeusMonitor` instance configured to measure the energy of all and only the GPUs used for training.
            server_url: url of batch size optimizer server
            job: job specification. Refer to `JobSpec` for job specifcatio parameters.
            rank: rank of gpu in the case of distributed training. We only let rank = 0 gpu to request for a batch size.
        """
        self.monitor = monitor
        self.server_url = server_url
        self.cur_epoch = 0  # 0-indexed
        self.running_time = 0.0
        self.consumed_energy = 0.0
        self.training_finished = False
        self.trial_number = 0
        self.rank = rank

        # Currently, the BSO only supports homogeneous GPU training.
        gpus = get_gpus(ensure_homogeneous=True)
        if len(gpus) == 0:
            raise ZeusBSOConfigError("No GPUs detected.")

        # set gpu configurations(max_power, number of gpus, and gpu model)
        self.job_spec = JobSpecFromClient(
            **job.dict(),
            max_power=gpus.getPowerManagementLimitConstraints(0)[1]
            // 1000
            * len(monitor.gpu_indices),
            number_of_gpus=len(monitor.gpu_indices),
            gpu_model=gpus.getName(0),
        )

        # Track the batch size of current job
        self.current_batch_size = 0

        # Register job
        res = httpx.post(
            self.server_url + REGISTER_JOB_URL, content=self.job_spec.json()
        )
        self._handle_response(res)

        self.job = CreatedJob.parse_obj(res.json())

        logger.critical(
            "Job is registered with job_id: \x1b[31;1m%s\x1b[0m", self.job.job_id
        )
        logger.info("Job is registered: %s", str(self.job))

    def get_batch_size(self) -> int:
        """Get batch size to use from the BSO server.

        Returns:
            return a batch size to use for current job

        Raises:
            `ZeusBSORuntimError`: if the batch size we receive is invalid
        """
        if self.rank != 0:
            raise ZeusBSOBadOperationError("Only rank 0 gpu can ask for a batch size.")

        if self.current_batch_size != 0:
            # If we already got the batch size, return
            return self.current_batch_size

        self.cur_epoch = 0
        res = httpx.get(
            self.server_url + GET_NEXT_BATCH_SIZE_URL,
            params={"job_id": self.job.job_id},
        )
        self._handle_response(res)
        trial_id = TrialId.parse_obj(res.json())

        if trial_id.batch_size not in self.job.batch_sizes:
            raise ZeusBSORuntimError(
                f"Zeus server returned a strange batch_size: {trial_id.batch_size}"
            )

        self.current_batch_size = trial_id.batch_size
        self.trial_number = trial_id.trial_number

        logger.info("[BatchSizeOptimizer] Chosen batch size: %s", trial_id.batch_size)

        def report_end() -> None:
            httpx.patch(self.server_url + REPORT_END_URL, content=trial_id.json())

        atexit.register(report_end)
        return trial_id.batch_size

    def on_train_begin(self) -> None:
        """Start the monitor window and mark training is started."""
        self.training_finished = False
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
            metric: Validation metric of this epoch. See also `higher_metric_is_better` in [`JobParams`][zeus.optimizer.batch_size.common.JobParams].

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

        if self.training_finished:
            return

        self.cur_epoch += 1
        measurement = self.monitor.end_window("BatciSizeOptimizerClient")

        # Accumulate time and energy
        self.running_time += measurement.time
        self.consumed_energy += measurement.total_energy

        training_result = TrainingResult(
            job_id=self.job.job_id,
            batch_size=self.current_batch_size,
            trial_number=self.trial_number,
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

        parsed_response = ReportResponse.parse_obj(res.json())

        if not parsed_response.stop_train:
            # Should keep training. Re-open the window
            self.monitor.begin_window("BatciSizeOptimizerClient")
        else:
            # Train is over. If not converged, raise an error
            self.training_finished = True
            if not parsed_response.converged:
                raise ZeusBSOTrainFailError(
                    f"Train failed: {parsed_response.message} This batch size will not be selected again. Please re-launch the training"
                )

    def _handle_response(self, res: httpx.Response) -> None:
        """Check if the response is success. Otherwise raise an error with error message from the server.

        Args:
            res: response from the server
        """
        if not (200 <= (code := res.status_code) < 300):
            raise ZeusBSORuntimError(
                f"Zeus server returned status code {code}: {res.text}"
            )
