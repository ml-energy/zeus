"""Shared model definitions for the server and client."""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, root_validator, validator
from pydantic.fields import Field


REGISTER_JOB_URL = "/jobs"
GET_NEXT_BATCH_SIZE_URL = "/jobs/batch_size"
REPORT_RESULT_URL = "/jobs/report"


class JobSpec(BaseModel):
    """Job specification that user inputs.

    Attributes:
        job_id: unique ID for the job
        batch_sizes: list of batch sizes to try
        default_batch_size: first batch size to try
        eta_knob: eta for computing `zeus_cost`
        beta_knob: beta for early stopping. If min_cost*beta_knob < current_cost, job will be stopped by bso server.
                    To disable, set it to -1.
        target_metric: target metric to achieve for training.
        high_is_better_metric: if the goal of training is achieving higher metric that `target_metric`
        max_epochs: max_epochs to try if the train doesn't converge.
        num_pruning_rounds: Number of rounds we are trying for pruning stage
        window_size: For MAB, how many recent measurements to fetch for computing the arm states. If set to 0, fetch all measurements.

        mab_prior_mean: Mean of the belief prior distribution.
        mab_prior_precision: Precision of the belief prior distribution.
        mab_num_exploration: How many static explorations to run when no observations are available.
        mab_seed: The random seed to use.
    """

    job_id: UUID
    batch_sizes: list[int] = []
    default_batch_size: int = Field(gt=0)
    eta_knob: float = 0.5
    beta_knob: float = 2.0
    target_metric: float = 0.50
    high_is_better_metric: bool = True
    max_epochs: int = Field(100, gt=0)
    num_pruning_rounds: int = 2
    window_size: int = 10

    mab_prior_mean: float = 0.0
    mab_prior_precision: float = 0.0
    mab_num_exploration: int = 2
    mab_seed: Optional[int] = None
    gpu_model: str  # TODO: Can I get the gpu_model from monitor?

    @validator("batch_sizes")
    def _validate_batch_sizes(cls, bs: list[int]) -> int:
        if bs is not None and len(bs) > 0:
            bs.sort()
            return bs
        else:
            raise ValueError(f"Batch Sizes = {bs} is empty")

    @validator("eta_knob")
    def _validate_eta_knob(cls, v: float) -> int:
        if v < 0 or v > 1:
            raise ValueError("eta_knob should be in range [0,1]")
        return v

    @validator("beta_knob")
    def _validate_beta_knob(cls, v: float) -> int:
        if v == -1 or v > 0:
            return v
        else:
            raise ValueError(
                f"Invalid beta_knob({v}). To disable early stop, set beta_knob = -1 or positive value."
            )

    @root_validator(skip_on_failure=True)
    def _check_default_batch_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        bs = values["default_batch_size"]
        bss = values["batch_sizes"]
        if bs not in bss:
            raise ValueError(f"Default BS({bs}) not in batch_sizes({bss}).")
        return values


class JobConfig(JobSpec):
    """Internal job configuration including gpu settigns.

    Attributes:
        max_power: sum of maximum power limit of all gpus we are using
        number_of_gpus: number of gpus that are being used for training
    """

    max_power: float
    number_of_gpus: int


class TrainingResult(BaseModel):
    """Result of training for that job & batch size.

    Attributes:
        job_id: unique Id of job
        batch_size: batch size of this training
        time: total time consumption so far
        energy: total energy consumption so far
        metric: current metric value after `current_epoch`
        current_epoch: current epoch of training. Server can check if the train reached the `max_epochs`
    """

    job_id: UUID
    batch_size: int
    time: float
    energy: float
    metric: float
    current_epoch: int


class ReportResponse(BaseModel):
    """Response format from the server for client's training result report.

    Attributes:
        stop_train: Whether we should stop training or not.
        converged: converged or not.
        message: message from the server regarding training. ex) why train should be stopped.
    """

    stop_train: bool
    converged: bool
    message: str
