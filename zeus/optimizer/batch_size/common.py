"""Shared model definitions for the server and client."""

from typing import Any, Dict
from uuid import UUID

from pydantic import BaseModel, root_validator, validator

REGISTER_JOB_URL = "/jobs"
GET_NEXT_BATCH_SIZE_URL = "/jobs/batch_size"
REPORT_RESULT_URL = "/jobs/report"


class MabSetting(BaseModel):
    """Mab setting"""

    prior_mean: float = 0.0
    prior_precision: float = 0.0
    seed: int = 123456
    num_exploration: int = 1


class JobSpec(BaseModel):
    """Specification of a job submitted by users."""

    job_id: UUID
    seed: int = 1
    batch_sizes: list[int] = []
    default_batch_size: int = 1024
    eta_knob: float = 0.5
    beta_knob: float = 2.0
    target_metric: float = 0.50
    high_is_better_metric: bool = True
    max_epochs: int = 100
    num_pruning_rounds: int = 2
    window_size: int = 0
    mab_setting: MabSetting = MabSetting()

    @validator("batch_sizes")
    def _validate_batch_sizes(cls, bs: list[int]) -> int:
        if bs is None or len(bs) != 0:
            bs.sort()
            return bs
        else:
            raise ValueError(f"Batch Sizes = {bs} is empty")

    @validator("eta_knob")
    def _validate_eta_knob(cls, v: float) -> int:
        if v < 0 or v > 1:
            raise ValueError(f"eta_knob should be in range [0,1]")
        return v

    @validator("beta_knob")
    def _validate_beta_knob(cls, v: float) -> int:
        if v == -1 or v > 0:
            return v
        else:
            raise ValueError(
                f"Invalid beta_knob({v}). To disable early stop, set beta_knob = -1 or positive value."
            )

    @validator("default_batch_size", "max_epochs")
    def _check_positivity(cls, n: int) -> int:
        if n > 0:
            return n
        else:
            raise ValueError(f"{n} should be larger than 0")

    @root_validator
    def _check_default_batch_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        bs = values.get("default_batch_size")
        bss = values.get("batch_sizes")
        if bs not in bss:
            raise ValueError(f"Default BS({bs}) not in batch_sizes({bss}).")
        return values


class TrainingResult(BaseModel):
    """Result of training for that job & batch size
    current_epoch: For early stopping. Easier to just get current epoch from the client than server tracking it if there is a concurrency
    """

    job_id: UUID
    batch_size: int
    time: float
    energy: float
    max_power: int
    metric: float
    current_epoch: int


class ReportResponse(BaseModel):
    """Response format"""

    stop_train: bool
    converged: bool
    message: str


class ZeusBSOJobSpecMismatch(Exception):
    def __init__(self, message: str):
        self.message = message


class ZeusBSOValueError(Exception):
    def __init__(self, message: str):
        self.message = message
