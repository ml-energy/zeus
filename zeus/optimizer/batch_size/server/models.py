"""Shared model definitions for the server and client."""

from enum import Enum
from typing import List, Dict
from uuid import UUID

from pydantic import (
    BaseModel,
    validator,
)


class MabSetting(BaseModel):
    """Mab setting"""

    prior_mean: float = 0.0
    prior_precision: float = 0.0
    seed: int = 123456
    num_exploration: int = 2


class JobSpec(BaseModel):
    """Specification of a job submitted by users."""

    job_id: UUID
    seed: int = 1
    default_batch_size: int = 1024
    batch_sizes: list[int] = []
    eta_knob: float = 0.5
    beta_knob: float = 2.0
    target_metric: float = 0.50
    high_is_better_metric: bool = True
    max_epochs: int = 100
    num_pruning_rounds: int = 2
    window_size: int = 0
    mab_setting: MabSetting

    @validator("job_id")
    def _validate_job_id(cls, v: UUID) -> UUID:
        """
        Pure data validation. It is not ASYNC
        """
        # TODO: Check if the job_id already exists in DB.Jobs
        #   - If job exists, validate that the job spec is the same
        #     - If job spec is the same, return job_id
        #     - Otherwise, return an error that suggests the user to generate a new job_id for this new job
        #   - If job not exists, return job_id
        return v

    @validator("batch_sizes")
    def _validate_batch_sizes(cls, bs: list[int]) -> int:
        if bs is None or len(bs) != 0:
            bs.sort()
            return bs
        else:
            raise ValueError(f"Batch Sizes = {bs} is empty")

    @validator("default_batch_size", "max_epochs")
    def _check_positivity(cls, n: int) -> int:
        if n > 0:
            return n
        else:
            raise ValueError(f"{n} should be larger than 0")


class TrainingResult(BaseModel):
    """Result of training for that job & batch size"""

    job_id: UUID
    batch_size: int
    time: float
    energy: float
    max_power: int
    converged: bool  # Client computes converged anyways to check if it reached max epochs. So just send the result to the server
