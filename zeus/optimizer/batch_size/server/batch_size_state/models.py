"""
Pydantic models for Batch size/Exploration/Measurement/GaussianTsArmState
"""

from enum import Enum
from uuid import UUID
from pydantic.main import BaseModel


class BatchSizeBase(BaseModel):
    job_id: UUID
    batch_size: int


class GaussianTsArmStateModel(BatchSizeBase):
    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int


class MeasurementOfBs(BatchSizeBase):
    time: float
    energy: float
    converged: bool


class ExplorationOfBs(BatchSizeBase):
    class State(Enum):
        Exploring = "Exploring"
        Converged = "Converged"
        Unconverged = "Unconverged"

    trial_number: int
    state: State = State.Exploring
    cost: float | None = None


class ExplorationsPerBs(BatchSizeBase):
    explorations: list[ExplorationOfBs]

    # TODO: Add validator that check if the list is legit
    # 1. trial number <= len(explorations)
    # 2. trial number should be in order (0,1,2, ... )
    # 3. check bs corresponds to batchSizeBase


class MeasurementsPerBs(BatchSizeBase):
    measurements: list[MeasurementOfBs]


class BatchSizeStates(BatchSizeBase):
    explorations: ExplorationsPerBs
    measurements: MeasurementsPerBs
    arm_state: GaussianTsArmStateModel | None


class ExplorationsPerJob(BaseModel):
    job_id: UUID
    explorations: dict[int, ExplorationsPerBs]  # BS -> Explorations

    # Add validator, check trial number is in order per bs + bs and job_id correspond
