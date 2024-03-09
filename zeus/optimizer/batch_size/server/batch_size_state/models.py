"""
Pydantic models for Batch size/Exploration/Measurement/GaussianTsArmState
"""

from enum import Enum
from uuid import UUID
from pydantic.fields import Field
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmState,
    Measurement,
    State,
)


class BatchSizeBase(BaseModel):
    job_id: UUID
    batch_size: int

    class Config:
        validate_assignment = True
        frozen = True


class GaussianTsArmStateModel(BatchSizeBase):
    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int

    class Config:
        orm_mode = True
        frozen = True

    def to_orm(self) -> GaussianTsArmState:
        d = self.dict()
        g = GaussianTsArmState()
        for k, v in d.items():
            setattr(g, k, v)
        return g


class MeasurementOfBs(BatchSizeBase):
    time: float
    energy: float
    converged: bool

    class Config:
        orm_mode = True
        frozen = True

    def to_orm(self) -> Measurement:
        d = self.dict()
        m = Measurement()
        for k, v in d.items():
            setattr(m, k, v)
        return m


class ExplorationStateModel(BatchSizeBase):

    round_number: int
    state: State = State.Exploring
    cost: float | None = None

    class Config:
        orm_mode = True
        frozen = True


class ExplorationsPerBs(BatchSizeBase):
    explorations: list[ExplorationStateModel]

    # TODO: Add validator that check if the list is legit
    # 1. trial number <= len(explorations)
    # 2. trial number should be in order (0,1,2, ... )
    # 3. check bs corresponds to batchSizeBase

    class Config:
        validate_assignment = True
        frozen = True


class MeasurementsPerBs(BatchSizeBase):
    measurements: list[MeasurementOfBs]

    # Validate if job_id and bs are consistent

    # Trigger validator when the list changes
    # class Config:
    #     validate_assignment = True
    #     frozen = True


class BatchSizeStates(BatchSizeBase):
    explorations: ExplorationsPerBs
    measurements: MeasurementsPerBs
    arm_state: GaussianTsArmStateModel | None


class ExplorationsPerJob(BaseModel):
    job_id: UUID
    explorations_per_bs: dict[int, ExplorationsPerBs]  # BS -> Explorations

    # Add validator, check trial number is in order per bs + bs and job_id correspond
