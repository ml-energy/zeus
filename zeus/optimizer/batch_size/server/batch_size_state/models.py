"""
Pydantic models for Batch size/Exploration/Measurement/GaussianTsArmState
"""

from __future__ import annotations
from typing import Any, Optional
from uuid import UUID

from pydantic.class_validators import root_validator
from pydantic.fields import Field
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmState,
    Measurement,
    State,
)


class BatchSizeBase(BaseModel):
    job_id: UUID
    batch_size: int = Field(gt=0)

    class Config:
        validate_assignment = True
        frozen = True


class GaussianTsArmStateModel(BatchSizeBase):
    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int = Field(ge=0)

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
    time: float = Field(ge=0)
    energy: float = Field(ge=0)
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

    round_number: int = Field(ge=1)
    state: State = State.Exploring
    cost: Optional[float] = None

    class Config:
        orm_mode = True
        frozen = True


class ExplorationsPerBs(BatchSizeBase):
    explorations: list[ExplorationStateModel]

    class Config:
        validate_assignment = True
        frozen = True

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        bs: int = values["batch_size"]
        job_id: UUID = values["job_id"]
        exps: list[ExplorationStateModel] = values["explorations"]
        exps = sorted(exps, key=lambda exp: exp.round_number, reverse=True)

        round_number = -1
        for exp in exps:
            if job_id != exp.job_id:
                raise ValueError(
                    f"job_id doesn't correspond with explorations: {job_id} != {exp.job_id}"
                )
            if bs != exp.batch_size:
                raise ValueError(
                    f"Batch size doesn't correspond with explorations: {bs} != {exp.batch_size}"
                )
            if round_number != -1 and round_number - 1 != exp.round_number:
                raise ValueError(
                    f"Round number is not in order. Should be ordered in desc without any gaps: Expecting {round_number - 1} but got {exp.round_number}"
                )
            round_number = exp.round_number

        return values


class MeasurementsPerBs(BatchSizeBase):
    measurements: list[MeasurementOfBs]

    # Validate if job_id and bs are consistent
    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        bs: int = values["batch_size"]
        job_id: UUID = values["job_id"]
        ms: list[MeasurementOfBs] = values["measurements"]

        for m in ms:
            if job_id != m.job_id:
                raise ValueError(
                    f"job_id doesn't correspond with explorations: {job_id} != {m.job_id}"
                )
            if bs != m.batch_size:
                raise ValueError(
                    f"Batch size doesn't correspond with explorations: {bs} != {m.batch_size}"
                )

        return values


class ExplorationsPerJob(BaseModel):
    job_id: UUID
    explorations_per_bs: dict[int, ExplorationsPerBs]  # BS -> Explorations

    # Check bs and job_id corresponds to explorations_per_bs
    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        job_id: UUID = values["job_id"]
        exps_per_bs: dict[int, ExplorationsPerBs] = values["explorations_per_bs"]

        for bs, exps in exps_per_bs.items():
            if job_id != exps.job_id:
                raise ValueError(
                    f"job_id doesn't correspond with explorations: {job_id} != {exps.job_id}"
                )
            if bs != exps.batch_size:
                raise ValueError(
                    f"Batch size doesn't correspond with explorations: {bs} != {exps.batch_size}"
                )

        return values
