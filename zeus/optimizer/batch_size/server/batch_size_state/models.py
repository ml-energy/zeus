"""Pydantic models for Batch size/Exploration/Measurement/GaussianTsArmState."""

from __future__ import annotations
from typing import Any, Optional
from uuid import UUID

from pydantic.class_validators import root_validator
from pydantic.fields import Field
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmStateTable,
    MeasurementTable,
    State,
)


class BatchSizeBase(BaseModel):
    """Base model for representing batch size.

    Attributes:
        job_id (UUID): The ID of the job.
        batch_size (int): The size of the batch (greater than 0).
    """

    job_id: UUID
    batch_size: int = Field(gt=0)

    class Config:
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class GaussianTsArmState(BatchSizeBase):
    """Model representing Gaussian Thompson Sampling arm state.

    Attributes:
        param_mean (float): Mean of the belief prior distribution.
        param_precision (float): Precision of the belief prior distribution.
        reward_precision (float): Precision (inverse variance) of the reward distribution.
        num_observations (int): How many observations we made.
    """

    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int = Field(ge=0)

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    def to_orm(self) -> GaussianTsArmStateTable:
        """Convert pydantic model to ORM object.

        Returns:
            GaussianTsArmState: The ORM object of Gaussian Arm State.
        """
        d = self.dict()
        g = GaussianTsArmStateTable()
        for k, v in d.items():
            setattr(g, k, v)
        return g


class Measurement(BatchSizeBase):
    """Model representing a measurement(result) of training the batch size of the job. Immutable after it's created.

    Attributes:
        time (float): The time measurement in second (greater than or equal to 0).
        energy (float): The energy measurement in J (greater than or equal to 0).
        converged (bool): Indicates if the training was converged.
    """

    time: float = Field(ge=0)
    energy: float = Field(ge=0)
    converged: bool

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    def to_orm(self) -> MeasurementTable:
        """Convert it to ORM object.

        Returns:
            Measurement: The ORM object of Measruement.
        """
        d = self.dict()
        m = MeasurementTable()
        for k, v in d.items():
            setattr(m, k, v)
        return m


class ExplorationState(BatchSizeBase):
    """Model representing the state of one exploration. Immutable after it's created.

    Attributes:
        round_number (int): Which round we are in for exploring this batch size of the job.
        state (State): The state of exploration (Exploring|Converged|Unconverged).
        cost (Optional[float]): The result cost of training with this batch size.
    """

    round_number: int = Field(ge=1)
    state: State = State.Exploring
    cost: Optional[float] = None

    class Config:
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True


class ExplorationsPerBs(BatchSizeBase):
    """Model representing all explorations performed for that batch size and job. Immutable after it's created.

    Attributes:
        explorations (list[ExplorationStateModel]): List of exploration states for this batch size and job id.
    """

    explorations: list[ExplorationState]

    class Config:
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the consistency of explorations.

        We check if
            - job Id and batch size are consistent across all elements in the explorations.
            - round number is in descending order without any gaps between.
        """
        bs: int = values["batch_size"]
        job_id: UUID = values["job_id"]
        exps: list[ExplorationState] = values["explorations"]
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
    """Model representing all measurements we observed per batch size and job.

    Attributes:
        measurements (list[MeasurementOfBs]): List of measurements per batch size.
    """

    measurements: list[Measurement]

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate if job_id and bs are consistent across all items in measurements."""
        bs: int = values["batch_size"]
        job_id: UUID = values["job_id"]
        ms: list[Measurement] = values["measurements"]

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
    """Model representing all explorations we have done for a job. Immutable after it's created.

    Attributes:
        job_id (UUID): The ID of the job.
        explorations_per_bs (dict[int, ExplorationsPerBs]): Dictionary of explorations per batch size.
    """

    job_id: UUID
    explorations_per_bs: dict[int, ExplorationsPerBs]  # BS -> Explorations

    class Config:
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check bs and job_id corresponds to explorations_per_bs and batch size is consistent."""
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
