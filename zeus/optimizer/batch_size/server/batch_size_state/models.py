"""Pydantic models for Batch size/Trials/GaussianTsArmState."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from zeus.utils.pydantic_v1 import Field, root_validator, validator, BaseModel
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmStateTable,
    TrialStatus,
    TrialType,
)


class BatchSizeBase(BaseModel):
    """Base model for representing batch size.

    Attributes:
        job_id (str): The ID of the job.
        batch_size (int): The size of the batch (greater than 0).
    """

    job_id: str
    batch_size: int = Field(gt=0)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class Trial(BatchSizeBase):
    """Pydantic model that represents Trial.

    Attributes:
        job_id (str): The ID of the job.
        batch_size (int): The size of the batch (greater than 0).
        trial_number (int): Number of trial.
        start_timestamp (datetime): Start time of trial.
        end_timestamp (datetime): End time of trial.
        type (TrialType): Type of this trial, which means in which stage this trial was executed.
        status (TrialStatus): Status of trial
        time (Optional[float]): Total time consumption of this trial.
        energy (Optional[float]): Total energy consumption of this trial.
        converged (Optional[bool]): Whether this trial is converged or not.
    """

    trial_number: int = Field(gt=0)
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = Field(None)
    type: TrialType
    status: TrialStatus
    time: Optional[float] = Field(None, ge=0)
    energy: Optional[float] = Field(None, ge=0)
    converged: Optional[bool] = None

    class Config:  # type: ignore
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    @root_validator(skip_on_failure=True)
    def _validate_mab(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate trial.

        We are checking
            - start_timestamp <= end_timestamp
            - if status == dispatched | Failed, time/energy/converged = None
                else time/energy/converged != None
        """
        start_timestamp: datetime = values["start_timestamp"]
        end_timestamp: datetime | None = values["end_timestamp"]
        status: TrialStatus = values["status"]
        time: float | None = values["time"]
        energy: float | None = values["energy"]
        converged: bool | None = values["converged"]

        if end_timestamp is not None and start_timestamp > end_timestamp:
            raise ValueError(
                f"start is earlier than end: {start_timestamp} > {end_timestamp}"
            )
        if status in (TrialStatus.Dispatched, TrialStatus.Failed):
            if not (time is None and energy is None and converged is None):
                raise ValueError("Trial status and result is not matching.")
            if status == TrialStatus.Failed and end_timestamp is None:
                raise ValueError("Trial ended but end_timestamp is None.")
        elif (
            time is None or energy is None or converged is None or end_timestamp is None
        ):
            raise ValueError(
                f"Trial ended but the result is incomplete: time({time}), energy({energy}), converged({converged}), end_timestamp({end_timestamp})"
            )

        return values


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

    class Config:  # type: ignore
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


# Helper models


class TrialResult(BatchSizeBase):
    """Model for reading the result of the trial.

    Refer to [`Trial`][zeus.optimizer.batch_size.server.batch_size_state.models.Trial] for attributes.
    """

    trial_number: int = Field(gt=0)
    status: TrialStatus
    time: float = Field(ge=0)
    energy: float = Field(ge=0)
    converged: bool

    class Config:  # type: ignore
        """Model configuration.

        Enable instantiating model from an ORM object, and make it immutable after it's created.
        """

        orm_mode = True
        frozen = True

    @validator("status")
    def _check_state(cls, s: TrialStatus) -> TrialStatus:
        """Check if status is equal to succeeded."""
        if s == TrialStatus.Succeeded:
            return s
        else:
            raise ValueError(f"{s} should be succeeded to have a valid result.")


class TrialResultsPerBs(BatchSizeBase):
    """Model representing all succeeded results of trial for a given batch size.

    Attributes:
        results (list[TrialResult]): List of TrialResult per batch size.
    """

    results: list[TrialResult]

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate if job_id and bs are consistent across all items in results."""
        bs: int = values["batch_size"]
        job_id: str = values["job_id"]
        ms: list[TrialResult] = values["results"]
        ms.sort(key=lambda x: x.trial_number, reverse=True)

        for m in ms:
            if job_id != m.job_id:
                raise ValueError(
                    f"job_id doesn't correspond with results: {job_id} != {m.job_id}"
                )
            if bs != m.batch_size:
                raise ValueError(
                    f"Batch size doesn't correspond with results: {bs} != {m.batch_size}"
                )
            if m.status != TrialStatus.Succeeded:
                raise ValueError(
                    f"This list should only contain succeeded trials. Encounted trial({m.trial_number}) of status = {m.status}"
                )

        return values


class ExplorationsPerJob(BaseModel):
    """Model representing all succeeded explorations we have done for a job. Immutable after it's created.

    Attributes:
        job_id (str): The ID of the job.
        explorations_per_bs (dict[int, list[Trial]]): Dictionary of "succeeded" explorations per batch size in trial_number ascending order.
    """

    job_id: str
    explorations_per_bs: dict[int, list[Trial]]  # BS -> Trials with exploration type

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _check_explorations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check bs and job_id corresponds to explorations_per_bs and batch size is consistent."""
        job_id: str = values["job_id"]
        exps_per_bs: dict[int, list[Trial]] = values["explorations_per_bs"]

        for bs, exps in exps_per_bs.items():
            # Sort ascending just in case. Sql will return asc order anyways.
            exps.sort(key=lambda x: x.trial_number)
            for exp in exps:
                if job_id != exp.job_id:
                    raise ValueError(
                        f"job_id doesn't correspond with explorations: {job_id} != {exp.job_id}"
                    )
                if bs != exp.batch_size:
                    raise ValueError(
                        f"Batch size doesn't correspond with explorations: {bs} != {exp.batch_size}"
                    )
                if exp.type != TrialType.Exploration:
                    raise ValueError("Trial type is not equal to Exploration.")
                if exp.status == TrialStatus.Failed:
                    raise ValueError("Should not include failed trial.")

        return values
