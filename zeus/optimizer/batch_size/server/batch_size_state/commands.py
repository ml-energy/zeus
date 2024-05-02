"""Commands to use `BatchSizeStateRepository`."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from zeus.utils.pydantic_v1 import root_validator, validator, Field
from zeus.optimizer.batch_size.server.batch_size_state.models import BatchSizeBase
from zeus.optimizer.batch_size.server.database.schema import (
    TrialStatus,
    TrialTable,
    TrialType,
)


class ReadTrial(BatchSizeBase):
    """Command to read a trial.

    Equivalent to primary key of Trial.

    Attributes:
        job_id: ID of job
        batch_size: batch size of a given trial
        trial_number: number of trial
    """

    trial_number: int = Field(gt=0)


class CreateTrialBase(BatchSizeBase):
    """Base command to create trial."""

    type: TrialType
    start_timestamp: datetime = Field(default_factory=datetime.now)
    status: TrialStatus = Field(default=TrialStatus.Dispatched, const=True)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class CreateTrial(CreateTrialBase):
    """Internal command to create trial.

    trial_number is populate within ZeusService.
    """

    trial_number: int = Field(gt=0)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    def to_orm(self) -> TrialTable:
        """Create an ORM object from pydantic model.

        Returns:
            `TrialTable`: ORM object representing the trial.
        """
        d = self.dict()
        t = TrialTable()
        for k, v in d.items():
            setattr(t, k, v)
        return t


class CreateExplorationTrial(CreateTrialBase):
    """Create a exploration."""

    type: TrialType = Field(default=TrialType.Exploration, const=True)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class CreateMabTrial(CreateTrialBase):
    """Create a MAB trial."""

    type: TrialType = Field(default=TrialType.MAB, const=True)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class CreateConcurrentTrial(CreateTrialBase):
    """Create a exploration."""

    type: TrialType = Field(default=TrialType.Concurrent, const=True)

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True


class UpdateTrial(BatchSizeBase):
    """Report the result of trial."""

    trial_number: int = Field(gt=0)
    end_timestamp: datetime = Field(default_factory=datetime.now)
    status: TrialStatus
    time: Optional[float] = Field(default=None, ge=0)
    energy: Optional[float] = Field(default=None, ge=0)
    converged: Optional[bool] = None

    class Config:  # type: ignore
        """Model configuration.

        Make it immutable after it's created.
        """

        frozen = True

    @validator("status")
    def _check_status(cls, s: TrialStatus) -> TrialStatus:
        """Check if status is equal to Dispatched."""
        if s != TrialStatus.Dispatched:
            return s
        else:
            raise ValueError(
                f"{s} shouldn't be Dispatched since this is reporting the result."
            )

    @root_validator(skip_on_failure=True)
    def _validate_sanity(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate result.

        We are checking
            - if status == Failed, time/energy/converged == None.
                else, time/energy/converged != None.
        """
        status: TrialStatus = values["status"]

        time: float | None = values["time"]
        energy: float | None = values["energy"]
        converged: bool | None = values["converged"]

        if status != TrialStatus.Failed and (
            time is None or energy is None or converged is None
        ):
            raise ValueError(
                f"Result is incomplete: time({time}), energy({energy}), converged({converged})"
            )

        return values
