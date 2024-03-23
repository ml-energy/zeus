"""Commands to use `BatchSizeStateRepository`."""

from pydantic import Field
from pydantic.class_validators import validator
from zeus.optimizer.batch_size.server.batch_size_state.models import BatchSizeBase
from zeus.optimizer.batch_size.server.database.schema import (
    ExplorationStateTable,
    State,
)


class UpdateExploration(BatchSizeBase):
    """Parameters to update exploration.

    Attributes:
        job_id: ID of job
        batch_size: batch size of this exploration.
        round_number: updated round number of exploration
        state: updated state of exploration
        cost: training cost of this exploration.
    """

    round_number: int = Field(ge=1)
    state: State
    cost: float

    @validator("state")
    def _check_state(cls, s: State) -> State:
        """Check if state is not Exploring. Since we are updating the state after observation, it should be eiter Converged or Unconverged."""
        if s != State.Exploring:
            return s
        else:
            raise ValueError(f"{s} shouldn't be exploring.")


class CreateExploration(BatchSizeBase):
    """Parameters to create a new exploration.

    Attributes:
        job_id: ID of job
        batch_size: batch size of this exploration.
        round_number: which round this exploration is in.
        state: state of exploration. Should be Exploring since we are starting a new exploration (immutable).
        cost: training cost of exploration. Cost should be None since we are issuing this exploration (immutable).
    """

    round_number: int = Field(ge=1)
    state: State = Field(State.Exploring, const=True)
    cost: None = Field(None, const=True)

    def to_orm(self) -> ExplorationStateTable:
        """Create an ORM object from pydantic model.

        Returns:
            `ExplorationState`: ORM object representing the exploration state.
        """
        d = self.dict()
        exp = ExplorationStateTable()
        for k, v in d.items():
            setattr(exp, k, v)
        return exp
