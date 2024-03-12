from pydantic import Field
from pydantic.class_validators import validator
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.batch_size_state.models import BatchSizeBase
from zeus.optimizer.batch_size.server.database.schema import ExplorationState, State


class UpdateExploration(BatchSizeBase):
    round_number: int
    state: State
    cost: float

    @validator("state")
    def _check_positivity(cls, s: State) -> State:
        if s != State.Exploring:
            return s
        else:
            raise ValueError(f"{s} shouldn't be exploring.")


class CreateExploration(BatchSizeBase):
    round_number: int = Field(ge=1)
    state: State = Field(State.Exploring, const=True)
    cost: None = Field(None, const=True)

    def to_orm(self) -> ExplorationState:
        d = self.dict()
        exp = ExplorationState()
        for k, v in d.items():
            setattr(exp, k, v)
        return exp
