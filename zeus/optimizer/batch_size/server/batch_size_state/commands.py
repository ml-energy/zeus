from typing import Any
from uuid import UUID
from pydantic import Field
from pydantic.class_validators import validator
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
)
from zeus.optimizer.batch_size.server.database.schema import ExplorationState, State


class UpdateExploration(BatchSizeBase):
    round_number: int
    state: State
    cost: float

    # state shouldn't be exploring

    @validator("state")
    def _check_positivity(cls, s: State) -> State:
        if s != State.Exploring:
            return s
        else:
            raise ValueError(f"{s} shouldn't be exploring.")


# sub_item = SubItem(**base_item.dict(), extra_field=10)
class CreateExploration(BatchSizeBase):
    round_number: int
    state: State = Field(State.Exploring, const=True)
    cost: None = Field(None, const=True)

    # Validate trial_number is in order

    def to_orm(self) -> ExplorationState:
        d = self.dict()
        exp = ExplorationState()
        for k, v in d.items():
            setattr(exp, k, v)
        return exp


class UpsertGaussianTsArmState(BatchSizeBase):
    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int
