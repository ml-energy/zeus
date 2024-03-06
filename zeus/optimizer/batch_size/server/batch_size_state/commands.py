from typing import Any
from uuid import UUID
from pydantic import Field
from pydantic.class_validators import validator
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationOfBs,
    ExplorationsPerBs,
    GaussianTsArmStateModel,
)


class UpdateExploration(BatchSizeBase):
    state: ExplorationOfBs.State
    cost: float


# sub_item = SubItem(**base_item.dict(), extra_field=10)
class CreateExploration(BatchSizeBase):
    trial_number: int
    state: ExplorationOfBs.State = Field(ExplorationOfBs.State.Exploring, const=True)
    cost: None = Field(None, const=True)

    # Validate trial_number is in order


class UpsertGaussianTsArmState(BatchSizeBase):
    param_mean: float
    param_precision: float
    reward_precision: float
    num_observations: int
