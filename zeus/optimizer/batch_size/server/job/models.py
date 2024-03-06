"""
Pydantic models for Job
"""

from enum import Enum
import json
from typing import Any
from uuid import UUID
import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.utils import GetterDict
from zeus.optimizer.batch_size.common import JobSpec


class Stage(Enum):
    Pruning = "Pruning"
    MAB = "MAB"


class JobGetter(GetterDict):
    def get(self, key: str, default: Any) -> Any:
        if key in {"batch_sizes"}:
            return [bs.batch_size for bs in self._obj.batch_sizes]

        return super().get(key, default)


class JobState(JobSpec):
    exp_default_batch_size: int

    min_cost: float | None = None
    min_batch_size: int
    stage: Stage = Stage.Pruning

    mab_random_generator_state: str | None = None

    # batch_size_states

    # TODO: Validate generator state is not empty when seed is not empty
    # batch_size_states = []

    class Config:
        orm_mode = True
        getter_dict = JobGetter
