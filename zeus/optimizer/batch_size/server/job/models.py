"""
Pydantic models for Job
"""

from __future__ import annotations
import json
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic.class_validators import root_validator
from pydantic.utils import GetterDict
from zeus.optimizer.batch_size.common import JobConfig


class Stage(Enum):
    Pruning = "Pruning"
    MAB = "MAB"


class JobGetter(GetterDict):
    def get(self, key: str, default: Any) -> Any:
        if key in {"batch_sizes"}:
            return [bs.batch_size for bs in self._obj.batch_sizes]

        return super().get(key, default)


class JobState(JobConfig):
    exp_default_batch_size: int

    min_cost: Optional[float] = None
    min_batch_size: int
    stage: Stage = Stage.Pruning

    mab_random_generator_state: Optional[str] = None

    class Config:
        orm_mode = True
        getter_dict = JobGetter

    @root_validator(skip_on_failure=True)
    def _validate_mab(cls, values: dict[str, Any]) -> dict[str, Any]:
        state: str | None = values["mab_random_generator_state"]
        mab_seed: int | None = values["mab_seed"]

        if mab_seed != None:
            if state == None:
                raise ValueError("mab_seed is not none, but generator state is none")
            else:
                try:
                    np.random.default_rng(1).__setstate__(json.loads(state))
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid generator state ({state})")

        return values
