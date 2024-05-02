"""Pydantic models for Job."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

import numpy as np
from zeus.utils.pydantic_v1 import root_validator
from pydantic.utils import GetterDict
from zeus.optimizer.batch_size.common import GpuConfig, JobParams


class Stage(Enum):
    """Job Stage."""

    Pruning = "Pruning"
    MAB = "MAB"


class JobGetter(GetterDict):
    """Getter for batch size to convert ORM batch size object to integer."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from dict."""
        if key == "batch_sizes":
            # If the key is batch_sizes, parse the integer from object.
            return [bs.batch_size for bs in self._obj.batch_sizes]

        return super().get(key, default)


class JobState(JobParams, GpuConfig):
    """Pydantic model for Job which includes job-level states.

    Attributes:
        exp_default_batch_size: Exploration default batch size that is used during Pruning stage.
        min_cost: Min training cost observed. Initially, None.
        min_cost_batch_size: Batch size that has minimum training cost observed.
        stage: Stage of the job.
        mab_random_generator_state: Generator state if mab_seed is not None. Otherwise, None.

    For the rest of attributes, refer to [`JobParams`][zeus.optimizer.batch_size.common.JobParams] and [`GpuConfig`][zeus.optimizer.batch_size.common.GpuConfig]
    """

    exp_default_batch_size: int

    min_cost: Optional[float] = None
    min_cost_batch_size: int
    stage: Stage = Stage.Pruning

    mab_random_generator_state: Optional[str] = None

    class Config:
        """Model configuration.

        Allow instantiating the model from an ORM object.
        """

        orm_mode = True
        getter_dict = JobGetter

    @root_validator(skip_on_failure=True)
    def _validate_mab(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate generator state."""
        state: str | None = values["mab_random_generator_state"]
        mab_seed: int | None = values["mab_seed"]

        if mab_seed is not None:
            if state is None:
                raise ValueError("mab_seed is not none, but generator state is none")
            else:
                try:
                    # Check sanity of the generator state.
                    np.random.default_rng(1).__setstate__(json.loads(state))
                except (TypeError, ValueError) as err:
                    raise ValueError(f"Invalid generator state ({state})") from err

        return values
