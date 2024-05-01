"""Commands to use `JobStateRepository`."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from zeus.utils.pydantic_v1 import root_validator, validator, Field, BaseModel
from zeus.optimizer.batch_size.common import GpuConfig, JobSpecFromClient, JobParams
from zeus.optimizer.batch_size.server.database.schema import BatchSizeTable, JobTable
from zeus.optimizer.batch_size.server.job.models import Stage


class UpdateExpDefaultBs(BaseModel):
    """Parameters to update the exploration default batch size.

    Attributes:
        job_id: Job Id.
        exp_default_batch_size: new default batch size to use.
    """

    job_id: str
    exp_default_batch_size: int = Field(gt=0)


class UpdateJobStage(BaseModel):
    """Parameters to update the job stage.

    Attributes:
        job_id: Job Id.
        stage: Set it to MAB since we only go from Pruning to MAB.
    """

    job_id: str
    stage: Stage = Field(Stage.MAB, const=True)


class UpdateGeneratorState(BaseModel):
    """Parameters to update the generator state.

    Attributes:
        job_id: Job Id.
        state: Generator state.
    """

    job_id: str
    state: str

    @validator("state")
    def _validate_state(cls, state: str) -> str:
        """Validate the sanity of state."""
        try:
            np.random.default_rng(1).__setstate__(json.loads(state))
            return state
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid generator state ({state})") from err


class UpdateJobMinCost(BaseModel):
    """Parameters to update the min training cost and corresponding batch size.

    Attributes:
        job_id: Job Id.
        min_cost: Min training cost.
        min_cost_batch_size: Corresponding batch size.
    """

    job_id: str
    min_cost: float = Field(ge=0)
    min_cost_batch_size: int = Field(gt=0)


class CreateJob(GpuConfig, JobParams):
    """Parameters to create a new job.

    Attributes:
        exp_default_batch_size: Exploration default batch size that is used during Pruning stage.
        min_cost: Min training cost observed. Initially, None.
        min_cost_batch_size: Batch size that has minimum training cost observed.
        stage: Stage of the job.
        mab_random_generator_state: Generator state if mab_seed is not None. Otherwise, None.

    For the rest of attributes, refer to `JobParams`[zeus.optimizer.batch_size.common.JobParams] and `GpuConfig`[zeus.optimizer.batch_size.common.GpuConfig]
    """

    exp_default_batch_size: int
    min_cost: None = Field(None, const=True)
    min_cost_batch_size: int
    stage: Stage = Field(Stage.Pruning, const=True)
    mab_random_generator_state: Optional[str] = None

    class Config:
        """Model configuration.

        Make it immutable after creation.
        """

        frozen = True

    @root_validator(skip_on_failure=True)
    def _validate_states(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate Job states.

        We are checking,
            - If mab seed and generator state is matching.
            - If default, exp_default, min batch sizes are correctly intialized.
            - If default batch size is in the list of batch sizes.
        """
        state: str | None = values["mab_random_generator_state"]
        mab_seed: int | None = values["mab_seed"]
        bss: list[int] = values["batch_sizes"]
        dbs: int = values["default_batch_size"]
        ebs: int = values["exp_default_batch_size"]
        mbs: int = values["min_cost_batch_size"]

        if mab_seed is not None:
            if state is None:
                raise ValueError("mab_seed is not none, but generator state is none")
            else:
                try:
                    np.random.default_rng(1).__setstate__(json.loads(state))
                except (TypeError, ValueError) as err:
                    raise ValueError(f"Invalid generator state ({state})") from err

        if not (dbs == ebs == mbs):
            raise ValueError(
                f"During initialization, default_batch_size({dbs}), exp_default_batch_size({ebs}), min_batch_size({mbs}) should be all the same"
            )
        if dbs not in bss:
            raise ValueError(
                f"default_batch_size({dbs}) is not in the batch size list({bss})"
            )

        return values

    @classmethod
    def from_job_config(cls, js: JobSpecFromClient) -> "CreateJob":
        """From JobConfig, instantiate `CreateJob`.

        Initialize generator state, exp_default_batch_size, and min_cost_batch_size.
        """
        d = js.dict()
        d["exp_default_batch_size"] = js.default_batch_size
        if js.mab_seed is not None:
            rng = np.random.default_rng(js.mab_seed)
            d["mab_random_generator_state"] = json.dumps(rng.__getstate__())
        d["min_cost_batch_size"] = js.default_batch_size
        return cls.parse_obj(d)

    def to_orm(self) -> JobTable:
        """Convert pydantic model `CreateJob` to ORM object Job."""
        d = self.dict()
        job = JobTable()
        for k, v in d.items():
            if k != "batch_sizes":
                setattr(job, k, v)
        job.batch_sizes = [
            BatchSizeTable(job_id=self.job_id, batch_size=bs) for bs in self.batch_sizes
        ]
        return job
