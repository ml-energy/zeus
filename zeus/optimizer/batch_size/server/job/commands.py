from __future__ import annotations
import json
from typing import Any, Optional
from uuid import UUID

import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.fields import Field
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.common import JobSpec
from zeus.optimizer.batch_size.server.database.schema import BatchSize, Job
from zeus.optimizer.batch_size.server.job.models import Stage


class UpdateExpDefaultBs(BaseModel):
    job_id: UUID
    exp_default_batch_size: int = Field(gt=0)


class UpdateJobStage(BaseModel):
    job_id: UUID
    stage: Stage = Field(Stage.MAB, const=True)


class UpdateGeneratorState(BaseModel):
    job_id: UUID
    state: str

    @validator("state")
    def _validate_state(cls, state: str) -> str:
        try:
            np.random.default_rng(1).__setstate__(json.loads(state))
            return state
        except (TypeError, ValueError):
            raise ValueError(f"Invalid generator state ({state})")


class UpdateJobMinCost(BaseModel):
    job_id: UUID
    min_cost: float = Field(ge=0)
    min_batch_size: int = Field(gt=0)


class CreateJob(JobSpec):
    exp_default_batch_size: int
    min_cost: None = Field(None, const=True)
    min_batch_size: int
    stage: Stage = Field(Stage.Pruning, const=True)
    mab_random_generator_state: Optional[str] = None

    class Config:
        frozen = True
        validate_assignment = True

    @root_validator
    def _validate_mab_states(cls, values: dict[str, Any]) -> dict[str, Any]:
        state: str | None = values.get("mab_random_generator_state")
        mab_seed: int | None = values.get("mab_seed")
        bss: list[int] = values.get("batch_sizes")
        dbs: int = values.get("default_batch_size")
        ebs: int = values.get("exp_default_batch_size")
        mbs: int = values.get("min_batch_size")

        if mab_seed != None:
            if state == None:
                raise ValueError("mab_seed is not none, but generator state is none")
            else:
                try:
                    np.random.default_rng(1).__setstate__(json.loads(state))
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid generator state ({state})")

        if not (dbs == ebs == mbs):
            raise ValueError(
                f"During initialization, default_batch_size({dbs}), exp_default_batch_size({ebs}), min_batch_size({mbs}) should be all the same"
            )
        if dbs not in bss:
            raise ValueError(
                f"default_batch_size({dbs}) is not in the batch size list({bss})"
            )

        return values

    def from_jobSpec(js: JobSpec) -> "CreateJob":
        d = js.dict()
        d["exp_default_batch_size"] = js.default_batch_size
        if js.mab_seed != None:
            rng = np.random.default_rng(js.mab_seed)
            d["mab_random_generator_state"] = json.dumps(rng.__getstate__())
        d["min_batch_size"] = js.default_batch_size
        return CreateJob.parse_obj(d)

    def to_orm(self) -> Job:
        # Only takes care of job state. Defer batch sizes to batchSizeState
        d = self.dict()
        job = Job()
        for k, v in d.items():
            if k != "batch_sizes":
                setattr(job, k, v)
        job.batch_sizes = [
            BatchSize(job_id=self.job_id, batch_size=bs) for bs in self.batch_sizes
        ]
        return job
