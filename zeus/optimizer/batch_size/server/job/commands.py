import json
from uuid import UUID
import numpy as np
from pydantic.main import BaseModel
from zeus.optimizer.batch_size.common import JobSpec
from zeus.optimizer.batch_size.server.database.schema import BatchSize, Job
from zeus.optimizer.batch_size.server.job.models import Stage


class UpdateExpDefaultBs(BaseModel):
    job_id: UUID
    exp_default_batch_size: int


class UpdateJobStage(BaseModel):
    job_id: UUID
    stage: Stage


class UpdateJobMinCost(BaseModel):
    job_id: UUID
    min_cost: float
    min_batch_size: int


class CreateJob(JobSpec):
    exp_default_batch_size: int
    min_cost: float | None = None
    min_batch_size: int
    stage: Stage = Stage.Pruning
    mab_random_generator_state: str | None = None

    # TODO: Validate generator state is not empty when seed is not empty
    # batch_size_states = []

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
