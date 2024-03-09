from typing import Any
from uuid import UUID
from pydantic import BaseModel
from pydantic.fields import Field
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerBs,
)
from zeus.optimizer.batch_size.server.job.models import JobState

from pydantic.class_validators import root_validator


class CreateArms(BaseModel):
    explorations_per_bs: dict[int, ExplorationsPerBs]

    job_id: UUID

    @root_validator
    def _validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        exps: dict[int, ExplorationsPerBs] = values.get("explorations_per_bs")
        job_id: UUID = values.get("job_id")

        for bs, exp_list_per_bs in exps.items():
            for exp in exp_list_per_bs.explorations:
                if bs != exp.batch_size:
                    raise ValueError(f"Batch size should be consistent in explorations")
                if job_id != exp.job_id:
                    raise ValueError(f"Job_id should be consistent in explorations")

        return values

    def validate_exp_rounds(self, num_pruning_rounds: int):
        any_at_num_pruning_rounds = False
        print("IS THIS NONE", self.explorations_per_bs)
        for bs, exp_list_per_bs in self.explorations_per_bs.items():
            for exp in exp_list_per_bs.explorations:
                if exp.round_number > num_pruning_rounds:
                    raise ValueError(
                        f"Cannot have more than num_pruning_rounds({num_pruning_rounds}) of trials"
                    )
                elif exp.round_number == num_pruning_rounds:
                    any_at_num_pruning_rounds = True

                if bs != exp.batch_size:
                    raise ValueError(f"Batch size should be consistent in explorations")
        if any_at_num_pruning_rounds == False:
            raise ValueError(
                f"At least one exploration should have {num_pruning_rounds} number of trials"
            )


class GetRandomChoices(BaseModel):
    job_id: UUID
    choices: list[int]


class GetNormal(BaseModel):
    job_id: UUID
    loc: float
    scale: float
