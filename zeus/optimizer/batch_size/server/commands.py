from dataclasses import Field
from typing import Any
from uuid import UUID
from pydantic import BaseModel
from zeus.optimizer.batch_size.server.batch_size_state.models import ExplorationsPerBs
from zeus.optimizer.batch_size.server.job.models import JobState

from pydantic.class_validators import root_validator


class CreateArms(BaseModel):
    explorations: dict[int, ExplorationsPerBs] = Field(min_items=1)

    job: JobState

    @root_validator
    def _validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        exps: dict[int, ExplorationsPerBs] = values.get("explorations")
        job: JobState = values.get("job")

        any_at_num_pruning_rounds = False

        for bs, exp_list_per_bs in exps.items():
            for exp in exp_list_per_bs.explorations:
                if exp.trial_number > job.num_pruning_rounds:
                    raise ValueError(
                        f"Cannot have more than num_pruning_rounds({job.num_pruning_rounds}) of trials"
                    )
                elif exp.trial_number == job.num_pruning_rounds:
                    any_at_num_pruning_rounds = True

                if bs != exp.batch_size:
                    raise ValueError(f"Batch size should be consistent in explorations")
                if job.job_id != exp.job_id:
                    raise ValueError(f"Job_id should be consistent in explorations")

        if any_at_num_pruning_rounds == False:
            raise ValueError(
                f"At least one exploration should have {job.num_pruning_rounds} number of trials"
            )

        return values
