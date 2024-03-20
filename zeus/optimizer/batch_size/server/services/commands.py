"""Commands on how to use some methods from the `ZeusService`."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel
from pydantic.class_validators import root_validator
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    ExplorationsPerBs,
)


class CompletedExplorations(BaseModel):
    """Pydantic model for completed explorations.

    Attributes:
        explorations_per_bs: For each batch size, list of explorations
        job_id: Job Id
    """

    explorations_per_bs: dict[int, ExplorationsPerBs]
    job_id: UUID

    @root_validator(skip_on_failure=True)
    def _validate(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check the sanity of exploration_per_bs. Each exploration state should be consistent in terms of job_id and batch size."""
        exps: dict[int, ExplorationsPerBs] = values["explorations_per_bs"]
        job_id: UUID = values["job_id"]

        for bs, exp_list_per_bs in exps.items():
            for exp in exp_list_per_bs.explorations:
                if bs != exp.batch_size:
                    raise ValueError("Batch size should be consistent in explorations")
                if job_id != exp.job_id:
                    raise ValueError("Job_id should be consistent in explorations")

        return values

    def validate_exp_rounds(self, num_pruning_rounds: int):
        """Validate explorations in terms of job's pruning rounds configuration.

        This function validates:
            - If there is a batch size that have more than num_pruning_rounds number of rounds.
            - If at least one batch size has num_pruning_rounds number of rounds.
        """
        any_at_num_pruning_rounds = False
        for _, exp_list_per_bs in self.explorations_per_bs.items():
            for exp in exp_list_per_bs.explorations:
                if exp.round_number > num_pruning_rounds:
                    raise ValueError(
                        f"Cannot have more than num_pruning_rounds({num_pruning_rounds}) of rounds"
                    )
                elif exp.round_number == num_pruning_rounds:
                    any_at_num_pruning_rounds = True

        if not any_at_num_pruning_rounds:
            raise ValueError(
                f"At least one exploration should have {num_pruning_rounds} number of rounds"
            )


class GetRandomChoices(BaseModel):
    """Parameters for getting a random choices.

    Attributes:
        job_id: Job Id
        choices: List of choices
    """

    job_id: UUID
    choices: list[int]


class GetNormal(BaseModel):
    """Parameters for getting a random sample from normal distribution.

    Attributes:
        job_id: Job id
        loc: Mean
        scale: Stdev
    """

    job_id: UUID
    loc: float
    scale: float
