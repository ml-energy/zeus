"""Commands on how to use some methods from the `ZeusService`."""

from zeus.utils.pydantic_v1 import BaseModel
from zeus.optimizer.batch_size.server.batch_size_state.commands import ReadTrial
from zeus.optimizer.batch_size.server.batch_size_state.models import GaussianTsArmState


class GetRandomChoices(BaseModel):
    """Parameters for getting a random choices.

    Attributes:
        job_id: Job Id
        choices: List of choices
    """

    job_id: str
    choices: list[int]


class GetNormal(BaseModel):
    """Parameters for getting a random sample from normal distribution.

    Attributes:
        job_id: Job id
        loc: Mean
        scale: Stdev
    """

    job_id: str
    loc: float
    scale: float


class UpdateArm(BaseModel):
    """Parameters to update an arm.

    Attributes:
        trial: Identifier of trial
        updated_arm: Updated state of arm.
    """

    trial: ReadTrial
    updated_arm: GaussianTsArmState
