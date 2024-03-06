from uuid import UUID
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpsertGaussianTsArmState,
    UpdateExploration,
    UpdateGaussianTsArmState,
)

from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    BatchSizeStates,
    ExplorationOfBs,
    ExplorationsPerJob,
    GaussianTsArmStateModel,
    MeasurementOfBs,
    MeasurementsPerBs,
)
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository


class BatchSizeStateRepository(DatabaseRepository):

    async def get_measurements_of_bs(
        self, batch_size: BatchSizeBase, window_size: int
    ) -> MeasurementsPerBs:
        pass  # Load window size amount of measurement for that bs

    async def get_explorations_of_job(self, job_id: UUID) -> ExplorationsPerJob:

        pass

    async def get_arms(self, job_id: UUID) -> list[GaussianTsArmStateModel]:
        # This list should be "good" arms
        pass

    def add_exploration(self, exploration: CreateExploration) -> None:
        # Create ArmState for that bs
        pass

    async def update_exploration(self, update: UpdateExploration) -> None:
        pass

    def create_arms(self, new_arms: list[GaussianTsArmStateModel]) -> None:
        # Create ArmState for that bs
        pass

    async def update_arm_state(self, update: GaussianTsArmStateModel) -> None:
        # Update ArmState for that bs
        pass

    def add_measurement(self, measurement: MeasurementOfBs) -> None:
        # add measurement to that batch_size isntance
        pass
