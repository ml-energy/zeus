"""Repository for batch size states(batch size, exploration, measurement, Gaussian Ts arm state)."""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from sqlalchemy import and_, select, update
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpdateExploration,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerBs,
    ExplorationsPerJob,
    ExplorationStateModel,
    GaussianTsArmStateModel,
    MeasurementOfBs,
    MeasurementsPerBs,
)
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import (
    ExplorationState,
    GaussianTsArmState,
    Measurement,
)


class BatchSizeStateRepository(DatabaseRepository):
    """Repository for handling batch size related operations."""

    async def get_measurements_of_bs(
        self, batch_size: BatchSizeBase, window_size: int
    ) -> MeasurementsPerBs:
        """Load window size amount of measurement for a given batch size. If window size <= 0, load all measurements.

        Args:
            batch_size (BatchSizeBase): The batch size object.
            window_size (int): The size of the measurement window.

        Returns:
            MeasurementsPerBs: Measurements for the given batch size.
        """
        stmt = (
            select(Measurement)
            .where(
                and_(
                    Measurement.job_id == batch_size.job_id,
                    Measurement.batch_size == batch_size.batch_size,
                )
            )
            .order_by(Measurement.timestamp.desc())
        )
        if window_size > 0:
            stmt = stmt.limit(window_size)

        res = (await self.session.scalars(stmt)).all()
        return MeasurementsPerBs(
            job_id=batch_size.job_id,
            batch_size=batch_size.batch_size,
            measurements=[MeasurementOfBs.from_orm(m) for m in res],
        )

    async def get_explorations_of_job(self, job_id: UUID) -> ExplorationsPerJob:
        """Retrieve explorations for a given job.

        Args:
            job_id (UUID): The ID of the job.

        Returns:
            ExplorationsPerJob: Explorations for the given job.
        """
        stmt = (
            select(ExplorationState)
            .where(
                and_(
                    ExplorationState.job_id == job_id,
                )
            )
            .order_by(ExplorationState.batch_size.asc())
        )
        res = (await self.session.scalars(stmt)).all()

        explorations_per_bs: dict[int, ExplorationsPerBs] = {}
        exps: list[ExplorationStateModel] = []
        for exp in res:
            if len(exps) == 0 or exps[0].batch_size == exp.batch_size:
                exps.append(ExplorationStateModel.from_orm(exp))
            else:
                explorations_per_bs[exps[0].batch_size] = ExplorationsPerBs(
                    job_id=job_id,
                    batch_size=exps[0].batch_size,
                    explorations=deepcopy(exps),
                )

                exps = [ExplorationStateModel.from_orm(exp)]
        if len(exps) != 0:
            explorations_per_bs[exps[0].batch_size] = ExplorationsPerBs(
                job_id=job_id,
                batch_size=exps[0].batch_size,
                explorations=deepcopy(exps),
            )

        return ExplorationsPerJob(
            job_id=job_id, explorations_per_bs=explorations_per_bs
        )

    async def get_arms(self, job_id: UUID) -> list[GaussianTsArmStateModel]:
        """Retrieve Gaussian Thompson Sampling arms for a given job.

        Args:
            job_id (UUID): The ID of the job.

        Returns:
            List[GaussianTsArmStateModel]: List of Gaussian Thompson Sampling arms. These arms are all "good" arms (converged during pruning stage).
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        stmt = select(GaussianTsArmState).where(GaussianTsArmState.job_id == job_id)
        res = (await self.session.scalars(stmt)).all()
        return [GaussianTsArmStateModel.from_orm(arm) for arm in res]

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmStateModel | None:
        """Retrieve Gaussian Thompson Sampling arm for a given job id and batch size.

        Args:
            bs (BatchSizeBase): The batch size object.

        Returns:
            Optional[GaussianTsArmStateModel]: Gaussian Thompson Sampling arm if found, else None.
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        stmt = select(GaussianTsArmState).where(
            and_(
                GaussianTsArmState.job_id == bs.job_id,
                GaussianTsArmState.batch_size == bs.batch_size,
            )
        )
        arm = await self.session.scalar(stmt)
        if arm is None:
            return None
        return GaussianTsArmStateModel.from_orm(arm)

    def add_exploration(self, exploration: CreateExploration) -> None:
        """Add an exploration to the data base.

        Refer to `CreateExploration`[zeus.optimizer.batch_size.server.batch_size_state.models.CreateExploration] for attributes.

        Args:
            exploration (CreateExploration): The exploration to add.
        """
        self.session.add(exploration.to_orm())

    async def update_exploration(self, updated_exp: UpdateExploration) -> None:
        """Update an exploration in the database.

        Args:
            updated_exp (UpdateExploration): The updated exploration.
            Refer to `UpdateExploration`[zeus.optimizer.batch_size.server.batch_size_state.models.UpdateExploration] for attributes.
        """
        stmt = (
            update(ExplorationState)
            .where(
                and_(
                    ExplorationState.job_id == updated_exp.job_id,
                    ExplorationState.batch_size == updated_exp.batch_size,
                    ExplorationState.round_number == updated_exp.round_number,
                )
            )
            .values(state=updated_exp.state, cost=updated_exp.cost)
        )
        await self.session.execute(stmt)

    def create_arms(self, new_arms: list[GaussianTsArmStateModel]) -> None:
        """Create Gaussian Thompson Sampling arms in the database.

        Args:
            new_arms (List[GaussianTsArmStateModel]): List of new arms to create.
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        self.session.add_all([arm.to_orm() for arm in new_arms])

    async def update_arm_state(
        self, updated_mab_state: GaussianTsArmStateModel
    ) -> None:
        """Update Gaussian Thompson Sampling arm state in db.

        Args:
            updated_mab_state (GaussianTsArmStateModel): The updated arm state.
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        stmt = (
            update(GaussianTsArmState)
            .where(
                and_(
                    GaussianTsArmState.job_id == updated_mab_state.job_id,
                    GaussianTsArmState.batch_size == updated_mab_state.batch_size,
                )
            )
            .values(
                param_mean=updated_mab_state.param_mean,
                param_precision=updated_mab_state.param_precision,
                reward_precision=updated_mab_state.reward_precision,
                num_observations=updated_mab_state.num_observations,
            )
        )
        await self.session.execute(stmt)

    async def get_explorations_of_bs(self, bs: BatchSizeBase) -> ExplorationsPerBs:
        """Retrieve explorations for a given job id and batch size.

        Args:
            bs (BatchSizeBase): Job id and batch size.

        Returns:
            ExplorationsPerBs: Explorations for the given batch size.
            Refer to `ExplorationsPerBs`[zeus.optimizer.batch_size.server.batch_size_state.models.ExplorationsPerBs] for attributes.
        """
        stmt = (
            select(ExplorationState)
            .where(
                and_(
                    ExplorationState.job_id == bs.job_id,
                    ExplorationState.batch_size == bs.batch_size,
                )
            )
            .order_by(ExplorationState.round_number.desc())
        )

        explorations = (await self.session.scalars(stmt)).all()
        return ExplorationsPerBs(
            job_id=bs.job_id,
            batch_size=bs.batch_size,
            explorations=[ExplorationStateModel.from_orm(exp) for exp in explorations],
        )

    def add_measurement(self, measurement: MeasurementOfBs) -> None:
        """Add a measurement to db.

        Args:
            measurement (MeasurementOfBs): The measurement to add.
            Refer to `MeasurementOfBs`[zeus.optimizer.batch_size.server.batch_size_state.models.MeasurementOfBs] for attributes.
        """
        self.session.add(measurement.to_orm())
