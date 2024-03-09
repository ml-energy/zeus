from copy import deepcopy
from uuid import UUID
from sqlalchemy import and_, select, update

from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpdateExploration,
)

from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationStateModel,
    ExplorationsPerBs,
    ExplorationsPerJob,
    GaussianTsArmStateModel,
    MeasurementOfBs,
    MeasurementsPerBs,
)
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import (
    ExplorationState,
    GaussianTsArmState,
    Job,
    Measurement,
)


class BatchSizeStateRepository(DatabaseRepository):

    async def get_measurements_of_bs(
        self, batch_size: BatchSizeBase, window_size: int
    ) -> MeasurementsPerBs:
        # Load window size amount of measurement for that bs
        try:
            if window_size == 0:
                return []
            stmt = (
                select(Measurement)
                .where(
                    and_(
                        Measurement.job_id == batch_size.job_id,
                        Measurement.batch_size == batch_size.batch_size,
                    )
                )
                .order_by(Measurement.timestamp.desc())
                .limit(window_size)
            )
            res = (await self.session.scalars(stmt)).all()
            return MeasurementsPerBs(
                job_id=batch_size.job_id,
                batch_size=batch_size.batch_size,
                measurements=[MeasurementOfBs.from_orm(m) for m in res],
            )
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_measurements_of_bs: {str(err)}")
            raise err

    async def get_explorations_of_job(self, job_id: UUID) -> ExplorationsPerJob:
        try:
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

        except Exception as err:
            await self.session.rollback()
            self._log(f"get_explorations_of_job: {str(err)}")
            raise err

    async def get_arms(self, job_id: UUID) -> list[GaussianTsArmStateModel]:
        # This list should be "good" arms
        try:
            stmt = select(GaussianTsArmState).where(GaussianTsArmState.job_id == job_id)
            res = (await self.session.scalars(stmt)).all()
            return [GaussianTsArmStateModel.from_orm(arm) for arm in res]
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_measurements_of_bs: {str(err)}")
            raise err

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmStateModel | None:
        try:
            stmt = select(GaussianTsArmState).where(
                and_(
                    GaussianTsArmState.job_id == bs.job_id,
                    GaussianTsArmState.batch_size == bs.batch_size,
                )
            )
            arm = await self.session.scalar(stmt)
            if arm == None:
                return None
            return GaussianTsArmStateModel.from_orm(arm)
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_measurements_of_bs: {str(err)}")
            raise err

    def add_exploration(self, exploration: CreateExploration) -> None:
        self.session.add(exploration.to_orm())

    async def update_exploration(self, updated_exp: UpdateExploration) -> None:
        try:
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
        except Exception as err:
            await self.session.rollback()
            self._log(f"update_exploration: {str(err)}")
            raise err

    def create_arms(self, new_arms: list[GaussianTsArmStateModel]) -> None:
        self.session.add_all([arm.to_orm() for arm in new_arms])

    async def update_arm_state(
        self, updated_mab_state: GaussianTsArmStateModel
    ) -> None:
        try:
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
        except Exception as err:
            await self.session.rollback()
            self._log(f"update_arm_state: {str(err)}")
            raise err

    async def get_explorations_of_bs(self, bs: BatchSizeBase) -> ExplorationsPerBs:
        try:
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
                explorations=[
                    ExplorationStateModel.from_orm(exp) for exp in explorations
                ],
            )
        except Exception as err:
            await self.session.rollback()
            self._log(f"get_measurements_of_bs: {str(err)}")
            raise err

    def add_measurement(self, measurement: MeasurementOfBs) -> None:
        self.session.add(measurement.to_orm())

    def _log(self, msg: str):
        print(f"[BatchSizeStateRepository] {msg}")
