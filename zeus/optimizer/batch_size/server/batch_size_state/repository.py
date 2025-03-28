"""Repository for batch size states(Trial, Gaussian Ts arm state)."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    GaussianTsArmState,
    Trial,
    TrialResult,
    TrialResultsPerBs,
)
from zeus.optimizer.batch_size.server.database.repository import DatabaseRepository
from zeus.optimizer.batch_size.server.database.schema import (
    GaussianTsArmStateTable,
    TrialStatus,
    TrialTable,
    TrialType,
)
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError
from zeus.utils.logging import get_logger

logger = get_logger(__name__)


class BatchSizeStateRepository(DatabaseRepository):
    """Repository for handling batch size related operations."""

    def __init__(self, session: AsyncSession):
        """Set db session and intialize fetched trial. We are only updating one trial per session."""
        super().__init__(session)
        self.fetched_trial: TrialTable | None = None
        self.fetched_arm: GaussianTsArmStateTable | None = None

    async def get_next_trial_number(self, job_id: str) -> int:
        """Get next trial number of a given job. Trial number starts from 1 and increase by 1 at a time."""
        stmt = select(func.max(TrialTable.trial_number)).where(
            and_(
                TrialTable.job_id == job_id,
            )
        )
        res = await self.session.scalar(stmt)
        if res is None:
            return 1
        return res + 1

    async def get_trial_results_of_bs(
        self, batch_size: BatchSizeBase, window_size: int
    ) -> TrialResultsPerBs:
        """Load window size amount of results for a given batch size. If window size <= 0, load all of them.

        From all trials, we filter succeeded one since failed/dispatched ones doesn't have a valid result.

        Args:
            batch_size (BatchSizeBase): The batch size object.
            window_size (int): The size of the measurement window.

        Returns:
            TrialResultsPerBs: trial results for the given batch size.
        """
        stmt = (
            select(TrialTable)
            .where(
                and_(
                    TrialTable.job_id == batch_size.job_id,
                    TrialTable.batch_size == batch_size.batch_size,
                    TrialTable.status == TrialStatus.Succeeded,
                )
            )
            .order_by(TrialTable.trial_number.desc())
        )
        if window_size > 0:
            stmt = stmt.limit(window_size)

        res = (await self.session.scalars(stmt)).all()
        return TrialResultsPerBs(
            job_id=batch_size.job_id,
            batch_size=batch_size.batch_size,
            results=[TrialResult.from_orm(t) for t in res],
        )

    async def get_arms(self, job_id: str) -> list[GaussianTsArmState]:
        """Retrieve Gaussian Thompson Sampling arms for a given job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            List[GaussianTsArmStateModel]: List of Gaussian Thompson Sampling arms. These arms are all "good" arms (converged during pruning stage).
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        stmt = select(GaussianTsArmStateTable).where(
            GaussianTsArmStateTable.job_id == job_id
        )
        res = (await self.session.scalars(stmt)).all()
        return [GaussianTsArmState.from_orm(arm) for arm in res]

    async def get_arm(self, bs: BatchSizeBase) -> GaussianTsArmState | None:
        """Retrieve Gaussian Thompson Sampling arm for a given job id and batch size.

        Args:
            bs (BatchSizeBase): The batch size object.

        Returns:
            Optional[GaussianTsArmStateModel]: Gaussian Thompson Sampling arm if found, else None.
            Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        stmt = select(GaussianTsArmStateTable).where(
            and_(
                GaussianTsArmStateTable.job_id == bs.job_id,
                GaussianTsArmStateTable.batch_size == bs.batch_size,
            )
        )
        arm = await self.session.scalar(stmt)
        if arm is None:
            return None
        self.fetched_arm = arm
        return GaussianTsArmState.from_orm(arm)

    async def get_trial(self, trial: ReadTrial) -> Trial | None:
        """Get a corresponding trial.

        Args:
            trial: job_id, batch_size, trial_number.

        Returns:
            Found Trial. If none found, return None.
        """
        stmt = select(TrialTable).where(
            TrialTable.job_id == trial.job_id,
            TrialTable.batch_size == trial.batch_size,
            TrialTable.trial_number == trial.trial_number,
        )
        fetched_trial = await self.session.scalar(stmt)

        if fetched_trial is None:
            logger.info("get_trial: NoResultFound")
            return None

        self.fetched_trial = fetched_trial
        return Trial.from_orm(fetched_trial)

    def get_trial_from_session(self, trial: ReadTrial) -> Trial | None:
        """Fetch a trial from the session."""
        if (
            self.fetched_trial is None
            or self.fetched_trial.job_id != trial.job_id
            or self.fetched_trial.batch_size != trial.batch_size
            or self.fetched_trial.trial_number != trial.trial_number
        ):
            return None
        return Trial.from_orm(self.fetched_trial)

    def create_trial(self, trial: CreateTrial) -> None:
        """Create a trial in db.

        Refer to `CreateTrial`[zeus.optimizer.batch_size.server.batch_size_state.models.CreateTrial] for attributes.

        Args:
            trial (CreateTrial): The trial to add.
        """
        self.session.add(trial.to_orm())

    def updated_current_trial(self, updated_trial: UpdateTrial) -> None:
        """Update trial in the database (report the result of trial).

        Args:
            updated_trial (UpdateTrial): The updated trial. Refer to `UpdateTrial`[zeus.optimizer.batch_size.server.batch_size_state.models.UpdateTrial] for attributes.
        """
        if self.fetched_trial is None:
            raise ZeusBSOValueError("No trial is fetched.")

        if (
            self.fetched_trial.job_id != updated_trial.job_id
            or self.fetched_trial.batch_size != updated_trial.batch_size
            or self.fetched_trial.trial_number != updated_trial.trial_number
        ):
            raise ZeusBSOValueError("Trying to update invalid trial.")

        self.fetched_trial.end_timestamp = updated_trial.end_timestamp
        self.fetched_trial.status = updated_trial.status
        self.fetched_trial.time = updated_trial.time
        self.fetched_trial.energy = updated_trial.energy
        self.fetched_trial.converged = updated_trial.converged

    def create_arms(self, new_arms: list[GaussianTsArmState]) -> None:
        """Create Gaussian Thompson Sampling arms in the database.

        Args:
            new_arms (List[GaussianTsArmStateModel]): List of new arms to create.
                Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        self.session.add_all([arm.to_orm() for arm in new_arms])

    def update_arm_state(self, updated_mab_state: GaussianTsArmState) -> None:
        """Update Gaussian Thompson Sampling arm state in db.

        Args:
            updated_mab_state (GaussianTsArmStateModel): The updated arm state.
                Refer to `GaussianTsArmStateModel`[zeus.optimizer.batch_size.server.batch_size_state.models.GaussianTsArmStateModel] for attributes.
        """
        if self.fetched_arm is None:
            raise ZeusBSOValueError("No arm is fetched.")

        if (
            self.fetched_arm.job_id != updated_mab_state.job_id
            or self.fetched_arm.batch_size != updated_mab_state.batch_size
        ):
            raise ZeusBSOValueError(
                "Fetch arm does not correspond with the arm trying to update."
            )

        self.fetched_arm.param_mean = updated_mab_state.param_mean
        self.fetched_arm.param_precision = updated_mab_state.param_precision
        self.fetched_arm.reward_precision = updated_mab_state.reward_precision
        self.fetched_arm.num_observations = updated_mab_state.num_observations

    async def get_explorations_of_job(self, job_id: str) -> ExplorationsPerJob:
        """Retrieve succeeded or ongoing explorations for a given job.

        Args:
            job_id: ID of the job

        Returns:
            ExplorationsPerJob: Explorations for the given batch size.
            Refer to `ExplorationsPerJob`[zeus.optimizer.batch_size.server.batch_size_state.models.ExplorationsPerJob] for attributes.
        """
        stmt = (
            select(TrialTable)
            .where(
                and_(
                    TrialTable.job_id == job_id,
                    TrialTable.type == TrialType.Exploration,
                    TrialTable.status != TrialStatus.Failed,
                )
            )
            .order_by(TrialTable.trial_number.asc())
        )

        explorations = (await self.session.scalars(stmt)).all()
        exps_per_bs: defaultdict[int, list[Trial]] = defaultdict(list)
        for exp in explorations:
            exps_per_bs[exp.batch_size].append(Trial.from_orm(exp))

        return ExplorationsPerJob(job_id=job_id, explorations_per_bs=exps_per_bs)
