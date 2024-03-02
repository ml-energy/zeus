import asyncio
import enum
import json
from datetime import datetime
from math import isclose
from uuid import UUID

import numpy as np
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Uuid,
)
from sqlalchemy.ext.asyncio.session import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.sql.sqltypes import VARCHAR
from zeus.optimizer.batch_size.common import JobSpec, MabSetting
from zeus.util.metric import zeus_cost


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "Job"

    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    default_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    high_is_better_metric: Mapped[bool] = mapped_column(Boolean, default=True)
    eta_knob: Mapped[float] = mapped_column(Float, default=0.5)
    beta_knob: Mapped[float] = mapped_column(Float, default=2.0)
    target_metric: Mapped[float] = mapped_column(Float, default=0.5)
    max_epochs: Mapped[int] = mapped_column(Integer, default=100)
    num_pruning_rounds: Mapped[int] = mapped_column(Integer, default=2)
    window_size: Mapped[int] = mapped_column(Integer, default=10)

    max_power: Mapped[float] = mapped_column(Float, nullable=False)
    number_of_gpus: Mapped[int] = mapped_column(Integer, nullable=False)
    gpu_model: Mapped[str] = mapped_column(VARCHAR(length=30), nullable=False)

    mab_prior_mean: Mapped[float] = mapped_column(Float, default=0.0)
    mab_prior_precision: Mapped[float] = mapped_column(Float, default=0.0)
    mab_num_exploration: Mapped[int] = mapped_column(Integer, default=2)
    mab_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mab_random_generator_state: Mapped[str | None] = mapped_column(
        VARCHAR(length=300), nullable=True
    )

    exp_default_batch_size: Mapped[int] = mapped_column(Integer, default=0)

    batch_sizes: Mapped[list["BatchSize"]] = relationship(
        order_by="BatchSize.batch_size.asc()"
        # https://stackoverflow.com/questions/74252768/missinggreenlet-greenlet-spawn-has-not-been-called
        # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#relationship-loading-techniques
    )
    arm_states: Mapped[list["GaussianTsArmState"]] = relationship(
        lazy="joined",
        order_by="GaussianTsArmState.batch_size.asc()",
    )

    def populate_from_job_spec(self, jobSpec: JobSpec):
        self.job_id = jobSpec.job_id
        self.default_batch_size = jobSpec.default_batch_size
        self.high_is_better_metric = jobSpec.high_is_better_metric
        self.eta_knob = jobSpec.eta_knob
        self.beta_knob = jobSpec.beta_knob
        self.target_metric = jobSpec.target_metric
        self.max_epochs = jobSpec.max_epochs
        self.num_pruning_rounds = jobSpec.num_pruning_rounds
        self.window_size = jobSpec.window_size

        self.mab_prior_mean = jobSpec.mab_setting.prior_mean
        self.mab_prior_precision = jobSpec.mab_setting.prior_precision
        self.mab_seed = jobSpec.mab_setting.seed
        self.mab_num_exploration = jobSpec.mab_setting.num_exploration
        if self.mab_seed != None:
            rng = np.random.default_rng(jobSpec.mab_setting.seed)
            self.mab_random_generator_state = json.dumps(rng.__getstate__())

        self.number_of_gpus = jobSpec.number_of_gpus
        self.max_power = jobSpec.max_power
        self.gpu_model = jobSpec.gpu_model
        self.exp_default_batch_size = jobSpec.default_batch_size

    async def get_min_cost(self) -> tuple[float, int]:
        """
        Get min cost with batch size. If no trials have done, return default bs with INF cost
        """
        min_cost = np.inf
        best_bs = self.exp_default_batch_size

        # TODO: Currently, grab all measurements. Do we need to change this to windowed? Anyways only during Exploration stage
        await asyncio.gather(
            *[bs.awaitable_attrs.measurements for bs in self.batch_sizes]
        )

        # If no measurements available, give default bs
        for bs in self.batch_sizes:
            # await bs.awaitable_attrs.measurements
            for measurement in bs.measurements:
                cur_cost = zeus_cost(
                    measurement.energy,
                    measurement.time,
                    self.eta_knob,
                    self.max_power,
                )
                if (
                    (not isclose(min_cost, cur_cost))
                    and min_cost > cur_cost
                    and measurement.converged
                ):
                    min_cost = cur_cost
                    best_bs = bs.batch_size
        return (min_cost, best_bs)

    async def equal_to(self, jobSpec: JobSpec) -> bool:
        await self.awaitable_attrs.batch_sizes
        bss = [bs.batch_size for bs in self.batch_sizes]
        return (
            self.job_id == jobSpec.job_id
            and self.default_batch_size == jobSpec.default_batch_size
            and self.high_is_better_metric == jobSpec.high_is_better_metric
            and self.eta_knob == jobSpec.eta_knob
            and self.beta_knob == jobSpec.beta_knob
            and self.target_metric == jobSpec.target_metric
            and self.max_epochs == jobSpec.max_epochs
            and self.num_pruning_rounds == jobSpec.num_pruning_rounds
            and self.window_size == jobSpec.window_size
            and self.mab_prior_mean == jobSpec.mab_setting.prior_mean
            and self.mab_prior_precision == jobSpec.mab_setting.prior_precision
            and self.mab_seed == jobSpec.mab_setting.seed
            and self.mab_num_exploration == jobSpec.mab_setting.num_exploration
            and self.max_power == jobSpec.max_power
            and self.gpu_model == jobSpec.gpu_model
            and self.number_of_gpus == jobSpec.number_of_gpus
            and bss == jobSpec.batch_sizes
        )

    def get_mab_setting(self) -> MabSetting:
        return MabSetting(
            prior_mean=self.mab_prior_mean,
            prior_precision=self.mab_prior_precision,
            seed=self.mab_seed,
            num_exploration=self.mab_num_exploration,
            random_generator_state=self.mab_random_generator_state,
        )

    def __str__(self) -> str:
        instance_dict = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        instance_dict["batch_sizes"] = [str(bs) for bs in self.batch_sizes]
        instance_dict["arm_states"] = [str(arm) for arm in self.arm_states]
        return str(instance_dict)


class BatchSize(Base):
    __tablename__ = "BatchSize"
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)

    explorations: Mapped[list["ExplorationState"]] = relationship(
        order_by="ExplorationState.trial_number.asc()",
    )
    measurements: Mapped[list["Measurement"]] = relationship(
        order_by="Measurement.timestamp.asc()"
    )

    def __str__(self) -> str:
        instance_dict = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        instance_dict["explorations"] = [str(exp) for exp in self.explorations]
        instance_dict["measurements"] = [str(m) for m in self.measurements]
        return str(instance_dict)


class ExplorationState(Base):
    class State(enum.Enum):
        Exploring = "Exploring"
        Converged = "Converged"
        Unconverged = "Unconverged"

    __tablename__ = "ExplorationState"
    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    trial_number: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    state: Mapped[State] = mapped_column(Enum(State), default=State.Exploring)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    ForeignKeyConstraint(
        [job_id, batch_size], [BatchSize.job_id, BatchSize.batch_size]
    ),

    def __str__(self) -> str:
        instance_dict = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        return str(instance_dict)


class GaussianTsArmState(Base):
    __tablename__ = "GaussianTsArmState"
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)  # arm
    param_mean: Mapped[float] = mapped_column(Float, default=0.0)
    param_precision: Mapped[float] = mapped_column(Float, default=0.0)
    reward_precision: Mapped[float] = mapped_column(Float, default=0.0)
    num_observations: Mapped[int] = mapped_column(Integer, default=0)

    def __str__(self) -> str:
        instance_dict = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        return str(instance_dict)


class Measurement(Base):
    __tablename__ = "Measurement"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[UUID] = mapped_column(Uuid)
    batch_size: Mapped[int] = mapped_column(Integer)
    time: Mapped[float] = mapped_column(Float, nullable=False)
    energy: Mapped[float] = mapped_column(Float, nullable=False)
    converged: Mapped[bool] = mapped_column(Boolean, nullable=False)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )

    ForeignKeyConstraint(
        [job_id, batch_size], [BatchSize.job_id, BatchSize.batch_size]
    ),

    def __str__(self) -> str:
        instance_dict = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        return str(instance_dict)
