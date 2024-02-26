"""
ORM models 

"""

import enum
from datetime import datetime
from typing import List
from uuid import UUID
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
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from zeus.optimizer.batch_size.common import JobSpec


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "Job"

    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    default_batch_size: Mapped[int] = mapped_column(Integer)
    high_is_better_metric: Mapped[bool] = mapped_column(Boolean, default=True)
    eta_knob: Mapped[float] = mapped_column(Float, default=0.5)
    beta_knob: Mapped[float] = mapped_column(Float, default=2.0)
    target_metric: Mapped[float] = mapped_column(Float, default=0.5)
    max_epochs: Mapped[int] = mapped_column(Integer, default=100)
    num_pruning_rounds: Mapped[int] = mapped_column(Integer, default=2)
    window_size: Mapped[int] = mapped_column(Integer, default=0)

    mab_prior_mean: Mapped[float] = mapped_column(Float, default=0.0)
    mab_prior_precision: Mapped[float] = mapped_column(Float, default=0.0)
    mab_seed: Mapped[int] = mapped_column(Integer, default=123456)
    mab_num_exploration: Mapped[int] = mapped_column(Integer, default=2)

    batch_sizes: Mapped[List["BatchSize"]] = relationship()
    arm_states: Mapped[List["GaussianTsArmState"]] = relationship()

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

    def equal_to(self, jobSpec: JobSpec) -> bool:
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
        )


class State(enum.Enum):
    Exploring = "Exploring"
    Converged = "Converged"
    Unconverged = "Unconverged"
    Unexplored = "Unexplored"


class BatchSize(Base):  # TODO: better naming?
    __tablename__ = "BatchSize"
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    exploration_state: Mapped[State] = mapped_column(
        Enum(State), default=State.Unexplored
    )
    exploration_count: Mapped[int] = mapped_column(Integer, default=0)

    measurements: Mapped[List["Measurement"]] = relationship()


class GaussianTsArmState(Base):
    __tablename__ = "GaussianTsArmState"
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)  # arm
    arm_param_mean: Mapped[float] = mapped_column(Float, default=0.0)
    arm_param_prec: Mapped[float] = mapped_column(Float, default=0.0)
    reward_precision: Mapped[float] = mapped_column(Float, default=0.0)
    arm_num_observations: Mapped[float] = mapped_column(Float, default=0.0)


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
