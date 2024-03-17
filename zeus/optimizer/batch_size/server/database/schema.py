from __future__ import annotations
import enum
from datetime import datetime
from typing import Optional
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
from sqlalchemy.ext.asyncio.session import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.sql.sqltypes import VARCHAR
from zeus.optimizer.batch_size.server.job.models import Stage


class Base(DeclarativeBase):
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
    mab_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    mab_random_generator_state: Mapped[Optional[str]] = mapped_column(
        VARCHAR(length=300), nullable=True
    )
    exp_default_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)

    stage: Mapped[Stage] = mapped_column(Enum(Stage), default=Stage.Pruning)
    min_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)

    batch_sizes: Mapped[list["BatchSize"]] = relationship(
        order_by="BatchSize.batch_size.asc()",
        back_populates="job",
        lazy="joined",
        # https://stackoverflow.com/questions/74252768/missinggreenlet-greenlet-spawn-has-not-been-called
        # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#relationship-loading-techniques
    )


class BatchSize(Base):
    __tablename__ = "BatchSize"
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)

    explorations: Mapped[list["ExplorationState"]] = relationship(
        back_populates="batch_size_state",
        order_by="ExplorationState.round_number.asc()",
    )
    measurements: Mapped[list["Measurement"]] = relationship(
        back_populates="batch_size_state", order_by="Measurement.timestamp.asc()"
    )

    arm_state: Mapped[Optional["GaussianTsArmState"]] = relationship(
        back_populates="batch_size_state",  # populates GaussianTsArmState->BatchSize
        # https://stackoverflow.com/questions/398697
        # 93/when-do-i-need-to-use-sqlalchemy-back-populates#:~:text=didn%27t%20provide%20a-,back_populates,-argument%2C%20modifying%20one
    )

    job: Mapped["Job"] = relationship(back_populates="batch_sizes")


class State(enum.Enum):
    Exploring = "Exploring"
    Converged = "Converged"
    Unconverged = "Unconverged"


class ExplorationState(Base):
    __tablename__ = "ExplorationState"
    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False, primary_key=True)
    state: Mapped[State] = mapped_column(Enum(State), default=State.Exploring)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    batch_size_state: Mapped["BatchSize"] = relationship(back_populates="explorations")

    __table_args__ = (
        ForeignKeyConstraint(
            [job_id, batch_size], [BatchSize.job_id, BatchSize.batch_size]
        ),
    )


class GaussianTsArmState(Base):
    __tablename__ = "GaussianTsArmState"
    job_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)  # arm

    param_mean: Mapped[float] = mapped_column(Float, default=0.0)
    param_precision: Mapped[float] = mapped_column(Float, default=0.0)
    reward_precision: Mapped[float] = mapped_column(Float, default=0.0)
    num_observations: Mapped[int] = mapped_column(Integer, default=0)

    batch_size_state: Mapped["BatchSize"] = relationship(back_populates="arm_state")

    __table_args__ = (
        ForeignKeyConstraint(
            [job_id, batch_size], [BatchSize.job_id, BatchSize.batch_size]
        ),
    )


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

    batch_size_state: Mapped["BatchSize"] = relationship(back_populates="measurements")

    __table_args__ = (
        ForeignKeyConstraint(
            [job_id, batch_size], [BatchSize.job_id, BatchSize.batch_size]
        ),
    )
