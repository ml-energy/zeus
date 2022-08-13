# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the Job specification dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from zeus.util import LinearScaler, SquareRootScaler


@dataclass(frozen=True, unsafe_hash=True)
class Job:
    """Job specification tuple.

    Attributes:
        dataset: Name of the dataset.
        network: Name of the DNN model.
        optimizer: Name of the optimizer, e.g. Adam.
        target_metric: Target validation metric.
        max_epochs: Maximum number of epochs to train before terminating.
        default_bs: Initial batch size (b0) provided by the user.
        default_lr: Learning rate corresponding to the default batch size.
        workdir: Working directory in which to launch the job command.
        command: Job command template. See [`gen_command`][zeus.job.Job.gen_command].
    """

    dataset: str
    network: str
    optimizer: str
    target_metric: float
    max_epochs: int
    default_bs: int | None = None
    default_lr: float | None = None
    workdir: str | None = None
    command: list[str] | None = field(default=None, hash=False, compare=False)

    def __str__(self) -> str:
        """Generate a more conside representation of the object."""
        return (
            f"Job({self.dataset},{self.network},{self.optimizer},{self.target_metric}"
            f"{f',bs{self.default_bs}' if self.default_bs is not None else ''}~{self.max_epochs})"
        )

    def to_logdir(self) -> str:
        """Generate a logdir name that explains this job."""
        return (
            f"{self.dataset}+{self.network}+bs{self.default_bs}"
            f"+{self.optimizer}+lr{self.default_lr}"
            f"+tm{self.target_metric}+me{self.max_epochs}"
        )

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pick out the rows corresponding to this job from the DataFrame."""
        return df.loc[
            (df.dataset == self.dataset)
            & (df.network == self.network)
            & (df.optimizer == self.optimizer)
            & (df.target_metric == self.target_metric)
        ]

    def gen_command(
        self,
        batch_size: int,
        learning_rate: float,
        seed: int,
        rec_i: int,
    ) -> list[str]:
        """Format the job command with given arguments.

        Args:
            batch_size: Batch size to use for this job launch.
            learning_rate: Learning rate to use for this job launch.
            seed: Random seed to use for this job launch.
            rec_i: Recurrence number of this job launch.
        """
        assert self.command, "You must provide a command format string for this job."
        command = []
        for piece in self.command:
            if piece in ["{bs}", "{batch_size}"]:
                command.append(str(batch_size))
            elif piece in ["{lr}", "{learning_rate}"]:
                command.append(str(learning_rate))
            elif piece == "{seed}":
                command.append(str(seed))
            elif piece in ["{epoch}", "{epochs}"]:
                command.append(str(self.max_epochs))
            elif piece == "{slice_number}":
                command.append(str(rec_i))
            elif piece == "{target_metric}":
                command.append(str(self.target_metric))
            else:
                command.append(piece)
        return command

    def scale_lr(self, batch_size: int) -> float:
        """Scale the learning rate for the given batch size.

        Assumes that `self.default_bs` and `self.default_lr` were given.
        Then, `self.default_lr` is scaled for the given `batch_size` using
        square root scaling for adaptive optimizers (e.g. Adam, Adadelta,
        AdamW) and linear scaling for others (e.g. SGD).
        """
        assert self.default_bs, "You must provide default_bs to scale LR."
        assert self.default_lr, "You must provide default_lr to scale LR."

        optimizer = self.optimizer.lower()
        if optimizer in ["adam", "adadelta", "adamw"]:
            scaler = SquareRootScaler(bs=self.default_bs, lr=self.default_lr)
            return scaler.compute_lr(batch_size)
        if optimizer in ["sgd"]:
            scaler = LinearScaler(bs=self.default_bs, lr=self.default_lr)
            return scaler.compute_lr(batch_size)
        raise NotImplementedError(f"LR scaling for {self.optimizer} is not supported.")
