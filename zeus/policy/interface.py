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

"""Abstract classes for implementing custom optimization policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from zeus.job import Job


class BatchSizeOptimizer(ABC):
    """Finds out the best batch size to use for the job."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the batch size optimizer."""

    @abstractmethod
    def register_job(self, job: Job, batch_sizes: list[int]) -> None:
        """Prepare internal state so that it can handle the given job.

        It is assumed that the state of each [`Job`][zeus.job.Job] will be
        managed separately. Note that [`Job`][zeus.job.Job] is hashable,
        and thus can be used as dictionary keys.

        Args:
            job: New jobs to register.
            batch_sizes: Batch sizes to consider.
        """

    @abstractmethod
    def predict(self, job: Job) -> int:
        """Return the best batch size to use for the job.

        Args:
            job: The job to pick the best batch size for.
        """

    @abstractmethod
    def observe(
        self, job: Job, batch_size: int, cost: float, converged: bool | None = None
    ) -> None:
        """Observe the cost of using the given batch size for the job.

        Args:
            job: The job from which this cost observation resulted.
            batch_size: The batch size used for this run of the job.
            cost: The energy-time cost of running the job.
            converged: Whether the job reached its target metric. If may not have
                reached its target if the job was early stopped based on cost or
                the maximum epoch was reached. For BSO's that do not take this into
                account, `None` can be passed.
        """

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[{self.name}] {message}")


class PowerLimitOptimizer(ABC):
    """Finds out the best power limit to use for the job and batch size."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the power limit optimizer."""

    @abstractmethod
    def predict(self, job: Job, batch_size: int) -> int | None:
        """Return the best power limit for the job and batch size.

        Args:
            job: The job to pick the best power limit for.
            batch_size: The batch size chosen by the
                [`BatchSizeOptimizer`][zeus.policy.BatchSizeOptimizer] for this job.

        Returns:
            The best power limit, or `None` if profiling results via
            [`observe`][zeus.policy.interface.PowerLimitOptimizer.observe] are needed.
        """

    @abstractmethod
    def observe(self, job: Job, batch_size: int, power_limit: int, cost: float) -> None:
        """Observe the cost of using the given batch size and power limit for the job.

        Args:
            job: The job from which this cost observation resulted.
            batch_size: The batch size used for this run of the job.
            power_limit: The power limit used for this run of the job.
            cost: The cost of running the job.
        """

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[{self.name}] {message}")
