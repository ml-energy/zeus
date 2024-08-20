# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
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

"""Operations on the task DAG.

An `Operation` is a node on the task DAG. It can choose from multiple
`ExecutionOption`s to execute itself, represented by `knob` values.
Each `ExecutionOption` has a different time and cost expenditure. This
information is captured by the `OperationSpec` of the operation. Then,
multiple `Operation`s can share the same `OperationSpec` if they have
the same set of `ExecutionOption`s.
"""

from __future__ import annotations

import bisect
import logging
import sys
from functools import cached_property
from typing import Generic, TypeVar, TYPE_CHECKING

from attrs import define, field
from attrs.setters import frozen

from lowtime.cost_model import CostModel

if TYPE_CHECKING:
    from attr import Attribute


logger = logging.getLogger(__name__)

KnobT = TypeVar("KnobT")


@define(repr=False, slots=False)
class ExecutionOption(Generic[KnobT]):
    """One option for executing an operation.

    An operation can be executed with one among multiple possible knobs,
    and the execution expenditure of choosing this particular `knob` value is
    represented by this class.

    Attributes:
        real_time: The wall clock time it took to execute the operation.
        unit_time: The time unit used to quantize `real_time`.
        quant_time: Integer-quantized time (`int(real_time // unit_time)`).
        cost: The cost of the operation.
        knob: The knob value associated with this option.
    """

    real_time: float
    unit_time: float
    cost: float
    knob: KnobT

    @cached_property
    def quant_time(self) -> int:
        """Return the quantized time of this option."""
        return int(self.real_time // self.unit_time)

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"ExecutionOption(knob={self.knob}, quant_time={self.quant_time}, "
            f"cost={self.cost}, real_time={self.real_time}, unit_time={self.unit_time})"
        )


@define
class CandidateExecutionOptions(Generic[KnobT]):
    """A list of selected candidate execution options for an operation.

    Candidate execution options are filtered from execution options given to __init__:
    1. Filter Pareto-optimal options based on `real_time` and `cost`.
    2. Deduplicate `quant_time` by keeping only the option with the largest `cost`.
        This is because time quantization is inherently rounding down, so the closest
        `quant_time` is the one with the largest `cost`.
    3. Sort by `quant_time` in descending order.

    Args:
        options: All candidate execution options of the operation.
        noise_factor: A factor to multiply `real_time` and `cost` by to allow some slack.
    """

    options: list[ExecutionOption[KnobT]]
    noise_factor: float = 1.0
    _knob_cache: dict[int, KnobT] = field(init=False, repr=False, factory=dict)

    def __attrs_post_init__(self) -> None:
        """Return a new `ExecutionOptions` object with only Pareto-optimal options."""
        # Find and only leave Pareto-optimal options.
        orig_options = sorted(self.options, key=lambda x: x.real_time, reverse=True)
        filtered_options: list[ExecutionOption[KnobT]] = []
        for option in orig_options:
            real_time = option.real_time * self.noise_factor
            cost = option.cost * self.noise_factor
            if any(
                other.real_time < real_time and other.cost < cost for other in orig_options
            ):
                continue
            filtered_options.append(option)

        # There may be multiple options with the same `quant_time`.
        # Only leave the option with the largest `cost` because that's the one whose
        # `quant_time` is closest to `real_time`.
        filtered_options.sort(key=lambda x: x.cost, reverse=True)
        orig_options, filtered_options = filtered_options, []
        quant_times = set()
        for option in orig_options:
            if option.quant_time in quant_times:
                continue
            filtered_options.append(option)
            quant_times.add(option.quant_time)

        # Sort by `quant_time` in descending order.
        self.options = filtered_options
        self.options.sort(key=lambda x: x.quant_time, reverse=True)

    def get_knob_for(self, quant_time: int) -> KnobT:
        """Find the slowest `knob` value that still executes within `quant_time`."""
        try:
            return self._knob_cache[quant_time]
        except KeyError:
            pass

        # Run binary search on options.
        sorted_options = list(reversed(self.options))
        i = bisect.bisect_right([o.quant_time for o in sorted_options], quant_time)
        # Just in case `quant_time` is smaller than the smallest `quant_time`.
        # We select the knob for the smallest `quant_time` in this case.
        i = max(1, i)
        knob = sorted_options[i - 1].knob

        # Cache the result.
        self._knob_cache[quant_time] = knob
        return knob


@define
class OperationSpec(Generic[KnobT]):
    """An operation spec with multiple Pareto-optimal (time, cost) execution options.

    In the computation DAG, there may be multiple operations with the same type
    that share the set of execution options and cost model. This class specifies the
    type of operation (i.e., spec), and actual operations on the DAG hold a reference
    to its operation spec.

    Attributes:
        options: Candidate execution options of this operation.
        cost_model: A continuous cost model fit from candidate execution options.
    """

    options: CandidateExecutionOptions[KnobT]
    cost_model: CostModel


def knob_setter(self: Operation, _: Attribute, duration: int) -> int:
    """Find and assign the slowest `knob` value that still meets `duration`."""
    self.assigned_knob = self.spec.options.get_knob_for(duration)
    return duration


@define(slots=False)
class Operation(Generic[KnobT]):
    """Base class for operations on the computation DAG.

    `__repr__` will display all object fields, so should only be used for debugging.
    On the other hand, `__str__` is for a human readable representation of the object.

    Attributes:
        spec: The operation spec of this operation.
        is_dummy: Whether this operation is a dummy operation. See `DummyOperation`.
        duration: The current planned duration, in quanitzed time.
        max_duration: The maximum duration of this operation, in quantized time.
        min_duration: The minimum duration of this operation, in quantized time.
        assigned_knob: The knob value chosen for the value of `duration`.
        earliest_start: The earliest time this operation can start, in quantized time.
        latest_start: The latest time this operation can start, in quantized time.
        earliest_finish: The earliest time this operation can finish, in quantized time.
        latest_finish: The latest time this operation can finish, in quantized time.
    """

    spec: OperationSpec[KnobT] = field(on_setattr=frozen)
    is_dummy: bool = field(default=False, init=False, on_setattr=frozen)

    duration: int = field(init=False, on_setattr=knob_setter)
    max_duration: int = field(init=False)
    min_duration: int = field(init=False)
    assigned_knob: KnobT = field(init=False)

    earliest_start: int = field(default=0, init=False)
    latest_start: int = field(default=sys.maxsize, init=False)
    earliest_finish: int = field(default=0, init=False)
    latest_finish: int = field(default=sys.maxsize, init=False)

    def __attrs_post_init__(self) -> None:
        """Compute derived attributes."""
        quant_times = [o.quant_time for o in self.spec.options.options]
        self.max_duration = max(quant_times)
        self.min_duration = min(quant_times)

        # By default, execute with the slowest speed. `assigned_knob` will
        # automatically be set by `duration_setter`.
        self.duration = self.max_duration

    def __str__(self) -> str:
        """Default implementation that shows current duration (number) and knob (@)."""
        return f"Operation({self.duration}@{self.assigned_knob})"

    def get_cost(self, duration: int | None = None) -> float:
        """Return the cost prediction of this operation at its current duration."""
        return self.spec.cost_model(self.duration if duration is None else duration)

    def reset_times(self) -> None:
        """Reset earliest/latest start/finish attributes to their default values."""
        self.earliest_start = 0
        self.latest_start = sys.maxsize
        self.earliest_finish = 0
        self.latest_finish = sys.maxsize


@define(slots=False)
class DummyOperation(Operation):
    """A dummy operation that does nothing.

    An `AttributeError` is raised when you try to access `spec`.
    """

    spec: OperationSpec = field(init=False, repr=False, on_setattr=frozen)
    is_dummy: bool = field(default=True, init=False, on_setattr=frozen)

    # Delete the duration setter, which accesses `spec`.
    duration: int = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Compute derived attributes."""
        self.max_duration = sys.maxsize
        self.min_duration = 0
        self.duration = 0

    def __str__(self) -> str:
        """Return a readable string representation."""
        return "DummyOperation()"

    def get_cost(self, duration: int | None = None) -> float:
        """No cost for dummy operations."""
        raise AttributeError("DummyOperation has no cost.")
