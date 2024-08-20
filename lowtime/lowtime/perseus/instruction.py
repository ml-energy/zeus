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

"""An instruction is an atomic an operation in pipeline training."""

from __future__ import annotations

from typing import Sequence, Type, Callable, get_type_hints

from attrs import define

from lowtime.operation import Operation


@define(slots=False, kw_only=True)
class Instruction(Operation[int]):
    """An operation on a pipeline training schedule."""

    stage_id: int
    micro_batch_id: int

    def __str__(self) -> str:
        """Return a human-readable string representation of the instruction."""
        return (
            f"{type(self).__name__}(S{self.stage_id}B{self.micro_batch_id}, "
            f"{self.duration}@{self.assigned_knob})"
        )


@define
class Forward(Instruction):
    """Forward computation for a pipeline stage."""


@define
class Backward(Instruction):
    """Backward computation for a pipeline stage."""


@define
class ForwardBackward(Instruction):
    """ForwardBackward computation for a pipeline stage."""


@define
class Recomputation(Instruction):
    """Activation recomputation (forward) for a pipeline stage."""


def forward_dep(inst1: Forward, inst2: Forward) -> bool:
    """Dependency rule between Forward instructions.

    Forward(stage i, microbatch j) -> Forward(stage i+1, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id + 1 == inst2.stage_id
    )


def backward_dep(inst1: Backward, inst2: Backward) -> bool:
    """Dependency rule between Backward instructions.

    Backward(stage i+1, microbatch j) -> Backward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


def forwardbackward_dep(inst1: ForwardBackward, inst2: ForwardBackward) -> bool:
    """Dependency rule between ForwardBackward instructions.

    ForwardBackward(stage i+1, microbatch j) -> ForwardBackward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


def forwardbackward_backward_dep(inst1: ForwardBackward, inst2: Backward) -> bool:
    """Dependency rule between ForwardBackward and Backward.

    ForwardBackward(stage i+1, microbatch j) -> Backward(stage i, microbatch j)
    """
    return (
        inst1.micro_batch_id == inst2.micro_batch_id
        and inst1.stage_id == inst2.stage_id + 1
    )


class DependencyResolver:
    """Finds whether two operations are dependent.

    Given a sequence of dependency rules, this class checks whether two
    operations should have a dependency edge between them in the DAG.
    Dependency rules are functions that take two operations and return
    a boolean, True if there is a dependency and False otherwise.
    """

    def __init__(
        self, dependency_rules: Sequence[Callable[..., bool]], node_type: Type
    ) -> None:
        """Initialize the dependency manager with dependency rules.

        Args:
            dependency_rules: Sequence of dependency rules. Each rule is a type-annotated
                function that takes two operations and returns a boolean.
            node_type: The base type of nodes in the DAG.
        """
        arg_types = []
        for rule in dependency_rules:
            type_hints = get_type_hints(rule)

            # We'll forgive missing return types.
            if "return" in type_hints:
                type_hints.pop("return")

            # Exactly two input arguments.
            if len(type_hints) != 2:
                raise ValueError("Dependency rules must have exactly two arguments.")

            # Cache type hints.
            op1_t, op2_t = type_hints.values()
            arg_types.append((op1_t, op2_t))

            # Both input argumens must be Instructions.
            if not issubclass(op1_t, node_type):
                raise ValueError("First argument is not a subclass of Instruction.")
            if not issubclass(op2_t, node_type):
                raise ValueError("Second argument is not a subclass of Instruction.")

        self.rules = dependency_rules
        self.arg_types = arg_types

    def is_dependent(self, op1: Instruction, op2: Instruction) -> bool:
        """Check if there is a dependency from `op1` and `op2`."""
        for rule, (op1_t, op2_t) in zip(self.rules, self.arg_types):
            if isinstance(op1, op1_t) and isinstance(op2, op2_t):
                result = rule(op1, op2)
                if not isinstance(result, bool):
                    raise RuntimeError("Dependency rule returned a non-boolean value.")
                if result:
                    return True
        return False
