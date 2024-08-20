from __future__ import annotations

import pytest

from lowtime.operation import (
    CandidateExecutionOptions,
    DummyOperation,
    ExecutionOption,
    OperationSpec,
    Operation,
)


@pytest.fixture
def mock_spec() -> OperationSpec:
    unit_time = 1.0
    spec = OperationSpec(
        options=CandidateExecutionOptions(
            options=[
                ExecutionOption[str](
                    real_time=123.0,
                    unit_time=unit_time,
                    cost=3.0,
                    knob="one",
                ),
                ExecutionOption[str](
                    real_time=789.0,
                    unit_time=unit_time,
                    cost=1.0,
                    knob="three",
                ),
                ExecutionOption[str](
                    real_time=456.0,
                    unit_time=unit_time,
                    cost=2.0,
                    knob="two",
                ),
                ExecutionOption[str](
                    real_time=234.0,
                    unit_time=unit_time,
                    cost=3.1,
                    knob="four",
                ),
            ]
        ),
        cost_model=None,  # type: ignore
    )
    return spec


def test_operation_spec_option_filtering(mock_spec: OperationSpec) -> None:
    assert len(mock_spec.options.options) == 3
    assert set(o.knob for o in mock_spec.options.options) == set(
        ["one", "two", "three"]
    )
    assert set(o.quant_time for o in mock_spec.options.options) == set([123, 456, 789])


def test_dummy_operation() -> None:
    op = DummyOperation()
    assert op.is_dummy

    # Dummy operations have no spec.
    with pytest.raises(AttributeError):
        op.spec

    # No knob for execution either.
    with pytest.raises(AttributeError):
        op.assigned_knob

    # Setting their duration should not invoke the assigned_knob setter.
    op.duration = 123
    assert op.duration == 123


def test_operation_computed_fields(mock_spec: OperationSpec) -> None:
    op = Operation(spec=mock_spec)
    assert op.min_duration == 123
    assert op.max_duration == 789
    assert not op.is_dummy


def test_knob_assignment(mock_spec: OperationSpec) -> None:
    # Slowest knob by default.
    oper = Operation(spec=mock_spec)
    assert oper.assigned_knob == "three"

    def op_assign_assert(duration: int, knob) -> None:
        op = Operation(spec=mock_spec)
        op.duration = duration
        assert op.assigned_knob == knob

    op_assign_assert(20, "one")
    op_assign_assert(122, "one")
    op_assign_assert(123, "one")
    op_assign_assert(124, "one")
    op_assign_assert(234, "one")
    op_assign_assert(455, "one")
    op_assign_assert(456, "two")
    op_assign_assert(457, "two")
    op_assign_assert(458, "two")
    op_assign_assert(654, "two")
    op_assign_assert(788, "two")
    op_assign_assert(789, "three")
    op_assign_assert(790, "three")
    op_assign_assert(1234, "three")
