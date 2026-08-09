"""Tests for the power monitor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zeus.monitor.power import infer_counter_update_period

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_infers_half_the_fastest_counter_period(mocker: MockerFixture) -> None:
    """The fastest probed counter period is halved and returned.

    Regression test for the accumulator being seeded with `0.0` and folded with
    `min`, which pinned the result at `0.0` for every input.
    """
    gpus = mocker.MagicMock()
    gpus.get_name.side_effect = lambda index: ["A40", "V100"][index]
    mocker.patch("zeus.monitor.power.get_gpus", return_value=gpus)
    mocker.patch(
        "zeus.monitor.power._infer_counter_update_period_single",
        side_effect=lambda index: [0.4, 0.12][index],
    )

    # 0.12 s is the fastest counter, so poll twice per update at 0.06 s.
    assert infer_counter_update_period([0, 1]) == pytest.approx(0.06)
