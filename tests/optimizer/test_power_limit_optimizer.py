from __future__ import annotations

import inspect
import os
import typing
from typing import Generator, Iterable
from unittest.mock import call

import pytest
import pandas as pd

from zeus.device.gpu import get_gpus
from zeus.optimizer.power_limit import (
    GlobalPowerLimitOptimizer,
    HFGlobalPowerLimitOptimizer,
    Ready,
    Done,
    Energy,
    Time,
    ZeusCost,
    MaxSlowdownConstraint,
)
from zeus.utils.testing import ReplayZeusMonitor
from zeus.utils.metric import zeus_cost

if typing.TYPE_CHECKING:
    from pathlib import Path
    from pytest_mock import MockerFixture


PROFILE_DATA_DIR = "tests/profile_data/"


def get_dataloader(step_per_epoch: int) -> Iterable[int]:
    """Returns a DataLoader that iterates over a fake dataset."""

    class DataLoader:
        def __init__(self, step_per_epoch: int) -> None:
            self.step_per_epoch = step_per_epoch

        def __iter__(self) -> Generator[int, None, None]:
            for i in range(self.step_per_epoch):
                yield i

    return DataLoader(step_per_epoch)


class ReplayLog:
    """Compute the optimal power limit from the JIT profiling log and the eta knob."""

    def __init__(self, log_file: str) -> None:
        df = typing.cast(pd.DataFrame, pd.read_csv(log_file))
        assert all(
            col.startswith("gpu") and col.endswith("_energy") for col in df.columns[3:]
        )
        self.gpu_indices = list(map(lambda x: int(x[3]), df.columns[3:]))

        df["energy"] = df[df.columns[3:]].sum(axis=1)  # type: ignore
        df["power_limit"] = df.window_name.map(lambda name: int(name.split("_")[-1]))

        self.df = df
        self.log_file = log_file
        self.power_limits = df.power_limit.to_list()

    def optimal_zeus_cost_power_limit(self, eta_knob: float) -> int:
        cost = self.df.apply(
            lambda row: zeus_cost(
                energy=row["energy"],
                time=row["elapsed_time"],
                eta_knob=eta_knob,
                max_power=max(self.power_limits) * len(self.gpu_indices),
            ),
            axis=1,
        )
        return int(self.df.iloc[cost.argmin()].power_limit)

    def optimal_time_power_limit(self) -> int:
        return int(self.df.iloc[self.df.elapsed_time.argmin()].power_limit)

    def optimal_energy_power_limit(self) -> int:
        return int(self.df.iloc[self.df.energy.argmin()].power_limit)

    def optimal_max_slowdown_constraint_power_limit(self, factor: float) -> int:
        shortest_time = self.df.query(
            f"power_limit == {max(self.power_limits)}"
        ).elapsed_time.item()
        filtered_df = self.df.query(f"elapsed_time <= {shortest_time * factor}")
        return int(filtered_df.power_limit.min())


@pytest.fixture(
    params=map(lambda p: PROFILE_DATA_DIR + p, os.listdir(PROFILE_DATA_DIR))
)
def replay_log(request) -> ReplayLog:
    return ReplayLog(request.param)


def training_loop(
    plo: GlobalPowerLimitOptimizer,
    train_loader: Iterable[int],
    num_epochs: int,
    wait_steps: int = 0,
) -> None:
    """Simulate a training loop."""
    for epoch in range(num_epochs):
        plo.on_epoch_begin()
        for step, _ in enumerate(train_loader):
            print(f"Epoch {epoch:03d} step {step:03d}")
            plo.on_step_begin()
            if wait_steps > 0 and epoch == 0 and step < wait_steps:
                assert isinstance(plo.state, Ready)
            plo.on_step_end()
        plo.on_epoch_end()


@pytest.mark.parametrize("eta_knob", [0.00, 0.25, 0.50, 0.75, 1.00])
@pytest.mark.parametrize("num_interrupts", [0, 1, 2, "every time"])
def test_power_limit_optimizer(
    mocker: MockerFixture,
    replay_log: ReplayLog,
    eta_knob: float,
    num_interrupts: int | str,
    tmp_path: Path,
):
    # Mock PyNVML.
    pynvml_mock = mocker.patch("zeus.device.gpu.nvidia.pynvml", autospec=True)
    pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle{i}"
    pynvml_mock.nvmlDeviceGetPowerManagementLimitConstraints.side_effect = lambda _: (
        min(replay_log.power_limits) * 1000,
        max(replay_log.power_limits) * 1000,
    )

    # Mock away the atexit hook, which raises an NVML error when testing finishes.
    mocker.patch("zeus.optimizer.power_limit.atexit", autospec=True)

    monitor = ReplayZeusMonitor(
        log_file=replay_log.log_file,
        ignore_sync_execution=True,
        match_window_name=False,
    )
    assert monitor.gpu_indices == replay_log.gpu_indices

    ############################
    # Test JIT profiling
    ############################

    # Disable `SYS_ADMIN` capability check.
    get_gpus()._disable_sys_admin_warning = True

    plo = GlobalPowerLimitOptimizer(
        monitor,
        optimum_selector=ZeusCost(
            eta_knob=eta_knob,
            world_size=len(monitor.gpu_indices),
        ),
        wait_steps=1,
        warmup_steps=10,
        profile_steps=40,
        pl_step=25,
        profile_path=tmp_path / "power_limit_optimizer.json",
    )

    if num_interrupts == 0:
        # Never interrupted.
        pls_per_epoch = len(replay_log.power_limits)
        train_loader = get_dataloader(step_per_epoch=50 * pls_per_epoch + 10)
        num_epochs = 3
    elif num_interrupts == 1:
        # Interrupted during warmup.
        pls_per_epoch = len(replay_log.power_limits) // 2
        train_loader = get_dataloader(step_per_epoch=50 * pls_per_epoch + 5)
        num_epochs = 4
    elif num_interrupts == 2:
        # Interrupted during profiling.
        pls_per_epoch = len(replay_log.power_limits) // 3
        train_loader = get_dataloader(step_per_epoch=50 * pls_per_epoch + 15)
        num_epochs = 5
    elif num_interrupts == "every time":
        # Interrupted for every power limit, on the last profiling step.
        pls_per_epoch = 1
        train_loader = get_dataloader(step_per_epoch=50 * pls_per_epoch + 50)
        num_epochs = len(replay_log.power_limits) + 2
    else:
        raise ValueError(f"Unexpected number of interrupts: {num_interrupts}")

    training_loop(plo, train_loader, num_epochs=num_epochs, wait_steps=1)

    # The PLO scans the power limit from the highest to the lowest, and finally
    # sets the optimal power limit. The PLO automatically skips NVML calls when
    # the current power limit matches the target power limit, so we can just
    # assert unique power limit setting calls.
    call_list = []
    optimal_pl = replay_log.optimal_zeus_cost_power_limit(eta_knob)
    power_limit = -1
    power_limits = sorted(replay_log.power_limits, reverse=True)
    # JIT profiling stage with potential interrupts.
    while power_limits:
        # One epoch, going through each power limit.
        # The final power limit gets interrupted.
        # This loop always runs at least once, so `power_limit` is always bound.
        for _ in range(pls_per_epoch + 1):
            try:
                power_limit = power_limits.pop(0)
            except IndexError:
                break
            for i in sorted(monitor.gpu_indices):
                call_list.append(call(f"handle{i}", power_limit * 1000))
        # The else clause runs if the for loop was not interrupted with `break`.
        else:
            # After being interrupted, the PLO will set the power limit to the max.
            for i in sorted(monitor.gpu_indices):
                call_list.append(
                    call(f"handle{i}", max(replay_log.power_limits) * 1000)
                )
            # Return the power limit back to the list for retry.
            power_limits.insert(0, power_limit)
    # If the final power limit tried was the optimal one, the PLO will not set it again.
    if power_limit != optimal_pl:
        for i in sorted(monitor.gpu_indices):
            call_list.append(call(f"handle{i}", optimal_pl * 1000))
    pynvml_mock.nvmlDeviceSetPowerManagementLimit.assert_has_calls(
        call_list, any_order=False
    )
    pynvml_mock.reset_mock()

    # Print out the profile data for debugging purposes.
    with open(tmp_path / "power_limit_optimizer.json", "r") as f:
        print(f.read())

    ########################################
    # Test optimum power limit selection
    ########################################
    temp_plo = GlobalPowerLimitOptimizer(
        monitor,
        optimum_selector=Energy(),
        profile_path=tmp_path / "power_limit_optimizer.json",
    )
    assert isinstance(temp_plo.state, Done)
    assert (
        temp_plo.state.optimal_power_limit
        == replay_log.optimal_energy_power_limit() * 1000
    )

    temp_plo = GlobalPowerLimitOptimizer(
        monitor,
        optimum_selector=Time(),
        profile_path=tmp_path / "power_limit_optimizer.json",
    )
    assert isinstance(temp_plo.state, Done)
    assert (
        temp_plo.state.optimal_power_limit
        == replay_log.optimal_time_power_limit() * 1000
    )

    for factor in [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 100.0]:
        temp_plo = GlobalPowerLimitOptimizer(
            monitor,
            optimum_selector=MaxSlowdownConstraint(factor=factor),
            profile_path=tmp_path / "power_limit_optimizer.json",
        )
        assert isinstance(temp_plo.state, Done)
        assert (
            temp_plo.state.optimal_power_limit
            == replay_log.optimal_max_slowdown_constraint_power_limit(factor) * 1000
        )

    ########################################
    # Test loading from saved profile data
    ########################################
    plo = GlobalPowerLimitOptimizer(
        monitor,
        optimum_selector=ZeusCost(
            eta_knob=eta_knob,
            world_size=len(monitor.gpu_indices),
        ),
        warmup_steps=10,
        profile_steps=40,
        pl_step=25,
        profile_path=tmp_path / "power_limit_optimizer.json",
    )

    training_loop(plo, train_loader, num_epochs=1)

    pynvml_mock.nvmlDeviceSetPowerManagementLimit.assert_has_calls(
        [call(f"handle{i}", optimal_pl * 1000) for i in sorted(monitor.gpu_indices)],
        any_order=False,
    )


def test_HFGPLO_signature_equivalence() -> None:
    """Ensure that the constructor signatures of GlobalPowerLimitOptimizer and HFGlobalPowerLimitOptimizer are equivalent."""
    gplo_signature = inspect.signature(GlobalPowerLimitOptimizer.__init__)
    hfgplo_signature = inspect.signature(HFGlobalPowerLimitOptimizer.__init__)

    assert (
        gplo_signature == hfgplo_signature
    ), "GlobalPowerLimitOptimizer and HFGlobalPowerLimitOptimizer signatures do not match."
