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

from __future__ import annotations

import os
import typing
from unittest.mock import call

import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from zeus.optimizer import GlobalPowerLimitOptimizer
from zeus.util.testing import ReplayZeusMonitor
from zeus.util.metric import zeus_cost

if typing.TYPE_CHECKING:
    from pathlib import Path
    from pytest_mock import MockerFixture


PROFILE_DATA_DIR = "tests/profile_data/"


def get_dataloader(step_per_epoch: int, batch_size: int = 128) -> DataLoader:
    """Returns a DataLoader that iterates over a fake dataset."""
    class FakeDataset(Dataset):
        def __len__(self) -> int:
            return batch_size * step_per_epoch

        def __getitem__(self, index) -> torch.Tensor:
            return torch.tensor(index)

    return DataLoader(FakeDataset(), batch_size=batch_size)


class ReplayLog:
    """Compute the optimal power limit from the JIT profiling log and the eta knob."""

    def __init__(self, log_file: str) -> None:
        df = typing.cast(pd.DataFrame, pd.read_csv(log_file))
        assert all(col.startswith("gpu") and col.endswith("_energy") for col in df.columns[3:])
        self.gpu_indices = list(map(lambda x: int(x[3]), df.columns[3:]))

        df["energy"] = df[df.columns[3:]].sum(axis=1)  # type: ignore
        df["power_limit"] = df.window_name.map(lambda name: int(name.split("_")[-1]))

        self.df = df
        self.log_file = log_file
        self.power_limits = df.power_limit.to_list()

    def optimal_power_limit(self, eta_knob: float) -> int:
        cost = self.df.apply(lambda row: zeus_cost(
            energy=row["energy"],
            time=row["elapsed_time"],
            eta_knob=eta_knob,
            max_power=max(self.power_limits) * len(self.gpu_indices),
        ), axis=1)
        return int(self.df.iloc[cost.argmin()].power_limit)


@pytest.fixture(params=map(lambda p: PROFILE_DATA_DIR + p, os.listdir(PROFILE_DATA_DIR)))
def replay_log(request) -> ReplayLog:
    return ReplayLog(request.param)


def training_loop(plo: GlobalPowerLimitOptimizer, train_loader: DataLoader, num_epochs: int) -> None:
    """Simulate a training loop."""
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch:02d}")
        plo.on_epoch_begin()
        for step, _ in enumerate(train_loader):
            print(f"Training step {step:03d}")
            plo.on_step_begin()
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
    pynvml_mock = mocker.patch("zeus.optimizer.power_limit.pynvml", autospec=True)
    pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle{i}"
    pynvml_mock.nvmlDeviceGetPowerManagementLimitConstraints.side_effect = \
        lambda _: (min(replay_log.power_limits) * 1000, max(replay_log.power_limits) * 1000)

    # Mock away the atexit hook, which raises an NVML error when testing finishes.
    mocker.patch("zeus.optimizer.power_limit.atexit", autospec=True)

    monitor = ReplayZeusMonitor(
        log_file=replay_log.log_file,
        ignore_sync_cuda=True,
        match_window_name=False,
    )
    assert monitor.gpu_indices == replay_log.gpu_indices

    ############################
    # Test JIT profiling
    ############################

    plo = GlobalPowerLimitOptimizer(
        monitor,
        eta_knob=eta_knob,
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

    training_loop(plo, train_loader, num_epochs=num_epochs)

    # The PLO scans the power limit from the highest to the lowest, and finally
    # sets the optimal power limit. The PLO automatically skips NVML calls when
    # the current power limit matches the target power limit, so we can just
    # assert unique power limit setting calls.
    call_list = []
    optimal_pl = replay_log.optimal_power_limit(eta_knob)
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
                call_list.append(call(f"handle{i}", max(replay_log.power_limits) * 1000))
            # Return the power limit back to the list for retry.
            power_limits.insert(0, power_limit)
    # If the final power limit tried was the optimal one, the PLO will not set it again.
    if power_limit != optimal_pl:
        for i in sorted(monitor.gpu_indices):
            call_list.append(call(f"handle{i}", optimal_pl * 1000))
    pynvml_mock.nvmlDeviceSetPowerManagementLimit.assert_has_calls(call_list, any_order=False)
    pynvml_mock.reset_mock()

    ########################################
    # Test loading from saved profile data
    ########################################
    plo = GlobalPowerLimitOptimizer(
        monitor,
        eta_knob=eta_knob,
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
