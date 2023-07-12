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

import pytest
import torch
from torch.utils.data import Dataset

from zeus.run.dataloader import ZeusDataLoader
from zeus.util.testing import ReplayZeusMonitor

if typing.TYPE_CHECKING:
    from pytest_mock import MockerFixture


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index) -> torch.Tensor:
        return torch.tensor(index)


@pytest.fixture(autouse=True, params=os.listdir("tests/profile_data"))
def mock_zeus_monitor(mocker: MockerFixture, request):
    """Replaces the ZeusMonitor class with ReplayZeusMonitor."""
    def replay_zeus_monitor(*args, **kwargs):
        return ReplayZeusMonitor(
            *args,
            **kwargs,
            log_file="tests/run/profile_data/" + request.param,
            ignore_sync_cuda=True,
        )
    mocker.patch("zeus.run.dataloader.ZeusMonitor", replay_zeus_monitor)


@pytest.fixture(autouse=True)
def mock_zeus_environment_variables(mocker: MockerFixture):
    """Sets the environment variables that ZeusDataLoader uses."""
    mocker.patch.dict("os.environ", {"ZEUS_TARGET_METRIC": "0.99"})


@pytest.fixture(autouse=True)
def mock_pynvml(mocker: MockerFixture):
    """Replaces the pynvml module with a mock."""
    mock = mocker.patch("zeus.run.dataloader.pynvml")
    mock.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (100_000, 300_000)


def training_loop():
    train_loader = ZeusDataLoader(FakeDataset(), batch_size=128, max_epochs=100)
    for _ in train_loader:
        pass
