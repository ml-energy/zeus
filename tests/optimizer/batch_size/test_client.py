from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from zeus.monitor.energy import Measurement, ZeusMonitor
from zeus.optimizer.batch_size.client import BatchSizeOptimizer
from zeus.optimizer.batch_size.common import JobSpec
from zeus.optimizer.batch_size.exceptions import (
    ZeusBSOBadOperationError,
    ZeusBSOTrainFailError,
)


@pytest.fixture
def mock_monitor(mocker: MockerFixture):
    mocker.patch("pynvml.nvmlInit")

    zeus_monitor_mock_instance = MagicMock(spec=ZeusMonitor)
    zeus_monitor_mock_instance.gpu_indices = [0, 1, 2, 3]
    zeus_monitor_mock_instance.end_window.return_value = Measurement(
        time=37.24807469360,
        gpu_energy={
            0: 4264.87199999392,
            1: 4367.186999991536,
            2: 4342.869000002742,
            3: 4158.034000009298,
        },
    )

    mocker.patch(
        "zeus.monitor.energy.ZeusMonitor", return_value=zeus_monitor_mock_instance
    )
    mocker.patch("pynvml.nvmlDeviceGetName").return_value = "Tesla V100"
    mocker.patch("pynvml.nvmlDeviceGetCount").return_value = 4
    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex").return_value = 0
    mocker.patch("pynvml.nvmlDeviceGetPowerManagementLimitConstraints").return_value = [
        100000,
        300000,
    ]
    return zeus_monitor_mock_instance


@pytest.fixture(autouse=True)
def mock_http_call(client, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)
    mocker.patch("httpx.patch", side_effect=client.patch)

    mocker.patch("atexit.register")


def test_register_job(mock_monitor, helpers):
    job = JobSpec.parse_obj(helpers.get_fake_job("test_register_job"))
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.job.max_power == 300 * len(mock_monitor.gpu_indices)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.job.max_power == 300 * len(mock_monitor.gpu_indices)


def test_batch_sizes(mock_monitor, helpers):
    job = JobSpec.parse_obj(helpers.get_fake_job("test_batch_sizes"))
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bs = bso_client.get_batch_size()

    assert bs == 1024 and bso_client.current_batch_size == 1024

    bso_client.on_train_begin()
    bso_client.on_evaluate(0.1)
    bso_client.on_evaluate(0.2)
    bso_client.on_evaluate(0.6)  # Converged

    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bs = bso_client.get_batch_size()
    bso_client.on_train_begin()

    assert bs == 512 and bso_client.current_batch_size == 512

    i = 0
    with pytest.raises(Exception) as e_info:
        while i < job.max_epochs - 10:  # Test Early stop
            bso_client.on_evaluate(0.3)
            i += 1
            assert i == bso_client.cur_epoch
            assert bso_client.current_batch_size == 512

        assert str(e_info.value).find("cost upper bound") != -1

    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bs = bso_client.get_batch_size()
    bso_client.on_train_begin()

    assert bs == 2048 and bso_client.current_batch_size == 2048


def test_converge_fail(mock_monitor, helpers):
    job = JobSpec.parse_obj(helpers.get_fake_job("test_converge_fail"))
    job.beta_knob = None  # disable early stop
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bso_client.on_train_begin()
    bs = bso_client.get_batch_size()

    assert bs == 1024 and bso_client.current_batch_size == 1024

    i = 0
    with pytest.raises(Exception) as e_info:
        while i < job.max_epochs + 10:  # Fail after max_epoch
            bso_client.on_evaluate(0.3)
            i += 1
            assert i == bso_client.cur_epoch
            assert bso_client.current_batch_size == 1024

        print(e_info.value, i)
        assert str(e_info.value).find("Train failed to converge within max_epoch") != -1

    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bs = bso_client.get_batch_size()
    bso_client.on_train_begin()

    assert bs == 2048 and bso_client.current_batch_size == 2048


def test_distributed_setting(mock_monitor, helpers):
    job = JobSpec.parse_obj(helpers.get_fake_job("test_distributed_setting"))
    NGPU = 4
    bso_clients = [
        BatchSizeOptimizer(mock_monitor, "", job, rank=i) for i in range(NGPU)
    ]

    # Only rank=0 can ask for bs
    for i in range(1, NGPU):
        with pytest.raises(ZeusBSOBadOperationError) as e_info:
            bso_clients[i].get_batch_size()

    bs = bso_clients[0].get_batch_size()
    # distribute batch size to other clients
    for i in range(1, NGPU):
        bso_clients[i].current_batch_size = bs
        bso_clients[i].trial_number = bso_clients[0].trial_number

    # Mark as unconverged from rank = 0 client
    i = 0
    with pytest.raises(ZeusBSOTrainFailError) as e_info:
        while i < job.max_epochs + 10:  # Fail after max_epoch
            bso_clients[0].on_evaluate(0.3)
            i += 1
            assert i == bso_clients[0].cur_epoch
            assert bso_clients[0].current_batch_size == 1024

        print("[ERROR]", e_info.value, i)
        assert str(e_info.value).find("Train failed to converge within max_epoch") != -1

    for i in range(1, NGPU):
        with pytest.raises(ZeusBSOTrainFailError) as e_info:
            bso_clients[i].on_evaluate(0.3)
            assert bso_clients[i].current_batch_size == 1024
        assert str(e_info.value).find("is already reported.") != -1
