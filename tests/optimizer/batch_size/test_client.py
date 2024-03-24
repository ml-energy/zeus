from copy import deepcopy
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from zeus.monitor.energy import Measurement, ZeusMonitor
from zeus.optimizer.batch_size.client import BatchSizeOptimizer
from zeus.optimizer.batch_size.common import JobSpec


@pytest.fixture
def mock_monitor(mocker: MockerFixture):
    mocker.patch("pynvml.nvmlInit")

    zeus_monitor_mock_instance = MagicMock(spec=ZeusMonitor)
    zeus_monitor_mock_instance.nvml_gpu_indices = [0, 1, 2, 3]
    zeus_monitor_mock_instance.gpu_indices = [0, 1, 2, 3]
    zeus_monitor_mock_instance.end_window.return_value = Measurement(
        time=37.24807469360,
        energy={
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
    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex").return_value = 0
    mocker.patch("pynvml.nvmlDeviceGetPowerManagementLimitConstraints").return_value = [
        100000,
        300000,
    ]
    return zeus_monitor_mock_instance


def test_register_job(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    job = JobSpec.parse_obj(pytest.fake_job)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.job.max_power == 300 * len(mock_monitor.gpu_indices)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.job.max_power == 300 * len(mock_monitor.gpu_indices)

    no_job_id = deepcopy(pytest.fake_job)
    no_job_id["job_id"] = None
    bso_client = BatchSizeOptimizer(mock_monitor, "", JobSpec.parse_obj(no_job_id))
    assert bso_client.job.max_power == 300 * len(mock_monitor.gpu_indices)
    assert bso_client.job.job_id.startswith(pytest.fake_job["job_id_prefix"])


def test_batch_sizes(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)
    job = JobSpec.parse_obj(pytest.fake_job)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    bs = bso_client.get_batch_size()

    assert bs == 1024 and bso_client.current_batch_size == 1024

    bso_client.on_train_begin()
    bso_client.on_evaluate(0.1)
    bso_client.on_evaluate(0.2)
    bso_client.on_evaluate(0.6)  # Converged

    bso_client.on_train_begin()
    bs = bso_client.get_batch_size()

    assert bs == 512 and bso_client.current_batch_size == 512

    i = 0
    with pytest.raises(Exception) as e_info:
        while i < job.max_epochs - 10:  # Test Early stop
            bso_client.on_evaluate(0.3)
            i += 1
            assert i == bso_client.cur_epoch
            assert bso_client.current_batch_size == 512

    assert str(e_info.value).find("cost upper bound") != -1

    bso_client.on_train_begin()
    bs = bso_client.get_batch_size()

    assert bs == 2048 and bso_client.current_batch_size == 2048


def test_converge_fail(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)
    job = JobSpec.parse_obj(pytest.fake_job)
    job.job_id = "test-something"
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

    bso_client.on_train_begin()
    bs = bso_client.get_batch_size()

    assert bs == 2048 and bso_client.current_batch_size == 2048
