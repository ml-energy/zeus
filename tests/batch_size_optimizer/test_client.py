from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from zeus.monitor.energy import Measurement, ZeusMonitor
from zeus.optimizer.batch_size.client import BatchSizeOptimizer
from zeus.optimizer.batch_size.common import JobSpec
from zeus.optimizer.batch_size.server.router import app

fake_job = {
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "seed": 1,
    "default_batch_size": 1024,
    "batch_sizes": [32, 64, 256, 512, 1024, 4096, 2048],
    "eta_knob": 0.5,
    "beta_knob": 2,
    "target_metric": 0.5,
    "high_is_better_metric": True,
    "max_epochs": 100,
    "num_pruning_rounds": 2,
    "mab_setting": {
        "prior_mean": 0,
        "prior_precision": 0,
        "window_size": 0,
        "seed": 123456,
        "num_exploration": 2,
    },
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


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
    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex").return_value = 0
    mocker.patch("pynvml.nvmlDeviceGetPowerManagementLimitConstraints").return_value = (
        300
    )
    return zeus_monitor_mock_instance


@pytest.mark.anyio
def test_register_job(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    job = JobSpec.parse_obj(fake_job)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.max_power == 300 * len(mock_monitor.gpu_indices)
    bso_client = BatchSizeOptimizer(mock_monitor, "", job)
    assert bso_client.max_power == 300 * len(mock_monitor.gpu_indices)


@pytest.mark.anyio
def test_batch_sizes(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)
    job = JobSpec.parse_obj(fake_job)
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


@pytest.mark.anyio
def test_converge_fail(client, mock_monitor, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)
    job = JobSpec.parse_obj(fake_job)
    job.job_id = uuid4()
    job.beta_knob = -1
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
