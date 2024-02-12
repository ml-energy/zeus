from copy import deepcopy
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest
import httpx
from pytest_mock import MockerFixture
from tests.test_monitor import mock_gpus, pynvml_mock
from zeus.monitor.energy import ZeusMonitor
from zeus.optimizer.batch_size.client import BatchSizeOptimizerClient
from zeus.optimizer.batch_size.server.models import JobSpec

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


#  pynvml.nvmlInit()
#         pls = []
#         self.max_power = 0
#         for index in self.monitor.nvml_gpu_indices:
#             device = pynvml.nvmlDeviceGetHandleByIndex(index)
#             pls.append(pynvml.nvmlDeviceGetPowerManagementLimitConstraints(device))
#         if not all(pls[0] == pl for pl in pls):
#             raise ValueError("Power limits ranges are not uniform across GPUs.")

#         self.max_power = max(pls) * len(self.monitor.gpu_indices)


@pytest.mark.anyio
def test_register_job(client, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("pynvml.nvmlInit")

    zeus_monitor_mock_instance = MagicMock(spec=ZeusMonitor)
    zeus_monitor_mock_instance.nvml_gpu_indices = [0]
    zeus_monitor_mock_instance.gpu_indices = [0]

    mocker.patch(
        "zeus.monitor.energy.ZeusMonitor", return_value=zeus_monitor_mock_instance
    )

    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex").return_value = 0
    mocker.patch("pynvml.nvmlDeviceGetPowerManagementLimitConstraints").return_value = (
        300
    )
    job = JobSpec.parse_obj(fake_job)
    print(f"Parsed Job: {job.json()}")
    bso_client = BatchSizeOptimizerClient(zeus_monitor_mock_instance, "", job)
    assert bso_client.max_power == 300
