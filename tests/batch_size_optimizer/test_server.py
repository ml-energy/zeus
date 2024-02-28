import asyncio
from copy import deepcopy
import random
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)

from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.server.database.models import Base
from zeus.optimizer.batch_size.server.router import app


# https://fastapi.tiangolo.com/tutorial/testing/

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
    "max_power": 3000,
    "number_of_gpus": 4,
    "gpu_model": "A100",
}

sessionmanager = DatabaseSessionManager("sqlite+aiosqlite:///test.db", {"echo": False})


async def override_db_session() -> AsyncIterator[AsyncSession]:
    async with sessionmanager.session() as session:
        yield session


app.dependency_overrides[get_db_session] = override_db_session


async def create():
    print("Create tables")
    async with sessionmanager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def clean():
    print("Clean")
    async with sessionmanager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="session", autouse=True)
def database_setup():
    asyncio.run(create())
    yield
    asyncio.run(clean())


@pytest.fixture
def client():
    with TestClient(app=app) as c:
        yield c


@pytest.mark.anyio
def test_register_job(client):
    response = client.post("/jobs", json=fake_job)
    print(response.text)
    print(str(response))
    assert response.status_code == 201

    response = client.post("/jobs", json=fake_job)
    print(response.text)
    assert response.status_code == 200


@pytest.mark.anyio
def test_register_job_with_diff_config(client):
    fake_job_diff = deepcopy(fake_job)
    fake_job_diff["default_batch_size"] = 512

    response = client.post("/jobs", json=fake_job_diff)
    print(response.text)
    assert response.status_code == 409


@pytest.mark.anyio
def test_register_job_validation_error(client):
    temp = deepcopy(fake_job)
    temp["default_batch_size"] = 128
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp["default_batch_size"] = 0
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp = deepcopy(fake_job)
    temp["max_epochs"] = 0
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp = deepcopy(fake_job)
    temp["batch_sizes"] = []
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp = deepcopy(fake_job)
    temp["eta_knob"] = 1.1
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp = deepcopy(fake_job)
    temp["beta_knob"] = 0
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422


@pytest.mark.anyio
def test_predict(client):
    cur_default_bs = fake_job["default_batch_size"]
    response = client.get(
        "/jobs/batch_size",
        params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
    )
    print(response.text)
    assert response.status_code == 200
    assert response.json() == cur_default_bs

    # concurrent job submission
    response = client.get(
        "/jobs/batch_size",
        params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
    )
    print(response.text)
    assert response.status_code == 200
    assert response.json() == cur_default_bs


@pytest.mark.anyio
def test_report(client):
    # Converged within max epoch => successful training
    response = client.post(
        "/jobs/report",
        json={
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "batch_size": 1024,
            "time": 14.438,
            "energy": 3000.123,
            "max_power": 300,
            "metric": 0.55,
            "current_epoch": 98,
        },
    )
    assert (
        response.status_code == 200
        and response.json()["converged"] == True
        and response.json()["stop_train"] == True
    )
    # NO update in exploration state since this was a concurrent job submission
    response = client.post(
        "/jobs/report",
        json={
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "batch_size": 1024,
            "time": 14.438,
            "energy": 3000.123,
            "max_power": 300,
            "metric": 0.55,
            "current_epoch": 98,
        },
    )
    assert (
        response.status_code == 200
        and response.json()["converged"] == True
        and response.json()["stop_train"] == True
    )


@pytest.mark.anyio
def test_predict_report_sequence(client):
    cur_default_bs = fake_job["default_batch_size"]

    # Previous default batch size is converged
    for trial in range(1, fake_job["num_pruning_rounds"] + 1):
        idx = fake_job["batch_sizes"].index(cur_default_bs)
        down = sorted(fake_job["batch_sizes"][: idx + 1], reverse=True)
        up = sorted(fake_job["batch_sizes"][idx + 1 :])

        print("Exploration space:", [down, up])
        for bs_list in [down, up]:
            for bs in bs_list:
                if (
                    trial == 1 and bs == cur_default_bs
                ):  # already reported converged before
                    continue

                # Predict
                response = client.get(
                    "/jobs/batch_size",
                    params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
                )
                assert response.status_code == 200
                assert response.json() == bs

                # Concurrent job
                response = client.get(
                    "/jobs/batch_size",
                    params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
                )
                assert response.status_code == 200
                assert response.json() == (
                    cur_default_bs if trial == 1 and bs == 512 else 512
                )

                time = 14.438
                converged = random.choice([True, True, False])
                if (
                    bs == 512
                ):  # make 512 as the best bs so that we can change the default bs to 512 next round
                    converged = True
                    time = 12

                response = client.post(
                    "/jobs/report",
                    json={
                        "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "batch_size": bs,
                        "time": time,
                        "energy": 3000.123,
                        "max_power": 300,
                        "metric": 0.55 if converged else 0.33,
                        "current_epoch": 98 if converged else 100,
                    },
                )
                assert (
                    response.status_code == 200
                    and response.json()["converged"] == converged
                    and response.json()["stop_train"] == True
                )
                if not converged:
                    break

        cur_default_bs = 512
