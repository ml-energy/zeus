<<<<<<< HEAD
import asyncio
from copy import deepcopy
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import MetaData
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)

from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.server.database.models import Base
from zeus.optimizer.batch_size.server.router import app


=======
from copy import deepcopy
from fastapi.testclient import TestClient
import pytest

from zeus.optimizer.batch_size.server.router import app

>>>>>>> 9a91219 (checkpoint - testing client)
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
}

<<<<<<< HEAD
sessionmanager = DatabaseSessionManager("sqlite+aiosqlite:///test.db", {"echo": True})


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

=======
>>>>>>> 9a91219 (checkpoint - testing client)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.anyio
def test_register_job(client):
    response = client.post("/jobs", json=fake_job)
<<<<<<< HEAD
    print(response.text)
    assert response.status_code == 201

    response = client.post("/jobs", json=fake_job)
    print(response.text)
=======
    assert response.status_code == 201

    response = client.post("/jobs", json=fake_job)
>>>>>>> 9a91219 (checkpoint - testing client)
    assert response.status_code == 200


@pytest.mark.anyio
<<<<<<< HEAD
def test_register_job_with_diff_config(client):
    fake_job_diff = deepcopy(fake_job)
    fake_job_diff["default_batch_size"] = 512

    response = client.post("/jobs", json=fake_job_diff)
    print(response.text)
    assert response.status_code == 409


@pytest.mark.anyio
=======
>>>>>>> 9a91219 (checkpoint - testing client)
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

<<<<<<< HEAD
    temp = deepcopy(fake_job)
    temp["eta_knob"] = 1.1
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422

    temp = deepcopy(fake_job)
    temp["beta_knob"] = 0
    response = client.post("/jobs", json=temp)
    assert response.status_code == 422


# @pytest.mark.anyio
# def test_predict(client):
#     # @app.get("/jobs/batch_size")
#     response = client.post("/jobs", json=fake_job)
#     assert response.status_code == 201

#     response = client.get(
#         "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
#     )
#     assert response.status_code == 200
#     assert response.json() == 1024

#     print(response.status_code)


# @pytest.mark.anyio
# def test_report(client):
#     # @app.post("/jobs/report")
#     # job_id: UUID
#     # batch_size: int
#     # cost: float
#     # converged: bool | None = None  # for pruning stage
#     response = client.post("/jobs", json=fake_job)
#     assert response.status_code == 201

#     response = client.get(
#         "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
#     )
#     assert response.status_code == 200
#     assert response.json() == 1024

#     # Converged within max epoch => successful training
#     response = client.post(
#         "/jobs/report",
#         json={
#             "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#             "batch_size": 1024,
#             "time": "14.438",
#             "energy": 3000.123,
#             "max_power": 300,
#             "metric": 0.55,
#             "current_epoch": 98,
#         },
#     )
#     assert (
#         response.status_code == 200
#         and response.json()["converged"] == True
#         and response.json()["stop_train"] == True
#     )

#     # Should get 512 since the cost converged
#     response = client.get(
#         "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
#     )

#     assert response.status_code == 200
#     assert response.json() == 512

#     # Converge fail before after max_epoch reached => Should keep training
#     response = client.post(
#         "/jobs/report",
#         json={
#             "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#             "batch_size": 512,
#             "time": "16.438",
#             "energy": 2787.123,
#             "max_power": 300,
#             "metric": 0.3,
#             "current_epoch": 56,
#         },
#     )

#     assert (
#         response.status_code == 200
#         and response.json()["converged"] == False
#         and response.json()["stop_train"] == False
#     )

#     # Converge fail after max_epoch reached => Should stop training with err
#     response = client.post(
#         "/jobs/report",
#         json={
#             "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#             "batch_size": 512,
#             "time": "16.438",
#             "energy": 2787.123,
#             "max_power": 300,
#             "metric": 0.3,
#             "current_epoch": 100,
#         },
#     )
#     assert (
#         response.status_code == 200
#         and response.json()["converged"] == False
#         and response.json()["stop_train"] == True
#     )

#     response = client.get(
#         "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
#     )

#     assert response.status_code == 200
#     assert response.json() == 2048
=======

@pytest.mark.anyio
def test_predict(client):
    # @app.get("/jobs/batch_size")
    response = client.post("/jobs", json=fake_job)
    assert response.status_code == 201

    response = client.get(
        "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
    )
    assert response.status_code == 200
    assert response.json() == 1024

    print(response.status_code)


@pytest.mark.anyio
def test_report(client):
    # @app.post("/jobs/report")
    # job_id: UUID
    # batch_size: int
    # cost: float
    # converged: bool | None = None  # for pruning stage
    response = client.post("/jobs", json=fake_job)
    assert response.status_code == 201

    response = client.get(
        "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
    )
    assert response.status_code == 200
    assert response.json() == 1024

    # Converged within max epoch => successful training
    response = client.post(
        "/jobs/report",
        json={
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "batch_size": 1024,
            "time": "14.438",
            "energy": 3000.123,
            "max_power": 300,
            "converged": True,
            "current_epoch": 98,
        },
    )
    assert response.status_code == 200

    # Should get 512 since the cost converged
    response = client.get(
        "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
    )

    assert response.status_code == 200
    assert response.json() == 512

    # Converge fail before after max_epoch reached => Should keep training
    response = client.post(
        "/jobs/report",
        json={
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "batch_size": 512,
            "time": "16.438",
            "energy": 2787.123,
            "max_power": 300,
            "converged": False,
            "current_epoch": 56,
        },
    )
    assert response.status_code == 200

    # Converge fail after max_epoch reached => Should stop training with err
    response = client.post(
        "/jobs/report",
        json={
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "batch_size": 512,
            "time": "16.438",
            "energy": 2787.123,
            "max_power": 300,
            "converged": False,
            "current_epoch": 100,
        },
    )
    assert response.status_code == 500

    response = client.get(
        "/jobs/batch_size", params={"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
    )

    assert response.status_code == 200
    assert response.json() == 2048
>>>>>>> 9a91219 (checkpoint - testing client)
