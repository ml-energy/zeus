import asyncio
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)
from zeus.optimizer.batch_size.server.database.schema import Base
from zeus.optimizer.batch_size.server.router import app

"""
Just for testing. Will be cleaned up later
"""
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
def test_play(client):
    response = client.get("/test", params={"job_id": fake_job["job_id"]})
    print("Test play", response.text)
