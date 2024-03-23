import asyncio
from typing import AsyncIterator
from fastapi.testclient import TestClient
import pytest

from sqlalchemy.ext.asyncio.session import AsyncSession

from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)
from zeus.optimizer.batch_size.server.database.schema import Base
from zeus.optimizer.batch_size.server.router import app


def pytest_configure():
    # Test wide global variable.
    # https://docs.pytest.org/en/latest/deprecations.html#pytest-namespace
    pytest.fake_job = {
        "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "seed": 1,
        "default_batch_size": 1024,
        "batch_sizes": [32, 64, 256, 512, 1024, 4096, 2048],
        "eta_knob": 0.5,
        "beta_knob": 2,
        "target_metric": 0.5,
        "higher_is_better_metric": True,
        "max_epochs": 100,
        "num_pruning_rounds": 2,
        "window_size": 5,
        "mab_prior_mean": 0,
        "mab_prior_precision": 0,
        "mab_seed": 123456,
        "mab_num_explorations": 2,
    }

    pytest.fake_job_config = {
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
        "window_size": 0,
        "mab_prior_mean": 0,
        "mab_prior_precision": 0,
        "mab_seed": 123456,
        "mab_num_exploration": 2,
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


@pytest.fixture(scope="module", autouse=True)
def database_setup():
    asyncio.run(clean())
    asyncio.run(create())
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
