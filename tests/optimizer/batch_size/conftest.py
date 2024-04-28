from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest

# HACK: ZeusBsoSettings will complain if `database_url` is not set
#       via environment variable.
os.environ["ZEUS_BSO_DATABASE_URL"] = "sqlite+aiosqlite:///dummy.db"

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)
from zeus.optimizer.batch_size.server.database.schema import Base
from zeus.optimizer.batch_size.server.router import app


class Helpers:
    @staticmethod
    def get_fake_job(job_id: str) -> dict:
        return {
            "job_id": job_id,
            "job_id_prefix": "test",
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

    @classmethod
    def get_fake_job_config(cls, job_id: str) -> dict:
        fake_job_config = cls.get_fake_job(job_id)
        fake_job_config["max_power"] = 3000
        fake_job_config["number_of_gpus"] = 4
        fake_job_config["gpu_model"] = "A100"
        return fake_job_config


@pytest.fixture(scope="module")
def helpers():
    return Helpers


def init(db_url: str):
    sessionmanager = DatabaseSessionManager(
        f"sqlite+aiosqlite:///{db_url}", {"echo": False}
    )

    async def override_db_session() -> AsyncIterator[AsyncSession]:
        async with sessionmanager.session() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_db_session
    return sessionmanager


async def create(sessionmanager: DatabaseSessionManager):
    print("Create tables")
    async with sessionmanager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def clean(sessionmanager: DatabaseSessionManager):
    print("Clean")
    async with sessionmanager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# For each worker, set up db
@pytest.fixture(scope="module", autouse=True)
def session_data(tmp_path_factory, worker_id):
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    sm = init(str(root_tmp_dir / f"test-{worker_id}.db"))
    asyncio.run(clean(sm))
    asyncio.run(create(sm))
    return
