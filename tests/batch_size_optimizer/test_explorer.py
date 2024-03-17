import asyncio
import logging
import re
import uuid
from typing import AsyncIterator

import pytest
from zeus.optimizer.batch_size.common import TrainingResult
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)
from zeus.optimizer.batch_size.server.database.schema import Base
from zeus.optimizer.batch_size.server.router import app

fake_job = {
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "seed": 1,
    "eta_knob": 0.5,
    "beta_knob": -1,
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


@pytest.fixture(scope="session", autouse=True)
def database_setup():
    logger = logging.getLogger(
        "zeus.optimizer.batch_size.server.mab"
    )  # for testing, propagate the log to the root logger so that caplog can capture
    logger.propagate = True

    asyncio.run(clean())
    asyncio.run(create())
    yield


@pytest.fixture
def client():
    with TestClient(app=app) as c:
        yield c


@pytest.mark.usefixtures("client")
@pytest.mark.usefixtures("caplog")
class TestPruningExploreManager:
    """Unit test class for pruning exploration."""

    batch_sizes: list[int] = [8, 16, 32, 64, 128, 256]

    def exploration_to_training_result(
        self, exploration: tuple[int, float, bool]
    ) -> TrainingResult:
        return TrainingResult(
            job_id=fake_job["job_id"],
            batch_size=exploration[0],
            time=2 * (exploration[1] - 1),
            energy=2,
            max_power=1,
            metric=0.55 if exploration[2] else 0.4,
            current_epoch=100,
        )

    # 0.5 * energy + (1 - eta_knob) * max_power * time
    def register_job_with_default_bs(self, client, default_bs: int) -> str:
        job_id = str(uuid.uuid4())
        fake_job["job_id"] = job_id
        fake_job["batch_sizes"] = self.batch_sizes
        fake_job["default_batch_size"] = default_bs

        response = client.post("/jobs", json=fake_job)
        assert response.status_code == 201

        return job_id

    def run_exploration(
        self,
        client,
        caplog,
        job_id: str,
        exploration: list[tuple[int, float, bool]],
        result: list[int],
    ) -> None:
        """Drive the pruning explore manager and check results."""
        caplog.set_level(logging.INFO)
        for exp in exploration:
            res = self.exploration_to_training_result(exp)
            response = client.get(
                "/jobs/batch_size",
                params={"job_id": job_id},
            )
            assert response.status_code == 200
            assert (
                response.json() == exp[0]
            ), f"Expected {exp[0]} but got {response.json()} ({exp})"

            response = client.post(
                "/jobs/report",
                content=res.json(),
            )
            assert response.status_code == 200
            assert response.json()["converged"] == exp[2]
            print(response.json()["message"])
        # Now good_bs should be equal to result!

        # this will construct mab
        response = client.get(
            "/jobs/batch_size",
            params={"job_id": job_id},
        )
        assert response.status_code == 200

        # Capture list of arms from stdout
        matches = re.search(r"with arms \[(.*?)\]", caplog.text)

        if matches:
            arms = [int(x) for x in matches.group(1).split(",")]
            assert arms == result
        else:
            assert False, "No output found from constructing Mab"

    def test_normal(self, client, caplog):
        """Test a typical case."""
        job_id = self.register_job_with_default_bs(client, 128)

        exploration = [
            (128, 10.0, True),
            (64, 9.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 21.0, False),
            (256, 15.0, True),
            (32, 8.0, True),
            (16, 12.0, False),
            (64, 9.0, True),
            (128, 10.0, True),
            (256, 17.0, False),
        ]

        result = [32, 64, 128]
        self.run_exploration(client, caplog, job_id, exploration, result)

    def test_default_is_largest(self, client, caplog):
        """Test the case when the default batch size is the largest one."""
        job_id = self.register_job_with_default_bs(client, 256)

        exploration = [
            (256, 7.0, True),
            (128, 8.0, True),
            (64, 9.0, True),
            (32, 13.0, True),
            (16, 22.0, False),
            (256, 8.0, True),
            (128, 8.5, True),
            (64, 9.0, True),
            (32, 12.0, True),
        ]
        result = [32, 64, 128, 256]
        self.run_exploration(client, caplog, job_id, exploration, result)

    def test_default_is_smallest(self, client, caplog):
        """Test the case when the default batch size is the smallest one."""
        job_id = self.register_job_with_default_bs(client, 8)

        exploration = [
            (8, 10.0, True),
            (16, 17.0, True),
            (32, 20.0, True),
            (64, 25.0, False),
            (8, 10.0, True),
            (16, 21.0, False),
        ]
        result = [8]
        self.run_exploration(client, caplog, job_id, exploration, result)

    def test_all_converge(self, client, caplog):
        """Test the case when every batch size converges."""
        job_id = self.register_job_with_default_bs(client, 64)
        exploration = [
            (64, 10.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 15.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
            (32, 7.0, True),
            (16, 10.0, True),
            (8, 15.0, True),
            (64, 10.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
        ]
        result = self.batch_sizes
        self.run_exploration(client, caplog, job_id, exploration, result)

    def test_every_bs_is_bs(self, client, caplog):
        """Test the case when every batch size other than the default fail to converge."""
        job_id = self.register_job_with_default_bs(client, 64)
        exploration = [
            (64, 10.0, True),
            (32, 22.0, False),
            (128, 25.0, False),
            (64, 9.0, True),
        ]
        result = [64]
        self.run_exploration(client, caplog, job_id, exploration, result)
