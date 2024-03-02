import asyncio
from copy import deepcopy
import random
import sys
from typing import AsyncIterator
import uuid
from build.lib.zeus.optimizer.batch_size.common import TrainingResult

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


@pytest.mark.usefixtures("client")
@pytest.mark.usefixtures("capsys")
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

    def run_exploration(
        self,
        client,
        capsys,
        job_id: str,
        exploration: list[tuple[int, float, bool]],
        result: list[int],
    ) -> None:
        """Drive the pruning explore manager and check results."""
        for exp in exploration:
            res = self.exploration_to_training_result(exp)
            response = client.get(
                "/jobs/batch_size",
                params={"job_id": job_id},
            )
            assert response.status_code == 200
            assert response.json() == exp[0]

            response = client.post(
                "/jobs/report",
                content=res.json(),
            )
            assert response.status_code == 200
            assert response.json()["converged"] == exp[2]
        # Now good_bs should be equal to result!

        out, err = capsys.readouterr()
        sys.stdout.write(out)
        sys.stderr.write(err)
        print("OUT", out)
        print("SYS", err)

    @pytest.mark.anyio
    def test_normal(self, client, capsys):
        job_id = str(uuid.uuid4())
        fake_job["job_id"] = job_id
        fake_job["batch_sizes"] = self.batch_sizes
        fake_job["default_batch_size"] = 128

        response = client.post("/jobs", json=fake_job)
        assert response.status_code == 201

        """Test a typical case."""
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
        self.run_exploration(client, capsys, job_id, exploration, result)

    # def test_default_is_largest(self):
    #     """Test the case when the default batch size is the largest one."""
    #     manager = PruningExploreManager(self.batch_sizes, 256)
    #     exploration = [
    #         (256, 7.0, True),
    #         (128, 8.0, True),
    #         (64, 9.0, True),
    #         (32, 13.0, True),
    #         (16, 22.0, False),
    #         (256, 8.0, True),
    #         (128, 8.5, True),
    #         (64, 9.0, True),
    #         (32, 12.0, True),
    #     ]
    #     result = [32, 64, 128, 256]
    #     self.run_exploration(manager, exploration, result)

    # def test_default_is_smallest(self):
    #     """Test the case when the default batch size is the smallest one."""
    #     manager = PruningExploreManager(self.batch_sizes, 8)
    #     exploration = [
    #         (8, 10.0, True),
    #         (16, 17.0, True),
    #         (32, 20.0, True),
    #         (64, 25.0, False),
    #         (8, 10.0, True),
    #         (16, 21.0, False),
    #     ]
    #     result = [8]
    #     self.run_exploration(manager, exploration, result)

    # def test_all_converge(self):
    #     """Test the case when every batch size converges."""
    #     manager = PruningExploreManager(self.batch_sizes, 64)
    #     exploration = [
    #         (64, 10.0, True),
    #         (32, 8.0, True),
    #         (16, 12.0, True),
    #         (8, 15.0, True),
    #         (128, 12.0, True),
    #         (256, 13.0, True),
    #         (32, 7.0, True),
    #         (16, 10.0, True),
    #         (8, 15.0, True),
    #         (64, 10.0, True),
    #         (128, 12.0, True),
    #         (256, 13.0, True),
    #     ]
    #     result = self.batch_sizes
    #     self.run_exploration(manager, exploration, result)

    # def test_every_bs_is_bs(self):
    #     """Test the case when every batch size other than the default fail to converge."""
    #     manager = PruningExploreManager(self.batch_sizes, 64)
    #     exploration = [
    #         (64, 10.0, True),
    #         (32, 22.0, False),
    #         (128, 25.0, False),
    #         (64, 9.0, True),
    #     ]
    #     result = [64]
    #     self.run_exploration(manager, exploration, result)
