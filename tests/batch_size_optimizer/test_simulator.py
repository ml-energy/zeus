import asyncio
import logging
import re
from typing import AsyncIterator
import pytest
from examples.trace_driven.run_single import read_trace
from fastapi.testclient import TestClient
from typing import Literal
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio.session import AsyncSession
from tests.batch_size_optimizer.simulate_with_server import SimulatorWithServer
from zeus.job import Job
from zeus.optimizer.batch_size.server.database.db_connection import (
    DatabaseSessionManager,
    get_db_session,
)
from zeus.optimizer.batch_size.server.database.schema import Base
from zeus.optimizer.batch_size.server.router import app
from zeus.policy.optimizer import JITPowerLimitOptimizer, PruningGTSBatchSizeOptimizer
from zeus.simulate import Simulator


config = {
    "gpu": "v100",
    "eta_knob": 0.5,
    "beta_knob": 2.0,
    "seed": 1,
    "dataset": "librispeech",
    "model": "deepspeech2",
    "optimizer": "adamw",
    "target_metric": 40.0,
    "max_epochs": 16,
    "b_0": 192,  # default_bs
    "num_recurrence": None,
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
    with TestClient(app) as c:
        yield c


def arm_state_parser(output):
    # Define regex patterns to match the numbers
    arm_pattern = r"Arm\s+(\d+)"
    mu_pattern = r"N\(([-+]?\d*\.\d+|\d+),\s+([-+]?\d*\.\d+|\d+)\)"
    arrow_pattern = r"-> ([-+]?\d*\.\d+|\d+)"

    # Use regex to find the numbers in the output
    arm_numbers = re.findall(arm_pattern, output, re.MULTILINE)
    mu_numbers = re.findall(mu_pattern, output, re.MULTILINE)
    arrow_numbers = re.findall(arrow_pattern, output, re.MULTILINE)

    d = []

    for arm, mu, arrow in zip(arm_numbers, mu_numbers, arrow_numbers):
        d.append({"Arm": arm, "Mean": mu[0], "stdev": mu[1], "Arrow": arrow})

    return d


def test_end_to_end(client, caplog, capsys, mocker: MockerFixture):
    mocker.patch("httpx.post", side_effect=client.post)
    mocker.patch("httpx.get", side_effect=client.get)

    ## CONFIG
    gpu: Literal["a40", "v100", "p100", "rtx6000"] = config["gpu"]
    eta_knob: float = config["eta_knob"]
    beta_knob: float = config["beta_knob"]
    num_recurrence: int | None = config["num_recurrence"]
    seed: int = config["seed"]

    job = Job(
        config["dataset"],
        config["model"],
        config["optimizer"],
        config["target_metric"],
        config["max_epochs"],
        config["b_0"],
    )

    train_df, power_df = read_trace(gpu)

    # # Use 2 * |B| * |P| is num_recurrence is None.
    # print(num_recurrence)
    if num_recurrence is None:
        job_df = job.filter_df(train_df.merge(power_df, how="inner"))
        num_recurrence = (
            2 * len(job_df.batch_size.unique()) * len(job_df.power_limit.unique())
        )

    ### New simulator
    # Instantiate optimizers.
    plo = JITPowerLimitOptimizer(verbose=False)

    # Instantitate the simulator.
    simulator = SimulatorWithServer(
        train_df, power_df, plo, gpu, seed=seed, verbose=False
    )
    # # Run the simulator.
    result = simulator.simulate_one_job(job, num_recurrence, beta_knob, eta_knob)
    selected_bs = [item.bs for item in result]

    ### Original Simulator
    # Instantiate optimizers.
    org_plo = JITPowerLimitOptimizer(verbose=False)
    org_bso = PruningGTSBatchSizeOptimizer(seed=seed, verbose=True)

    # Instantitate the simulator.
    original_simulator = Simulator(
        train_df, power_df, org_bso, org_plo, seed=seed, verbose=False
    )
    original_result = original_simulator.simulate_one_job(
        job, num_recurrence, beta_knob, eta_knob
    )
    org_selected_bs = [item.bs for item in original_result]

    out, err = capsys.readouterr()
    records = arm_state_parser(out)

    new_sim_records = arm_state_parser(caplog.text)

    # Compare arm states
    assert records == new_sim_records
    # Compare selected batch sizes
    assert selected_bs == org_selected_bs
