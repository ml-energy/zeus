import pytest
from examples.trace_driven.run_single import read_trace
from fastapi.testclient import TestClient
from pyparsing import Literal
from pytest_mock import MockerFixture
from tests.batch_size_optimizer.simulate_with_server import SimulatorWithServer
from zeus.job import Job
from zeus.optimizer.batch_size.server.router import app
from zeus.policy.optimizer import JITPowerLimitOptimizer, PruningGTSBatchSizeOptimizer
from zeus.simulate import Simulator

"""
TODO: Need update based on change in server
"""


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


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.anyio
def test_end_to_end(client, mocker: MockerFixture):
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

    # Instantiate optimizers.
    plo = JITPowerLimitOptimizer(verbose=True)

    # Instantitate the simulator.
    simulator = SimulatorWithServer(train_df, power_df, plo, seed=seed, verbose=True)

    # Use 2 * |B| * |P| is num_recurrence is None.
    print(num_recurrence)
    if num_recurrence is None:
        job_df = job.filter_df(train_df.merge(power_df, how="inner"))
        num_recurrence = (
            2 * len(job_df.batch_size.unique()) * len(job_df.power_limit.unique())
        )

    # # Run the simulator.
    result = simulator.simulate_one_job(job, num_recurrence, beta_knob, eta_knob)
    selected_bs = [item.bs for item in result]

    #### Original Simulator
    # Instantiate optimizers.
    org_plo = JITPowerLimitOptimizer(verbose=True)
    org_bso = PruningGTSBatchSizeOptimizer(seed=seed, verbose=True)

    # Instantitate the simulator.
    original_simulator = Simulator(
        train_df, power_df, org_bso, org_plo, seed=seed, verbose=True
    )
    original_result = original_simulator.simulate_one_job(
        job, num_recurrence, beta_knob, eta_knob
    )
    org_selected_bs = [item.bs for item in original_result]

    assert original_simulator.seed == simulator.seed
    assert selected_bs == org_selected_bs
