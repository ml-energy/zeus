from __future__ import annotations

import logging
import re
import uuid
from math import isclose

import pytest
from zeus.optimizer.batch_size.common import (
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_RESULT_URL,
    TrainingResult,
    TrialId,
)
from zeus.utils.metric import zeus_cost


@pytest.fixture(scope="session", autouse=True)
def logger_setup():
    logger = logging.getLogger(
        "zeus.optimizer.batch_size.server.mab"
    )  # for testing, propagate the log to the root logger so that caplog can capture
    logger.propagate = True
    yield


@pytest.mark.usefixtures("client")
@pytest.mark.usefixtures("caplog")
class TestPruningExploreManager:
    """Unit test class for pruning exploration."""

    batch_sizes: list[int] = [8, 16, 32, 64, 128, 256]

    def exploration_to_training_result(
        self,
        exploration: tuple[int, float, bool],
        job_id: str,
        trial_number: int,
        max_power: int,
    ) -> TrainingResult:
        energy = 1
        res = TrainingResult(
            job_id=job_id,
            batch_size=exploration[0],
            trial_number=trial_number,
            error=False,
            time=(2 * exploration[1] - energy) / max_power,
            energy=energy,
            metric=0.55 if exploration[2] else 0.4,
            current_epoch=100,
        )
        assert isclose(
            zeus_cost(res.energy, res.time, 0.5, max_power),
            exploration[1],
        )
        return res

    def register_job_with_default_bs(
        self, client, default_bs: int, helpers
    ) -> tuple[str, int]:
        job_id = f"test-{str(uuid.uuid4())}"
        fake_job = helpers.get_fake_job_config(job_id)
        fake_job["beta_knob"] = None
        fake_job["job_id"] = job_id
        fake_job["batch_sizes"] = self.batch_sizes
        fake_job["default_batch_size"] = default_bs

        response = client.post(REGISTER_JOB_URL, json=fake_job)
        assert response.status_code == 201

        return job_id, fake_job["max_power"]

    def run_exploration(
        self,
        client,
        caplog,
        job_id: str,
        exploration: list[tuple[int, float, bool]],
        result: list[int],
        max_power: int,
    ) -> None:
        """Drive the pruning explore manager and check results."""
        caplog.set_level(logging.INFO)

        for exp in exploration:
            response = client.get(
                GET_NEXT_BATCH_SIZE_URL,
                params={"job_id": job_id},
            )
            assert response.status_code == 200
            parsed_res = TrialId.parse_obj(response.json())
            assert (
                parsed_res.batch_size == exp[0]
            ), f"Expected {exp[0]} but got {parsed_res.batch_size} ({exp})"

            training_result = self.exploration_to_training_result(
                exp, job_id, parsed_res.trial_number, max_power
            )
            response = client.post(
                REPORT_RESULT_URL,
                content=training_result.json(),
            )
            assert response.status_code == 200, response.text
            assert response.json()["converged"] == exp[2]
            print(response.json()["message"])
        # Now good_bs should be equal to result!

        # this will construct mab
        response = client.get(
            GET_NEXT_BATCH_SIZE_URL,
            params={"job_id": job_id},
        )
        assert response.status_code == 200

        # Capture list of arms from stdout
        matches = re.search(r"with arms \[(.*?)\]", caplog.text)

        if matches:
            arms = [int(x) for x in matches.group(1).split(",")]
            arms.sort()
            assert arms == result
        else:
            assert False, "No output found from constructing Mab"

    def test_normal(self, client, caplog, helpers):
        """Test a typical case."""
        job_id, max_power = self.register_job_with_default_bs(client, 128, helpers)

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
        self.run_exploration(client, caplog, job_id, exploration, result, max_power)

    def test_default_is_largest(self, client, caplog, helpers):
        """Test the case when the default batch size is the largest one."""
        job_id, max_power = self.register_job_with_default_bs(client, 256, helpers)

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
        self.run_exploration(client, caplog, job_id, exploration, result, max_power)

    def test_default_is_smallest(self, client, caplog, helpers):
        """Test the case when the default batch size is the smallest one."""
        job_id, max_power = self.register_job_with_default_bs(client, 8, helpers)

        exploration = [
            (8, 10.0, True),
            (16, 17.0, True),
            (32, 20.0, True),
            (64, 25.0, False),
            (8, 10.0, True),
            (16, 21.0, False),
        ]
        result = [8]
        self.run_exploration(client, caplog, job_id, exploration, result, max_power)

    def test_all_converge(self, client, caplog, helpers):
        """Test the case when every batch size converges."""
        job_id, max_power = self.register_job_with_default_bs(client, 64, helpers)
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
        self.run_exploration(client, caplog, job_id, exploration, result, max_power)

    def test_every_bs_is_bs(self, client, caplog, helpers):
        """Test the case when every batch size other than the default fail to converge."""
        job_id, max_power = self.register_job_with_default_bs(client, 64, helpers)
        exploration = [
            (64, 10.0, True),
            (32, 22.0, False),
            (128, 25.0, False),
            (64, 9.0, True),
        ]
        result = [64]
        self.run_exploration(client, caplog, job_id, exploration, result, max_power)
