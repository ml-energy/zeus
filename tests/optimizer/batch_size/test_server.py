from __future__ import annotations

import random

import pytest
from zeus.optimizer.batch_size.common import (
    DELETE_JOB_URL,
    GET_NEXT_BATCH_SIZE_URL,
    REGISTER_JOB_URL,
    REPORT_END_URL,
    REPORT_RESULT_URL,
)

# https://fastapi.tiangolo.com/tutorial/testing/


def test_register_job(client, helpers):
    job_config = helpers.get_fake_job_config("test_register_job")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    print(response.text)
    print(str(response))
    assert response.status_code == 201

    response = client.post(REGISTER_JOB_URL, json=job_config)
    print(response.text)
    assert response.status_code == 200

    job_config["default_batch_size"] = 512
    response = client.post(REGISTER_JOB_URL, json=job_config)
    print(response.text)
    assert response.status_code == 409


def test_register_job_validation_error(client, helpers):
    temp = helpers.get_fake_job_config("test_register_job_validation_error")
    temp["default_batch_size"] = 128
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422

    temp["default_batch_size"] = 0
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422

    temp = helpers.get_fake_job_config("test_register_job_validation_error")
    temp["max_epochs"] = 0
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422

    temp = helpers.get_fake_job_config("test_register_job_validation_error")
    temp["batch_sizes"] = []
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422

    temp = helpers.get_fake_job_config("test_register_job_validation_error")
    temp["eta_knob"] = 1.1
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422

    temp = helpers.get_fake_job_config("test_register_job_validation_error")
    temp["beta_knob"] = 0
    response = client.post(REGISTER_JOB_URL, json=temp)
    assert response.status_code == 422


def test_predict(client, helpers):
    job_config = helpers.get_fake_job_config("test_predict")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    cur_default_bs = job_config["default_batch_size"]
    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    print(response.text)
    assert response.status_code == 200
    assert response.json()["batch_size"] == cur_default_bs
    assert response.json()["trial_number"] == 1

    # concurrent job submission
    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    print(response.text)
    assert response.status_code == 200
    assert response.json()["batch_size"] == cur_default_bs
    assert response.json()["trial_number"] == 2


def test_report(client, helpers):
    job_config = helpers.get_fake_job_config("test_report")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    # Converged within max epoch => successful training
    response = client.post(
        REPORT_RESULT_URL,
        json={
            "job_id": job_config["job_id"],
            "batch_size": response.json()["batch_size"],
            "trial_number": response.json()["trial_number"],
            "time": 14.438,
            "energy": 3000.123,
            "metric": 0.55,
            "current_epoch": 98,
        },
    )
    assert (
        response.status_code == 200
        and response.json()["converged"] == True
        and response.json()["stop_train"] == True
    )

    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    # Early stop
    response = client.post(
        REPORT_RESULT_URL,
        json={
            "job_id": job_config["job_id"],
            "batch_size": response.json()["batch_size"],
            "trial_number": response.json()["trial_number"],
            "time": 30,
            "energy": 6000,
            "metric": 0.55,
            "current_epoch": 98,
        },
    )
    assert (
        response.status_code == 200
        and response.json()["converged"] == False
        and response.json()["stop_train"] == True
    ), response.text


def test_exploration_stage(client, helpers):
    job_config = helpers.get_fake_job_config("test_exploration_stage")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    cur_default_bs = job_config["default_batch_size"]
    bss = job_config["batch_sizes"]
    trial_number = 1
    for trial in range(1, job_config["num_pruning_rounds"] + 1):
        idx = bss.index(cur_default_bs)
        down = sorted(bss[: idx + 1], reverse=True)
        up = sorted(bss[idx + 1 :])
        new_bss = []

        print("Exploration space:", [down, up])
        for bs_list in [down, up]:
            for bs in bs_list:
                # Predict
                response = client.get(
                    GET_NEXT_BATCH_SIZE_URL,
                    params={"job_id": job_config["job_id"]},
                )
                assert response.status_code == 200
                assert response.json()["batch_size"] == bs
                assert response.json()["trial_number"] == trial_number
                trial_number += 1

                # Concurrent job
                response = client.get(
                    GET_NEXT_BATCH_SIZE_URL,
                    params={"job_id": job_config["job_id"]},
                )
                assert response.status_code == 200
                assert (
                    response.json()["batch_size"] == cur_default_bs
                    if trial == 1 and bs == 512
                    else 512
                )
                assert response.json()["trial_number"] == trial_number
                trial_number += 1

                time = 14.438
                converged = random.choice([True, True, False])
                if bs == 1024:  # make default bs converged for 512
                    converged = True
                if (
                    bs == 512
                ):  # make 512 as the best bs so that we can change the default bs to 512 next round
                    converged = True
                    time = 12
                if converged:
                    new_bss.append(bs)

                response = client.post(
                    REPORT_RESULT_URL,
                    json={
                        "job_id": job_config["job_id"],
                        "batch_size": bs,
                        "error": False,
                        # Report to first trial in the loop
                        "trial_number": trial_number - 2,
                        "time": time,
                        "energy": 3000.123,
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
        bss = sorted(new_bss)
        cur_default_bs = 512


def test_mab_stage(client, helpers):
    job_config = helpers.get_fake_job_config("test_mab_stage")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    bs_seq = []
    # Previous default batch size is converged
    for _ in range(50):
        # Predict
        response = client.get(
            GET_NEXT_BATCH_SIZE_URL,
            params={"job_id": job_config["job_id"]},
        )
        assert response.status_code == 200
        bs = response.json()["batch_size"]
        trial_number = response.json()["trial_number"]
        bs_seq.append(response.json()["batch_size"])
        # Concurrent job
        response = client.get(
            GET_NEXT_BATCH_SIZE_URL,
            params={"job_id": job_config["job_id"]},
        )
        assert response.status_code == 200
        bs_seq.append(response.json()["batch_size"])

        response = client.post(
            REPORT_RESULT_URL,
            json={
                "job_id": job_config["job_id"],
                "batch_size": bs,
                "trial_number": trial_number,
                "error": False,
                "time": 15.123,
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
    print(bs_seq)


def test_end_trial(client, helpers):
    job_config = helpers.get_fake_job_config("test_end_trial")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    # Start trial
    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    assert response.status_code == 200
    trial_number = response.json()["trial_number"]
    bs = response.json()["batch_size"]

    # End Trial.
    response = client.patch(
        REPORT_END_URL,
        json={
            "job_id": job_config["job_id"],
            "batch_size": bs,
            "trial_number": trial_number,
        },
    )
    assert response.status_code == 200

    # Start trial
    response = client.get(
        GET_NEXT_BATCH_SIZE_URL,
        params={"job_id": job_config["job_id"]},
    )
    assert response.status_code == 200
    trial_number = response.json()["trial_number"]
    bs = response.json()["batch_size"]

    # Report result.
    response = client.post(
        REPORT_RESULT_URL,
        json={
            "job_id": job_config["job_id"],
            "batch_size": bs,
            "trial_number": trial_number,
            "error": False,
            "time": 15.123,
            "energy": 3000.123,
            "max_power": 300,
            "metric": 0.55,
            "current_epoch": 98,
        },
    )

    # End Trial.
    response = client.patch(
        REPORT_END_URL,
        json={
            "job_id": job_config["job_id"],
            "batch_size": bs,
            "trial_number": trial_number,
        },
    )
    assert response.status_code == 200


def test_delete_job(client, helpers):
    job_config = helpers.get_fake_job_config("test_delete_job")
    response = client.post(REGISTER_JOB_URL, json=job_config)
    assert response.status_code == 201

    response = client.delete(
        DELETE_JOB_URL,
        params={"job_id": job_config["job_id"]},
    )
    print(response.text)
    assert response.status_code == 200

    response = client.post(REGISTER_JOB_URL, json=job_config)
    print(response.text)
    assert response.status_code == 201

    response = client.delete(
        DELETE_JOB_URL,
        params={"job_id": job_config["job_id"]},
    )
    print(response.text)
    assert response.status_code == 200

    # Job doesn't exists
    response = client.delete(
        DELETE_JOB_URL,
        params={"job_id": "UNKNOWN"},
    )
    print(response.text)
    assert response.status_code == 404
