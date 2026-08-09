"""Tests for thermally stable energy profiling."""

from __future__ import annotations

from typing import Any

from zeus import profile


def test_measure_uses_iteration_based_warmup_by_default(mocker: Any) -> None:
    """Use the configured warmup iterations for calibration and the trial."""
    target_function = mocker.MagicMock()
    zeus_monitor = mocker.MagicMock()
    trial_result = mocker.sentinel.trial_result
    calibrate = mocker.patch(
        "zeus.profile._calibrate_iteration_duration",
        return_value=0.25,
    )
    run_trial = mocker.patch(
        "zeus.profile._run_trial",
        return_value=trial_result,
    )

    result = profile.measure(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        measurement_duration=4.0,
        cooldown_duration=1.0,
        num_warmup_iterations=7,
        num_calibration_iterations=20,
    )

    calibrate.assert_called_once_with(target_function, zeus_monitor, 7, 20)
    run_trial.assert_called_once_with(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        cooldown_duration=1.0,
        measurement_duration=4.0,
        num_warmup_iterations=7,
        iteration_duration=0.25,
    )
    assert result is trial_result


def test_measure_converts_warmup_settle_duration_to_iterations(mocker: Any) -> None:
    """Override only the trial warmup with the requested settle duration."""
    target_function = mocker.MagicMock()
    zeus_monitor = mocker.MagicMock()
    trial_result = mocker.sentinel.trial_result
    calibrate = mocker.patch(
        "zeus.profile._calibrate_iteration_duration",
        return_value=0.3,
    )
    run_trial = mocker.patch(
        "zeus.profile._run_trial",
        return_value=trial_result,
    )

    result = profile.measure(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        measurement_duration=4.0,
        cooldown_duration=1.0,
        num_warmup_iterations=5,
        num_calibration_iterations=20,
        warmup_settle_duration=2.0,
    )

    calibrate.assert_called_once_with(target_function, zeus_monitor, 5, 20)
    run_trial.assert_called_once_with(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        cooldown_duration=1.0,
        measurement_duration=4.0,
        num_warmup_iterations=6,
        iteration_duration=0.3,
    )
    assert result is trial_result


def test_measure_rejects_negative_warmup_settle_duration(mocker: Any) -> None:
    """Reject a negative settle duration before running calibration."""
    calibrate = mocker.patch("zeus.profile._calibrate_iteration_duration")

    try:
        profile.measure(
            target_function=mocker.MagicMock(),
            zeus_monitor=mocker.MagicMock(),
            measurement_duration=4.0,
            cooldown_duration=1.0,
            warmup_settle_duration=-1.0,
        )
    except ValueError as error:
        assert str(error) == "warmup_settle_duration must be non-negative"
    else:
        raise AssertionError("Expected a negative warmup settle duration to raise ValueError")

    calibrate.assert_not_called()


def test_run_trial_reads_temperature_after_warmup_before_measurement(mocker: Any) -> None:
    """Synchronize warmup and read temperature before opening the energy window."""
    events = []
    target_function = mocker.MagicMock(side_effect=lambda: events.append("target"))
    zeus_monitor = mocker.MagicMock()
    zeus_monitor.gpu_indices = [0]
    zeus_monitor.sync_with = "torch"
    zeus_monitor.begin_window.side_effect = lambda *args, **kwargs: events.append("begin")
    zeus_monitor.end_window.side_effect = lambda *args, **kwargs: (
        events.append("end") or mocker.MagicMock(total_energy=4.0, time=2.0)
    )
    sync_execution = mocker.patch(
        "zeus.profile.sync_execution",
        side_effect=lambda *args, **kwargs: events.append("sync"),
    )
    read_temperature = mocker.patch(
        "zeus.profile._read_avg_gpu_temperature",
        side_effect=lambda *args: events.append("temperature") or 40.0,
    )
    mocker.patch("zeus.profile.all_reduce", side_effect=lambda values, operation: values)
    mocker.patch("zeus.profile.get_world_size", return_value=1)

    profile._run_trial(
        target_function=target_function,
        zeus_monitor=zeus_monitor,
        cooldown_duration=0.0,
        measurement_duration=1.0,
        num_warmup_iterations=2,
        iteration_duration=0.5,
    )

    assert events == [
        "target",
        "target",
        "sync",
        "temperature",
        "begin",
        "target",
        "target",
        "end",
        "temperature",
    ]
    sync_execution.assert_called_once_with([0], sync_with="torch")
    zeus_monitor.begin_window.assert_called_once_with(
        "__zeus_profile_run_trial",
        sync_execution=False,
    )
    assert read_temperature.call_count == 2
