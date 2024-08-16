"""Defines the energy-time cost metric function."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.metrics import auc


def zeus_cost(
    energy: float, time: float, eta_knob: float, max_power: int | float
) -> float:
    """Compute Zeus's energy-time cost metric.

    Trades off ETA and TTA based on the value of `eta_knob`.
    The caller is expected to do bound checking for `eta_knob`,
    because `eta_knob` does not change frequently.

    Args:
        energy: Joules
        time: seconds
        eta_knob: Real number in [0, 1].
        max_power: The maximum power limit of the GPU.

    Returns:
        The cost of the DL training job.
    """
    return eta_knob * energy + (1 - eta_knob) * max_power * time


# ruff: noqa: PLR2004
def energy(
    logfile: Path | str,
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Compute the energy consumption from the Zeus monitor power log file.

    `start` and `end` are in units of seconds, relative to the beginning of
    the time window captured by the log file. Only the time window between
    `start` and `end` will be considered when computing energy.

    `start` and `end` can be negative, in which case the pointers wrap around
    and effectively the absolute value is subtracted from the end of the window.

    Args:
        logfile: Path to the power log file produced by the Zeus monitor.
        start: Start time of the window to consider.
        end: End time of the window to consider.
    """
    df = cast(pd.DataFrame, pd.read_csv(logfile, engine="python", skipfooter=1))
    df["Time"] = pd.to_datetime(df["Time"])
    start_timestamp = df.iloc[0]["Time"]
    end_timestamp = df.iloc[-1]["Time"]
    if start is not None:
        origin = start_timestamp if start >= 0.0 else end_timestamp
        df = df.loc[df["Time"] >= origin + timedelta(seconds=start)]
    if end is not None:
        origin = start_timestamp if end >= 0.0 else end_timestamp
        df = df.loc[df["Time"] <= origin + timedelta(seconds=end)]
    seconds = _get_seconds(df)
    watts = _get_watts(df)
    return auc(seconds, watts)


def avg_power(
    logfile: Path | str,
    start: float | None = None,
    end: float | None = None,
) -> float:
    """Compute the average power consumption from the Zeus monitor power log file.

    `start` and `end` are in units of seconds, relative to the beginning of
    the time window captured by the log file. Only the time window between
    `start` and `end` will be considered when computing average power.

    `start` and `end` can be negative, in which case the pointers wrap around
    and effectively the absolute value is subtracted from the end of the window.

    Args:
        logfile: Path to the power log file produced by the Zeus monitor.
        start: Start time of the window to consider.
        end: End time of the window to consider.

    Raises:
        ValueError: From `sklearn.metrics.auc`, when the duration of the
            profiling window is too small.
    """
    df = cast(pd.DataFrame, pd.read_csv(logfile, engine="python", skipfooter=1))
    df["Time"] = pd.to_datetime(df["Time"])
    if start is not None:
        df = df.loc[df["Time"] >= df.iloc[0]["Time"] + timedelta(seconds=start)]
    if end is not None:
        df = df.loc[df["Time"] <= df.iloc[0]["Time"] + timedelta(seconds=end)]
    seconds = _get_seconds(df)
    watts = _get_watts(df)
    area = auc(seconds, watts)
    return area / (max(seconds) - min(seconds))


def _get_seconds(df: pd.DataFrame) -> pd.Series:
    return df["Time"].map(lambda t: t.timestamp())


def _get_watts(df: pd.DataFrame) -> pd.Series:
    return df["Power"].div(1000.0)
