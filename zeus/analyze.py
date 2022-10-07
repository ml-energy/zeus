# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for result analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.metrics import auc


@dataclass
class HistoryEntry:
    """Represents the config and result of a job run that may have failed.

    Attributes:
        bs: Batch size
        pl: Power limit
        energy: Energy consumption in Joules
        reached: Whether the target metric was reached at the end
        time: Time consumption in seconds
    """

    bs: int
    pl: int
    energy: float
    reached: bool
    time: float


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
    return area / (seconds.max() - seconds.min())


def _get_seconds(df: pd.DataFrame) -> pd.Series:
    return df["Time"].map(lambda t: t.timestamp())


def _get_watts(df: pd.DataFrame) -> pd.Series:
    return df["Power"].div(1000.0)
