"""Utilities for saving and loading Perseus states."""

import os
import aiofiles
from pydantic import BaseModel

from perseus.models import (
    ProfilingResult,
    PowerStateSchedule,
    RankInfo,
    ServerInfo,
)


async def save_prof(
    data: list[ProfilingResult],
    directory: str,
    schedule_num: int,
) -> None:
    """Save a list of `ProfilingResult`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/{schedule_num}.prof.json", "w") as f:
        obj = _ProfilingResultList(__root__=data).json()
        await f.write(obj)


def load_prof(directory: str, schedule_num: int) -> list[ProfilingResult]:
    """Load a list of `ProfilingResult`s saved in the designated directory."""
    filepath = f"{directory}/{schedule_num}.prof.json"
    return _ProfilingResultList.parse_file(filepath).__root__


async def save_sched(
    data: list[PowerStateSchedule],
    directory: str,
    schedule_num: int,
) -> None:
    """Save a list of `PowerStateSchedule`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/{schedule_num}.sched.json", "w") as f:
        obj = _PowerStateScheduleList(__root__=data).json()
        await f.write(obj)


def load_sched(directory: str, schedule_num: int) -> list[PowerStateSchedule]:
    """Load a list of `PowerStateSchedule`s saved in the designated directory."""
    filepath = f"{directory}/{schedule_num}.sched.json"
    return _PowerStateScheduleList.parse_file(filepath).__root__


async def save_ranks(data: list[RankInfo], directory: str) -> None:
    """Save a list of `RankInfo`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/ranks.json", "w") as f:
        obj = _RankInfoList(__root__=data).json()
        await f.write(obj)


def load_ranks(directory: str) -> list[RankInfo]:
    """Load a list of `RankInfo`s saved in the designated directory."""
    filepath = f"{directory}/ranks.json"
    return _RankInfoList.parse_file(filepath).__root__


async def save_info(data: ServerInfo, directory: str) -> None:
    """Save `ServerInfo` in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/perseus.json", "w") as f:
        obj = data.json()
        await f.write(obj)


def load_info(directory: str) -> ServerInfo:
    """Load `ServerInfo` saved in the designated directory."""
    filepath = f"{directory}/perseus.json"
    return ServerInfo.parse_file(filepath)


# Proxy classes for a list of Pydantic objects.
# __root__ is making use of Pydantic's Custom Root Type for a cleaner JSON representation.


class _ProfilingResultList(BaseModel):
    __root__: list[ProfilingResult]


class _PowerStateScheduleList(BaseModel):
    __root__: list[PowerStateSchedule]


class _RankInfoList(BaseModel):
    __root__: list[RankInfo]
