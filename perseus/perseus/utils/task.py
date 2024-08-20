"""Utilities for dealing with `asyncio.Task`."""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

from fastapi.logger import logger

T = TypeVar("T")


def create_task(coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """Create an `asyncio.Task` but ensure that exceptions are logged.

    Reference: https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine)
    task.add_done_callback(_handle_task_exception)
    return task


def _handle_task_exception(task: asyncio.Task) -> None:
    """Print out exception and tracebook when a task dies with an exception."""
    try:
        task.result()
    except asyncio.CancelledError:
        # Cancellation should not be logged as an error.
        pass
    except Exception:  # pylint: disable=broad-except
        # `logger.exception` automatically handles exception and traceback info.
        logger.exception("Job task died with an exception!")
