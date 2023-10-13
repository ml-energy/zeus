# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
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

"""Utilities for asyncio."""

from __future__ import annotations

import asyncio
import logging
import functools
from typing import Any, Coroutine, TypeVar

from zeus.util.logging import get_logger

T = TypeVar("T")
default_logger = get_logger(__name__)


def create_task(
    coroutine: Coroutine[Any, Any, T],
    logger: logging.Logger | None = None,
) -> asyncio.Task[T]:
    """Create an `asyncio.Task` but ensure that exceptions are logged.

    Reference: https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/

    Args:
        coroutine: The coroutine to be wrapped.
        logger: The logger to be used for logging exceptions. If `None`, the
            the logger with the name `zeus.util.async_utils` is used.
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine)
    task.add_done_callback(
        functools.partial(_handle_task_exception, logger=logger or default_logger)
    )
    return task


def _handle_task_exception(task: asyncio.Task, logger: logging.Logger) -> None:
    """Print out exception and tracebook when a task dies with an exception."""
    try:
        task.result()
    except asyncio.CancelledError:
        # Cancellation should not be logged as an error.
        pass
    except Exception:
        # `logger.exception` automatically handles exception and traceback info.
        logger.exception("Job task died with an exception!")
