"""Shared test fixtures."""

import pytest

import zeus.device.gpu
import zeus.device.cpu


@pytest.fixture(autouse=True, scope="function")
def reset_gpus_and_cpus() -> None:
    """Reset the global variable `_gpus` and `_cpus` to None on every test."""
    zeus.device.gpu._gpus = None
    zeus.device.cpu._cpus = None
