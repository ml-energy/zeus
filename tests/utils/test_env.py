from __future__ import annotations

import os
from typing import Any
from contextlib import contextmanager

import pytest

from zeus.utils.env import get_env


@contextmanager
def temp_env(key: str, value: Any):
    if key in os.environ:
        old = os.environ[key]
        os.environ[key] = str(value)
        yield key, value
        os.environ[key] = old
    else:
        os.environ[key] = str(value)
        yield key, value
        os.environ.pop(key)


class TestGetEnv:
    """Test get_env."""

    def test_int(self):
        """Test the case when the value is type int."""
        with temp_env("TEST_INT", 123) as (key, value):
            assert get_env(key, int) == value

    def test_str(self):
        """Test the case when the value is type str."""
        with temp_env("TEST_STR", "hello world") as (key, value):
            assert get_env(key, str) == value

    def test_bool(self):
        """Test the case when the value is type bool."""
        with temp_env("TEST_BOOL", "true") as (key, _):
            assert get_env(key, bool) == True
        with temp_env("TEST_BOOL", "True") as (key, _):
            assert get_env(key, bool) == True
        with temp_env("TEST_BOOL", "false") as (key, _):
            assert get_env(key, bool) == False
        with temp_env("TEST_BOOL", "False") as (key, _):
            assert get_env(key, bool) == False
        with temp_env("TEST_BOOL", "fault") as (key, _):  # ruff: noqa: SIM117
            with pytest.raises(ValueError):
                get_env(key, bool)

    def test_no_env(self):
        """Test the case when there is no such env var."""
        with pytest.raises(ValueError):
            get_env("TEST_NONE", str)

    def test_default(self):
        """Test the case when there is no such env var but there's a default."""
        assert get_env("TEST_NONE", int, default=123) == 123

    def test_default_ignore(self):
        """Test the case when the default should be ignored when the env var exists."""
        with temp_env("TEST_DEFAULT", 123) as (key, value):
            assert get_env(key, int, default=321) == value
