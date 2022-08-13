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

"""Tools for checking stuff."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Any, Type, TypeVar, cast

T = TypeVar("T")


def get_env(name: str, valtype: Type[T], default: T | None = None) -> T:
    """Fetch an environment variable and cast it to the given type."""
    try:
        if valtype == bool:
            val = os.environ[name].lower()
            if val not in ["true", "false"]:
                raise RuntimeError(
                    f"Strange boolean environment variable value '{val}'"
                )
            return cast(T, val == "true")
        return valtype(os.environ[name])
    except KeyError:
        if default is not None:
            return default
        raise RuntimeError(f"Zeus requires the environment variable '{name}'") from None


@contextmanager
def _temp_env(key: str, value: Any):
    if key in os.environ:
        old = os.environ[key]
        os.environ[key] = str(value)
        yield key, value
        os.environ[key] = old
    else:
        os.environ[key] = str(value)
        yield key, value
        os.environ.pop(key)


class TestGetEnv(unittest.TestCase):
    """Test get_env."""

    def test_int(self):
        """Test the case when the value is type int."""
        with _temp_env("TEST_INT", 123) as (key, value):
            self.assertEqual(get_env(key, int), value)

    def test_str(self):
        """Test the case when the value is type str."""
        with _temp_env("TEST_STR", "hello world") as (key, value):
            self.assertEqual(get_env(key, str), value)

    def test_bool(self):
        """Test the case when the value is type bool."""
        with _temp_env("TEST_BOOL", "true") as (key, _):
            self.assertEqual(get_env(key, bool), True)
        with _temp_env("TEST_BOOL", "True") as (key, _):
            self.assertEqual(get_env(key, bool), True)
        with _temp_env("TEST_BOOL", "false") as (key, _):
            self.assertEqual(get_env(key, bool), False)
        with _temp_env("TEST_BOOL", "False") as (key, _):
            self.assertEqual(get_env(key, bool), False)
        with _temp_env("TEST_BOOL", "fault") as (key, _):
            with self.assertRaises(RuntimeError):
                get_env(key, bool)

    def test_no_env(self):
        """Test the case when there is no such env var."""
        with self.assertRaises(RuntimeError):
            get_env("TEST_NONE", str)

    def test_default(self):
        """Test the case when there is no such env var but there's a default."""
        self.assertEqual(get_env("TEST_NONE", int, default=123), 123)

    def test_default_ignore(self):
        """Test the case when the default should be ignored when the env var exists."""
        with _temp_env("TEST_DEFAULT", 123) as (key, value):
            self.assertEqual(get_env(key, int, default=321), value)


if __name__ == "__main__":
    unittest.main()
