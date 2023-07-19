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

"""Tools for checking stuff."""

from __future__ import annotations

import os
from typing import Type, TypeVar, cast

T = TypeVar("T")


def get_env(name: str, valtype: Type[T], default: T | None = None) -> T:
    """Fetch an environment variable and cast it to the given type."""
    try:
        if valtype == bool:
            val = os.environ[name].lower()
            if val not in ["true", "false"]:
                raise ValueError(f"Strange boolean environment variable value '{val}'")
            return cast(T, val == "true")
        return valtype(os.environ[name])
    except KeyError:
        if default is not None:
            return default
        raise ValueError(f"Missing environment variable '{name}'") from None
