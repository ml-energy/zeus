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

"""Compatibility layer for Pydantic v1 and v2.

We don't want to pin any specific version of Pydantic. With this, we can
import things from `zeus.util.pydantic_v1` and always use the V1 API
regardless of the installed version of Pydantic.

Inspired by Deepspeed:
https://github.com/microsoft/DeepSpeed/blob/5d754606/deepspeed/pydantic_v1.py
"""

try:
    from pydantic.v1 import *
except ImportError:
    from pydantic import *
