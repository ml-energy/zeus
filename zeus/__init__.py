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

"""
Zeus is an energy optimization framework for DNN training.

Modules:

- [`analyze`][zeus.analyze]: Functions for analyzing log files.
- [`profile`][zeus.profile]: Tools for profiling energy and time.
- [`job`][zeus.job]: Job specification.
- [`run`][zeus.run]: Machinery for actually running Zeus.
- [`simulate`][zeus.simulate]: Machinery for trace-driven Zeus.
- [`policy`][zeus.policy]: Collection of optimization policies.
- [`util`][zeus.util]: Utility functions and classes.
"""
