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

"""The Perseus server guides the PerseusOptimizer with frequency plans.

The server is agnostic to the training framework the PerseusOptimizer
is integrated with. A server is useful because large model training is
typically distributed, and we still need one place to coordinate the
frequency plans. Later, the server will be extended to support complete
online profiling and optimization.
"""
