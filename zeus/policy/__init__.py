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
Optimization policies for Zeus.

[`PowerLimitOptimizer`][zeus.policy.interface.PowerLimitOptimizer] and
[`BatchSizeOptimizer`][zeus.policy.interface.BatchSizeOptimizer] are
abstract classes. Users can implement custom policies by extending the
abstract classes and implementing required method.
Currently, [`Simulator`][zeus.simulate.Simulator] supports custom policies
for both classes, while [`ZeusMaster`][zeus.run.ZeusMaster] only supports
custom [`BatchSizeOptimizer`][zeus.policy.interface.BatchSizeOptimizer]s.
Custom [`PowerLimitOptimizer`][zeus.policy.PowerLimitOptimizer]s will
have to be integrated into [`ZeusDataLoader`][zeus.run.ZeusDataLoader].
"""

from zeus.policy.interface import BatchSizeOptimizer, PowerLimitOptimizer
from zeus.policy.optimizer import JITPowerLimitOptimizer, PruningGTSBatchSizeOptimizer
