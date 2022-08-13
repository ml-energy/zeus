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

"""Classes for running actual jobs with Zeus.

[`ZeusDataLoader`][zeus.run.ZeusDataLoader] is to be embedded in the training
script, automatically taking care of profiling power and determining the optimal
power limit. [`ZeusMaster`][zeus.run.ZeusMaster] is the top-level class that
drives batch size optimization (with Multi-Armed Bandit) and spawns training jobs.
"""

from zeus.run.master import ZeusMaster
from zeus.run.dataloader import ZeusDataLoader
