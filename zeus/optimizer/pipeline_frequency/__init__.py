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

"""Optimize the energy consumption of large model training with Perseus.

A a high-level, this optimizer assigns each forward and backward computation
in a pipeline parallel training iteration with a GPU frequency that leads to
a Pareto-optimal training iteration time and energy consumption.

Currently, this optimizer depends on PyTorch.
"""

from zeus.optimizer.pipeline_frequency.optimizer import PipelineFrequencyOptimizer
