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

from __future__ import annotations

import pytest

from zeus.policy.optimizer import PruningExploreManager


class TestPruningExploreManager:
    """Unit test class for pruning exploration."""

    batch_sizes: list[int] = [8, 16, 32, 64, 128, 256]

    def run_exploration(
        self,
        manager: PruningExploreManager,
        exploration: list[tuple[int, float, bool]],
        result: list[int],
    ) -> None:
        """Drive the pruning explore manager and check results."""
        for bs, cost, reached in exploration:
            assert manager.next_batch_size() == bs
            manager.report_batch_size_result(bs, cost, reached)
        with pytest.raises(StopIteration) as raised:
            manager.next_batch_size()
        assert raised.value.value == result

    def test_normal(self):
        """Test a typical case."""
        manager = PruningExploreManager(self.batch_sizes, 128)
        exploration = [
            (128, 10.0, True),
            (64, 9.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 21.0, False),
            (256, 15.0, True),
            (32, 8.0, True),
            (16, 12.0, False),
            (64, 9.0, True),
            (128, 10.0, True),
            (256, 17.0, False),
        ]
        result = [32, 64, 128]
        self.run_exploration(manager, exploration, result)

    def test_default_is_largest(self):
        """Test the case when the default batch size is the largest one."""
        manager = PruningExploreManager(self.batch_sizes, 256)
        exploration = [
            (256, 7.0, True),
            (128, 8.0, True),
            (64, 9.0, True),
            (32, 13.0, True),
            (16, 22.0, False),
            (256, 8.0, True),
            (128, 8.5, True),
            (64, 9.0, True),
            (32, 12.0, True),
        ]
        result = [32, 64, 128, 256]
        self.run_exploration(manager, exploration, result)

    def test_default_is_smallest(self):
        """Test the case when the default batch size is the smallest one."""
        manager = PruningExploreManager(self.batch_sizes, 8)
        exploration = [
            (8, 10.0, True),
            (16, 17.0, True),
            (32, 20.0, True),
            (64, 25.0, False),
            (8, 10.0, True),
            (16, 21.0, False),
        ]
        result = [8]
        self.run_exploration(manager, exploration, result)

    def test_all_converge(self):
        """Test the case when every batch size converges."""
        manager = PruningExploreManager(self.batch_sizes, 64)
        exploration = [
            (64, 10.0, True),
            (32, 8.0, True),
            (16, 12.0, True),
            (8, 15.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
            (32, 7.0, True),
            (16, 10.0, True),
            (8, 15.0, True),
            (64, 10.0, True),
            (128, 12.0, True),
            (256, 13.0, True),
        ]
        result = self.batch_sizes
        self.run_exploration(manager, exploration, result)

    def test_every_bs_is_bs(self):
        """Test the case when every batch size other than the default fail to converge."""
        manager = PruningExploreManager(self.batch_sizes, 64)
        exploration = [
            (64, 10.0, True),
            (32, 22.0, False),
            (128, 25.0, False),
            (64, 9.0, True),
        ]
        result = [64]
        self.run_exploration(manager, exploration, result)
