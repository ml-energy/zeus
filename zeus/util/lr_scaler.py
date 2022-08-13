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

"""Classes that enclose learning rate scaling rules."""

import math
from dataclasses import dataclass


@dataclass
class SquareRootScaler:
    """Square root scaling.

    Args:
        bs: The initial batch size
        lr: The initial learning rate
    """

    bs: int
    lr: float

    def compute_lr(self, new_bs: int) -> float:
        """Compute the scaled learning rate given the new batch size."""
        return self.lr * math.sqrt(new_bs / self.bs)


@dataclass
class LinearScaler:
    """Linear scaling.

    Args:
        bs: The initial batch size
        lr: The initial learning rate
    """

    bs: int
    lr: float

    def compute_lr(self, new_bs: int) -> float:
        """Compute the scaled learning rate given the new batch size."""
        return self.lr * new_bs / self.bs
