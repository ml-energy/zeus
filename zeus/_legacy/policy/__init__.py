"""Optimization policies for Zeus.

[`PowerLimitOptimizer`][zeus._legacy.policy.interface.PowerLimitOptimizer] and
[`BatchSizeOptimizer`][zeus._legacy.policy.interface.BatchSizeOptimizer] are
abstract classes. Users can implement custom policies by extending the
abstract classes, implementing required methods, and plugging them into
the [`Simulator`][zeus._legacy.simulate.Simulator].
"""

from zeus._legacy.policy.interface import BatchSizeOptimizer, PowerLimitOptimizer
from zeus._legacy.policy.optimizer import (
    JITPowerLimitOptimizer,
    PruningGTSBatchSizeOptimizer,
)
