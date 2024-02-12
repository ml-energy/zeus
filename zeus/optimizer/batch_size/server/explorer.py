from math import cos
from typing import Generator

import numpy as np


class PruningExploreManager:
    """Helper class that generates batch sizes to explore and prune."""

    def __init__(
        self,
        batch_sizes: list[int],
        default: int,
        num_pruning_rounds,
    ) -> None:
        """Initialze the object.

        Args:
            batch_sizes: The initial set of batch sizes to prune from.
            default: The default batch size (b0) to begin exploration from.
            num_pruning_rounds: How many rounds to run pruning.
        """
        # Sanity checks.
        if default not in batch_sizes:
            raise ValueError(f"Default batch size {default} not in {batch_sizes}.")

        # Save arguments.
        self.batch_sizes = batch_sizes
        self.default = default
        self.num_pruning_rounds = num_pruning_rounds

        # State
        self.expecting = default

        # Generator that returns batch sizes.
        self.gen = self._exploration_engine()

    """
    TODO: Probably just make this as a function, not a generator.
        Keep the search space list, and just return an index to use next. 
        Then observe the cost based on the index. 
        Currently, predict(), predict() will make an error 
    """

    def _exploration_engine(
        self,
    ) -> Generator[int | None, tuple[int, float, bool], list[int]]:
        """Drive pruning exploration.

        Yields the batch size to be explored.
        The caller should `send` a tuple of (explored batch size, cost, whether reached).
        As a safety measure, the explored batch size must match the most recently yielded
        batch size, and otherwise a `RuntimeError` is raised.
        Finally, when exploration is over, returns a sorted list of batch sizes that
        survived pruning.
        """
        for _ in range(self.num_pruning_rounds):
            # A list of batch sizes that reached the target metric.
            good: list[int] = []

            # We first explore downwards form the default batch size, and then go upwards.
            idx = self.batch_sizes.index(self.default)
            down = sorted(self.batch_sizes[: idx + 1], reverse=True)
            up = sorted(self.batch_sizes[idx + 1 :])

            # We track the best cost because the default batch size is updated to the batch
            # size that performed the best.
            best_cost = np.inf

            space = [down, up]
            self._log(f"Explore space: {space}")

            for bs_list in [down, up]:
                for bs in bs_list:
                    # We tell the outside world to explore `bs`, and we expect the outside
                    # world to give us back the cost of that `bs`.
                    self.expecting = bs
                    batch_size, cost, reached = yield bs
                    if self.expecting != batch_size:
                        raise RuntimeError(
                            f"PruningExplorationManager: {self.expecting=}, {batch_size=}"
                        )
                    self.expecting = 0

                    # An empty `yield` to not proceed to the next batch size when the caller
                    # `send`s in the results.
                    yield

                    # Only batch sizes that reached the target mteric are good.
                    if reached:
                        if best_cost > cost:
                            best_cost = cost
                            self.default = bs
                        self._log(f"Good batch size: {bs}")
                        good.append(bs)
                    # If the batch size did not reach the target metric, `break`ing here will
                    # allow us to move on to either the next direction of exploration (upwards)
                    # or end this round of pruning exploration.
                    else:
                        break

            self.expecting = 0
            self.batch_sizes = sorted(good)

        return sorted(self.batch_sizes)

    def next_batch_size(self) -> int:
        """Return the next batch size to explore.

        Raises `StopIteration` when pruning exploration phase is over.
        The exception instance contains the final set of batch sizes to consider.
        Access it through `exception.value`.
        """
        batch_size = next(self.gen)
        assert batch_size is not None, "Call order may have been wrong."
        return batch_size

    def report_batch_size_result(
        self, batch_size: int, cost: float, reached: bool
    ) -> None:
        """Report whether the previous batch size reached the target metric.

        Args:
            batch_size: The batch size which this cost observation is from.
            cost: The energy-time cost of running the job with this batch size.
            reached: Whether the job reached the target metric.
        """
        self._log(f"Report: {batch_size}, {cost}, {reached}")
        none = self.gen.send((batch_size, cost, reached))
        assert none is None, "Call order may have been wrong."

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[{self.name}] {message}")

    @property
    def name(self) -> str:
        return "Pruning Explore Manager"
