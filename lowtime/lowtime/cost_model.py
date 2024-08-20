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

"""Cost models for operations.

When operations have multiple discrete knobs, the cost model fits the
time-cost Pareto frontier to make the choice continuous. This is an
approximation that makes the discrete optimization problem solveable
with the Phillips-Dessouky algorithm.
"""

from __future__ import annotations

import logging
import itertools
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from lowtime.operation import CandidateExecutionOptions


logger = logging.getLogger(__name__)


class CostModel(ABC):
    """A continuous cost model fit from Pareto-optimal execution options."""

    @abstractmethod
    def __call__(self, quant_time: int) -> float:
        """Predict execution cost given quantized time."""
        ...

    def draw(
        self,
        ax: plt.Axes,
        options: CandidateExecutionOptions,
    ) -> None:
        """Plot a cost model's predictions with its target costs and save to path.

        Args:
            ax: Matplotlib axes to draw on.
            options: `quant_time` is taken as the x axis and `cost` will be drawn
                separately as the target (ground truth) cost plot.
        """
        quant_times = [option.quant_time for option in options.options]
        target_costs = [option.cost for option in options.options]

        # Plot the ground truth costs.
        ax.plot(quant_times, target_costs, "o")
        for option in options.options:
            ax.annotate(
                f"({option.quant_time}, {option.cost}, {option.knob})",
                (option.quant_time, option.cost),
            )

        # Plot the cost model's predictions.
        xs = np.arange(min(quant_times), max(quant_times), 0.01)
        ys = [self(x) for x in xs]
        ax.plot(xs, ys, "r-")

        # Plot metadata.
        ax.set_xlabel("quant_time")
        ax.set_ylabel("cost")


class ExponentialModel(CostModel):
    """An exponential cost model.

    cost = a * exp(b * quant_time) + c

    XXX(JW): For Perseus, first filter candidate execution options on measured cost.
    Then translate them into effective cost (cost - p2p_power * quant_time * unit_time)
    and fit the cost model on effective cost.
    """

    def __init__(
        self,
        options: CandidateExecutionOptions,
        initial_guess: tuple[float, float, float] | None = None,
        search_strategy: Literal["best", "first"] = "first",
    ) -> None:
        """Fit the cost model from Pareto-optimal execution options.

        Args:
            options: Candidate execution options to fit the cost model with.
            initial_guess: Initial guess for the parameters of the exponential function.
                If None, do a grid search on the initial guesses.
            search_strategy: Strategy to use when doing a grid search on the initial guesses.
                'first' means to take the first set of parameters that fit.
                'best' means to take the set of parameters that fit with the lowest error.
        """
        self.fn = lambda t, a, b, c: a * np.exp(b * t) + c

        quant_times = np.array([option.quant_time for option in options.options])
        target_costs = np.array([option.cost for option in options.options])

        def l2_error(coefficients: tuple[float, float, float]) -> float:
            preds = np.array([self.fn(t, *coefficients) for t in quant_times])
            return np.mean(np.square(target_costs - preds))

        # When an initial parameter guess is provided, just use it.
        if initial_guess is not None:
            self.coefficients, pcov = curve_fit(
                self.fn,
                quant_times,
                target_costs,
                p0=initial_guess,
                maxfev=50000,
            )
            if np.inf in pcov:
                raise ValueError("Initial guess failed to fit.")
            logger.info(
                "Exponential cost model with initial guesses: %s, coefficients: %s, L2 error: %f",
                initial_guess,
                self.coefficients,
                l2_error(self.coefficients),
            )
            return

        # Otherwise, do a grid search on the initial guesses.
        if search_strategy not in ["best", "first"]:
            raise ValueError("search_strategy must be either 'best' or 'first'.")

        unit_time = options.options[0].unit_time
        a_candidates = [1e1, 1e2, 1e3]
        b_candidates = [
            -c * unit_time / 0.001 for c in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
        ]
        c_candidates = [1e1, 1e2, 1e3]
        coefficients = self.run_grid_search(
            options, a_candidates, b_candidates, c_candidates, search_strategy
        )
        if coefficients is None:
            raise ValueError(
                "Grid search failed to fit. "
                "Manually fit the model with `Exponential.run_grid_search` "
                "and provide an initial guess to `Exponential.__init__`."
            )
        self.coefficients = coefficients

    def run_grid_search(
        self,
        options: CandidateExecutionOptions,
        a_candidates: list[float],
        b_candidates: list[float],
        c_candidates: list[float],
        search_strategy: Literal["best", "first"],
    ) -> tuple[float, float, float] | None:
        """Run a grid search on the initial guesses."""
        quant_times = np.array([option.quant_time for option in options.options])
        target_costs = np.array([option.cost for option in options.options])

        def l2_error(coefficients: tuple[float, float, float]) -> float:
            preds = np.array([self.fn(t, *coefficients) for t in quant_times])
            return np.mean(np.square(target_costs - preds))

        best_error = np.inf
        best_coefficients = (np.inf, np.inf, np.inf)
        initial_guess = next(
            itertools.product(a_candidates, b_candidates, c_candidates)
        )

        logger.info(
            "Running grid search for exponential model initial parameter guess."
        )
        for a, b, c in itertools.product(a_candidates, b_candidates, c_candidates):
            initial_guess = [a, b, c]
            (opt_a, opt_b, opt_c), pcov = curve_fit(
                self.fn,
                quant_times,
                target_costs,
                p0=initial_guess,
                maxfev=50000,
            )
            coefficients = (opt_a, opt_b, opt_c)

            # Skip if the fit failed.
            if np.inf in pcov:
                continue
            error = l2_error(coefficients)
            if error == np.inf:
                continue

            # The exponential model must be convex.
            if opt_a < 0.0:
                continue

            # We have coefficients that somewhat fit the data.
            logger.info(
                "Initial guess %s fit with coefficients %s and error %f.",
                initial_guess,
                coefficients,
                error,
            )
            if search_strategy == "first":
                logger.info("Strategy is 'first'. Search finished.")
                best_coefficients = coefficients
                best_error = error
                break
            elif search_strategy == "best":
                if error < best_error:
                    logger.info("Strategy is 'best' and error is better. Taking it.")
                    best_coefficients = coefficients
                    best_error = error
                else:
                    logger.info("Strategy is 'best' but error is worse.")

        if best_error == np.inf:
            raise ValueError("Nothing in the grid search space was able to fit.")

        logger.info("Final coefficients: %s", best_coefficients)
        return best_coefficients

    @lru_cache
    def __call__(self, quant_time: int) -> float:
        """Predict execution cost given quantized time."""
        return self.fn(quant_time, *self.coefficients)
