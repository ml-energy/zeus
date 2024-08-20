"""Utilities for analyzing stuff including profiling results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from perseus.models import ProfilingResult, PipeInstruction


def total_time_and_energy(
    profs: list[ProfilingResult], warmup_steps: int = 2,
) -> tuple[float, float]:
    """Calculate the total time and energy consumption of a list of profiling results.

    Args:
        profs: List of profiling results.
        warmup_steps: Number of warmup steps to remove from the beginning of iter_time/iter_energy.
    """
    iter_time_arr = np.array([p.iter_time for p in profs])
    iter_time = iter_time_arr[:, warmup_steps:].max(axis=0).mean().item()

    iter_energy_arr = np.array([p.iter_energy for p in profs])
    iter_energy = iter_energy_arr[:, warmup_steps:].sum(axis=0).mean().item()

    return iter_time, iter_energy


def pareto_frontier(
    times: list[float],
    energies: list[float],
    frequencies: list[int],
) -> tuple[list[float], list[float], list[int]]:
    """Get the pareto frontier of a list of times and energies.

    Keeps the order of items in the lists intact.

    Args:
        times: List of times.
        energies: List of energies.
        frequencies: List of frequencies.
    """
    # Sanity checks
    if len(times) != len(energies):
        raise ValueError("The number of times and energies must be the same.")

    # Get the pareto frontier
    pareto_times = []
    pareto_energies = []
    pareto_frequencies = []
    for freq, time1, energy1 in zip(frequencies, times, energies):
        if any(
            time2 < time1 and energy2 < energy1
            for time2, energy2 in zip(times, energies)
        ):
            continue
        pareto_times.append(time1)
        pareto_energies.append(energy1)
        pareto_frequencies.append(freq)

    return pareto_times, pareto_energies, pareto_frequencies


def avg_instruction_time_per_stage(
    profs: list[ProfilingResult], warmup_steps: int, instruction: PipeInstruction
) -> list[float]:
    """Get the average time of a single instruction for each stage.

    Args:
        profs: List of profling results.
        warmup_steps: Number of warmup steps to remove from the beginning of time_breakdown.
        instruction: The PipeInstruction to compute the average time for.
    """
    return [
        np.mean(prof.time_breakdown[instruction][warmup_steps:]).item()
        for prof in profs
    ]


def remove_extremes(arr: NDArray, num: int = 0) -> NDArray:
    """Remove extreme numbers from a 1D array.

    Args:
        arr: Array to filter from.
        num: If N is given, remove N maximum and N minimum values from the array.
    """
    # Sanity checks
    if arr.ndim != 1:
        raise ValueError("This function only deals with 1D arrays.")
    if 2 * num >= arr.shape[0]:
        raise ValueError(
            f"Removing {num} max and min items will remove everything from the array."
        )

    if num > 0:
        arr = np.partition(arr, kth=[-num, num])[num:-num]

    return arr
