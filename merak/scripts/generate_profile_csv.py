"""Post-process time and energy profiling results from decoupled profiling mode."""

from __future__ import annotations

import argparse
import warnings
from glob import glob
from typing import Literal

import numpy as np
import pandas as pd


class PiecewiseLinearModel:
    """A energy model that connects (x, y) measurements with a straight line."""
    
    def __init__(self, x_measurements: np.ndarray, y_measurements: np.ndarray) -> None:
        """Initialize the model with measurements."""
        self.xs = x_measurements
        self.ys = y_measurements

        # Both X and Y measurements must be sorted.
        if not np.all(np.diff(x_measurements) >= 0):
            raise ValueError("X values must be sorted.")
        if not np.all(np.diff(y_measurements) >= 0):
            raise ValueError("Y values must be sorted.")

    def __call__(self, x: float) -> float:
        """Return the estimated y value at the given x value."""
        if x < self.xs[0] or x > self.xs[-1]:
            raise ValueError(f"X value {x} is out of range [{self.xs[0]}, {self.xs[-1]}].")
        return np.interp(x, self.xs, self.ys).item()


def main(
    profile_dir: str,
    num_microbatches: int,
    num_prof_steps: int,
    warmup_iters: int,
    gpu_type: Literal["A100", "A40"],
) -> None:
    """Run the main routine."""
    print(f"Processing decoupled profiling results in {profile_dir}.")

    # Enumerate supported GPU frequencies.
    if gpu_type == "A100":
        freqs = np.arange(1410, 210 - 15, -15).tolist()
    elif gpu_type == "A40":
        freqs = np.arange(1740, 210 - 15, -15).tolist()
    else:
        raise ValueError(f"Unsupported GPU type {gpu_type}.")
    print(f"Frequencies: {freqs}")

    # Read in energy polling results
    energy_files = glob(f"{profile_dir}/time-energy-*.csv")
    num_ranks = len(energy_files)
    if num_ranks == 0:
        raise RuntimeError("No energy polling results found.")
    print(f"Found {num_ranks} energy polling results.")
    models: list[PiecewiseLinearModel] = []
    for rank in range(num_ranks):
        df = pd.read_csv(f"{profile_dir}/time-energy-{rank}.csv")
        model = PiecewiseLinearModel(df.time.to_numpy(), df.energy.to_numpy())
        models.append(model)
        del df

    # Read in instruction timing results
    timing_files = sorted(glob(f"{profile_dir}/instructions-*.csv"))
    if len(timing_files) != num_ranks:
        raise RuntimeError(
            f"Expected {num_ranks} instruction timing results, but found {len(timing_files)}."
        )
    print(f"Found {num_ranks} instruction timing results.")
    timing_dfs = [pd.read_csv(f) for f in timing_files]

    # Only choose odd index "batch_input" records in the last rank.
    # That's because for each forward pass in the last rank, two "batch_input"s are
    # recorded: one from recv_grad_send_activationa and one from actual load_microbatch.
    last_rank_df = timing_dfs[-1]
    if "batch_input" in last_rank_df.instruction.values:
        other_records = last_rank_df.query("instruction != 'batch_input'")
        batch_input_records = (
            last_rank_df
            .query("instruction == 'batch_input'")
            .reset_index(drop=True)
            .iloc[1::2]
        )
        if (batch_input_records.end - batch_input_records.start).min() < 0.0001:
            warnings.warn(
                "Last rank batch_input records after filtering includes records shorter than 0.1 ms."
            )
        timing_dfs[-1] = pd.concat([other_records, batch_input_records]).reset_index(drop=True)

    # Assert same number of records.
    if "batch_input" in timing_dfs[0].instruction.values:
        lens = [len(df) for df in timing_dfs]
        for rank in range(num_ranks - 1):
            if lens[rank] != lens[rank + 1]:
                raise ValueError(
                    f"Rank {rank} has {lens[rank]} records, but rank {rank + 1} has {lens[rank + 1]} records."
                )
    
    # For each rank, the timing dataframe contains instruction start and end
    # timing measurements.
    inst_name_map = {"forward_microstep": "forward", "backward_microstep": "backward", "batch_input": "load"}
    profile_csv = open(f"{profile_dir}/profile.csv", "w")
    profile_csv.write("stage,instruction,frequency,time,energy\n")
    for rank in range(num_ranks):
        print(f"Processing rank {rank}.")
        for inst, name in inst_name_map.items():
            freqs_iter = iter(freqs)
            timing_df = timing_dfs[rank].query(f"instruction == '{inst}'")
            if timing_df.empty:
                print(f"  No {inst} found.")
                continue
            print(f"  Processing {inst}.")
            inst_times, inst_energies = [], []
            i = 0
            for _, (inst, start, end) in timing_df.iterrows():
                i += 1
                if i <= warmup_iters * num_microbatches * num_prof_steps:
                    continue
                inst_times.append(end - start)
                model = models[rank]
                inst_energies.append(model(end) - model(start))
                if i % (num_microbatches * num_prof_steps) == 0:
                    freq = next(freqs_iter)
                    profile_csv.write(f"{rank},{name},{freq},{np.mean(inst_times)},{np.mean(inst_energies)}\n")
                    inst_times, inst_energies = [], []
    profile_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_dir", help="Directory containing profiling results.")
    parser.add_argument("--num_microbatches", type=int, help="Number of microbatches.")
    parser.add_argument("--num_prof_steps", type=int, help="Number of profiling steps.")
    parser.add_argument("--gpu_type", choices=["A40", "A100"], help="Name of the GPU type.")
    args = parser.parse_args()

    warmup_iters = min(int(args.num_prof_steps * 0.1), 10)

    main(args.profile_dir, args.num_microbatches, args.num_prof_steps, warmup_iters, args.gpu_type)
