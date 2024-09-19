"""Example script for running the Zeus trace-driven simulator."""

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Literal

import pandas as pd

from zeus._legacy.job import Job
from zeus._legacy.policy.optimizer import (
    JITPowerLimitOptimizer,
    PruningGTSBatchSizeOptimizer,
)
from zeus._legacy.simulate import Simulator, HistoryEntry


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--model", default="deepspeech2")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--target_metric", type=float, default=40.0)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--b_0", type=int, default=192)
    parser.add_argument(
        "--gpu", default="v100", choices=["a40", "v100", "p100", "rtx6000"]
    )
    parser.add_argument("--eta_knob", type=float, default=0.5)
    parser.add_argument("--beta_knob", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--num_recurrence",
        type=int,
        default=None,
        help="If None, 2*|B|*|P| will be used as in the paper.",
    )

    return parser.parse_args()


def read_trace(
    gpu: Literal["a40", "v100", "p100", "rtx6000"]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the train and power trace files as Pandas DataFrames."""
    train_df = pd.DataFrame(pd.read_csv("trace/summary_train.csv"))
    power_df = pd.DataFrame(pd.read_csv(f"trace/summary_power_{gpu}.csv"))
    return train_df, power_df


def run_simulator(
    job: Job,
    gpu: Literal["a40", "v100", "p100", "rtx6000"],
    eta_knob: float,
    beta_knob: float,
    num_recurrence: int | None,
    seed: int = 1,
) -> list[HistoryEntry]:
    """Run the simulator on the given job."""
    # Read in the train and power traces.
    train_df, power_df = read_trace(gpu)

    # Instantiate optimizers.
    plo = JITPowerLimitOptimizer(verbose=True)
    bso = PruningGTSBatchSizeOptimizer(seed=seed, verbose=True)

    # Instantitate the simulator.
    simulator = Simulator(train_df, power_df, bso, plo, seed=seed, verbose=True)

    # Use 2 * |B| * |P| is num_recurrence is None.
    print(num_recurrence)
    if num_recurrence is None:
        job_df = job.filter_df(train_df.merge(power_df, how="inner"))
        num_recurrence = (
            2 * len(job_df.batch_size.unique()) * len(job_df.power_limit.unique())
        )

    # Run the simulator.
    print(num_recurrence)
    return simulator.simulate_one_job(job, num_recurrence, beta_knob, eta_knob)


def main(args: argparse.Namespace) -> None:
    """Run the main routine."""
    # Instantitate the job specification dataclass.
    job = Job(
        args.dataset,
        args.model,
        args.optimizer,
        args.target_metric,
        args.max_epochs,
        args.b_0,
    )

    # Run the simulator.
    history = run_simulator(
        job, args.gpu, args.eta_knob, args.beta_knob, args.num_recurrence, args.seed
    )

    # Print out the list of HistoryEntry's.
    pprint(history)


if __name__ == "__main__":
    main(parse_args())
