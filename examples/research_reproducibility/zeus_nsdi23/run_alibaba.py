"""Example script for running the Zeus trace-driven simulator."""

from __future__ import annotations

import zlib
import argparse
import multiprocessing as mp
from pprint import pprint
from typing import Literal
from functools import lru_cache

import pandas as pd

from zeus._legacy.job import Job
from zeus._legacy.simulate import Simulator, HistoryEntry
from zeus._legacy.policy import JITPowerLimitOptimizer, PruningGTSBatchSizeOptimizer


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default="v100", choices=["a40", "v100", "p100", "rtx6000"]
    )
    parser.add_argument("--eta_knob", type=float, default=0.5)
    parser.add_argument("--beta_knob", type=float, default=2.0)
    return parser.parse_args()


def run_simulator(
    gpu: Literal["a40", "v100", "p100", "rtx6000"],
    eta_knob: float,
    beta_knob: float,
) -> list[tuple[str, list[HistoryEntry]]]:
    """Run the simulator on the given job."""
    # Read in the Alibaba trace
    alibaba_df = pd.DataFrame(pd.read_csv("trace/alibaba_groups.csv.xz"))
    print("Read in the Alibaba trace.")
    print(f"Number of groups: {alibaba_df.group.nunique()}")

    # Run simulation on all Alibaba recurring job groups.
    with mp.Pool(mp.cpu_count()) as p:
        alibaba_result = p.starmap(
            simulate_group,
            (
                (group, gpu, eta_knob, beta_knob)
                for _, group in alibaba_df.groupby("group")
            ),
        )

    return alibaba_result


def simulate_group(
    group: pd.DataFrame,
    gpu: Literal["a40", "v100", "p100", "rtx6000"],
    eta_knob: float,
    beta_knob: float,
) -> tuple[str, list[HistoryEntry]]:
    """Perform trace-driven simulation on one Alibaba recurring job group."""
    job = get_job_with_defaults(gpu, group.dataset.unique().item())

    # Deterministic hashing.
    seed = zlib.adler32(group.group.unique().item().encode("utf-8"))

    # Instantiate optimizers.
    bso = PruningGTSBatchSizeOptimizer(seed=seed, concurrency=True, verbose=False)
    plo = JITPowerLimitOptimizer(verbose=False)

    # Instantitate the simulator.
    simulator = Simulator(
        read_train_trace(), read_power_trace(gpu), bso, plo, seed=seed, verbose=False
    )

    # Run the simulator.
    history = simulator.simulate_one_alibaba_group(
        job, group, beta_knob=beta_knob, eta_knob=eta_knob
    )

    return (group.dataset.unique().item(), history)


@lru_cache(maxsize=1)
def read_train_trace() -> pd.DataFrame:
    """Read the train trace file as a Pandas DataFrame."""
    return pd.DataFrame(pd.read_csv("trace/summary_train.csv"))


@lru_cache(maxsize=1)
def read_power_trace(gpu: Literal["a40", "v100", "p100", "rtx6000"]) -> pd.DataFrame:
    """Read the power trace of the given GPU as a Pandas DataFrame."""
    return pd.DataFrame(pd.read_csv(f"trace/summary_power_{gpu}.csv"))


def get_job_with_defaults(
    gpu: Literal["a40", "v100", "p100", "rtx6000"], dataset: str
) -> Job:
    """Instantiate a Job instance with defaults for the given dataset."""
    if dataset not in [
        "cifar100",
        "imagenet",
        "squad",
        "librispeech",
        "movielens-1m",
        "sentiment140",
    ]:
        raise NotImplementedError(f"Unknown dataset {dataset}.")

    # Since GPUs have different VRAM capacities, the maximum batch size changes.
    power_df = read_power_trace(gpu)
    bmax = power_df.loc[power_df.dataset == dataset].batch_size.max().item()

    if dataset.lower() == "cifar100":
        b0 = min(1024, bmax)
        return Job("cifar100", "shufflenetv2", "adadelta", 0.6, 100, b0, 0.1)
    elif dataset.lower() == "imagenet":
        b0 = min(256, bmax)
        return Job("imagenet", "resnet50", "adadelta", 0.65, 100, b0)
    elif dataset.lower() == "squad":
        b0 = min(32, bmax)
        return Job("squad", "bert_base_uncased", "adamw", 84.0, 6, b0)
    elif dataset.lower() == "librispeech":
        b0 = min(256, bmax)
        return Job("librispeech", "deepspeech2", "adamw", 40.0, 16, b0)
    elif dataset.lower() == "movielens-1m":
        b0 = min(1024, bmax)
        return Job("movielens-1m", "ncf", "adam", 0.41, 100, b0)
    elif dataset.lower() == "sentiment140":
        b0 = min(128, bmax)
        return Job("sentiment140", "bert_base_uncased", "adamw", 0.84, 10, b0, 4.00e-7)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}.")


if __name__ == "__main__":
    # Parse commandline arguments.
    args = parse_args()

    # Run the simulator.
    history = run_simulator(args.gpu, args.eta_knob, args.beta_knob)

    # Print out the list of HistoryEntry's.
    pprint(history[:20])
