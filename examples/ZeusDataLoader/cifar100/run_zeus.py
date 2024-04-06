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

"""Example script for running Zeus on the CIFAR100 dataset."""

import argparse
import sys
from pathlib import Path

from zeus.job import Job
from zeus._legacy.policy import PruningGTSBatchSizeOptimizer
from zeus.run import ZeusMaster
from zeus.util import FileAndConsole


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()

    # This random seed is used for
    # 1. Multi-Armed Bandit inside PruningGTSBatchSizeOptimizer, and
    # 2. Providing random seeds for training.
    # Especially for 2, the random seed given to the nth recurrence job is args.seed + n.
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Default batch size and learning rate.
    # The first recurrence uses these parameters, and it must reach the target metric.
    # There is no learning rate because we train the model with Adadelta.
    parser.add_argument("--b_0", type=int, default=1024, help="Default batch size")

    # The range of batch sizes to consider. The example script generates a list of power-of-two
    # batch sizes, but batch sizes need not be power-of-two for Zeus.
    parser.add_argument(
        "--b_min", type=int, default=8, help="Smallest batch size to consider"
    )
    parser.add_argument(
        "--b_max", type=int, default=4096, help="Largest batch size to consider"
    )

    # The total number of recurrences.
    parser.add_argument(
        "--num_recurrence", type=int, default=100, help="Total number of recurrences"
    )

    # The \\eta knob trades off time and energy consumption. See Equation 2 in the paper.
    # The \\beta knob defines the early stopping threshold. See Section 4.4 in the paper.
    parser.add_argument(
        "--eta_knob", type=float, default=0.5, help="TTA-ETA tradeoff knob"
    )
    parser.add_argument(
        "--beta_knob", type=float, default=2.0, help="Early stopping threshold"
    )

    # Jobs are terminated when one of the three happens:
    # 1. The target validation metric is reached.
    # 2. The number of epochs exceeds the maximum number of epochs set.
    # 3. The cost of the next epoch is expected to exceed the early stopping threshold.
    parser.add_argument(
        "--target_metric", type=float, default=0.50, help="Target validation metric"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max number of epochs to train"
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run Zeus on CIFAR100."""
    # Zeus's batch size optimizer.
    # First prunes unpromising batch sizes, and then runs Gaussian Thompson Sampling MAB.
    bso = PruningGTSBatchSizeOptimizer(seed=args.seed, verbose=True)

    # The top-level class for running Zeus.
    # - The batch size optimizer is desinged as a pluggable policy.
    # - Paths (log_base and monitor_path) assume our Docker image's directory structure.
    # - For CIFAR100, one epoch of training may not take even ten seconds (especially for
    #   small models like squeezenet). ZeusDataLoader automatically handles profiling power
    #   limits over multiple training epochs such that the profiling window of each power
    #   limit fully fits in one epoch. However, the epoch duration may be so short that
    #   profiling even one power limit may not fully fit in one epoch. In such cases,
    #   ZeusDataLoader raises a RuntimeError, and the profiling window should be narrowed
    #   by giving smaller values to profile_warmup_iters and profile_measure_iters in the
    #   constructor of ZeusMaster.
    master = ZeusMaster(
        batch_size_optimizer=bso,
        log_base="/workspace/zeus_logs",
        seed=args.seed,
        monitor_path="/workspace/zeus/zeus_monitor/zeus_monitor",
        observer_mode=False,
        profile_warmup_iters=10,
        profile_measure_iters=40,
    )

    # Definition of the CIFAR100 job.
    # The `Job` class encloses all information needed to run training. The `command` parameter is
    # a command template. Curly-braced parameters are recognized by Zeus and automatically filled.
    job = Job(
        dataset="cifar100",
        network="shufflenet",
        optimizer="adadelta",
        target_metric=args.target_metric,
        max_epochs=args.max_epochs,
        default_bs=args.b_0,
        default_lr=0.1,  # Dummy placeholder value.
        workdir="/workspace/zeus/examples/cifar100",
        # fmt: off
        command=[
            "python",
            "train.py",
            "--zeus",
            "--arch", "shufflenet",
            "--batch_size", "{batch_size}",
            "--epochs", "{epochs}",
            "--seed", "{seed}",
        ],
        # fmt: on
    )

    # Generate a list of batch sizes with only power-of-two values.
    batch_sizes = [args.b_min]
    while (bs := batch_sizes[-1] * 2) <= args.b_max:
        batch_sizes.append(bs)

    # Create a designated log directory inside `args.log_base` just for this run of Zeus.
    # Six types of outputs are generated.
    # 1. Power monitor ouptut (`bs{batch_size}+e{epoch_num}+gpu{device_id}.power.log`):
    #      Raw output of the Zeus power monitor.
    # 2. Profiling results (`bs{batch_size}.power.json`):
    #      Train-time average power consumption and throughput for each power limit,
    #      the optimal power limit determined from the result of profiling, and
    #      eval-time average power consumption and throughput for the optimal power limit.
    # 3. Training script output (`rec{recurrence_num}+try{trial_num}.train.log`):
    #      The raw output of the training script. `trial_num` exists because the job
    #      may be early stopped and re-run with another batch size.
    # 4. Training result (`rec{recurrence_num}+try{trial_num}+bs{batch_size}.train.json`):
    #      The total energy, time, and cost consumed, and the number of epochs trained
    #      until the job terminated. Also, whether the job reached the target metric at the
    #      time of termination. Early-stopped jobs will not have reached their target metric.
    # 5. ZeusMaster output (`master.log`): Output from ZeusMaster, including MAB outputs.
    # 6. Job history (`history.py`):
    #      A python file holding a list of `HistoryEntry` objects. Intended use is:
    #      `history = eval(open("history.py").read())` after importing `HistoryEntry`.
    master_logdir = master.build_logdir(
        job=job,
        num_recurrence=args.num_recurrence,
        eta_knob=args.eta_knob,
        beta_knob=args.beta_knob,
        exist_ok=False,  # Should err if this directory exists.
    )

    # Overwrite the stdout file descriptor with an instance of `FileAndConsole`, so that
    # all calls to `print` will write to both the console and the master log file.
    sys.stdout = FileAndConsole(Path(master_logdir) / "master.log")

    # Run Zeus!
    master.run(
        job=job,
        num_recurrence=args.num_recurrence,
        batch_sizes=batch_sizes,
        beta_knob=args.beta_knob,
        eta_knob=args.eta_knob,
    )


if __name__ == "__main__":
    main(parse_args())
