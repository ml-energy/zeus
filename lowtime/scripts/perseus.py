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

"""Example script of running Perseus energy scheduling."""

from __future__ import annotations

import time
import itertools
import logging
from pathlib import Path
from typing import Literal, Optional, Type
from collections import defaultdict
from dataclasses import dataclass

import tyro
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from lowtime.operation import (
    CandidateExecutionOptions,
    OperationSpec,
    ExecutionOption,
)
from lowtime.cost_model import ExponentialModel
from lowtime.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    forward_dep,
    backward_dep,
    DependencyResolver,
)
from lowtime.solver import PhillipsDessouky
from lowtime.graph_utils import add_source_node, add_sink_node
from lowtime.perseus.schedule import PipelineSchedule, Synchronous1F1B
from lowtime.perseus.visualizer import PipelineVisualizer, ANNOTATE_ARGS, LINE_ARGS

logger = logging.getLogger()


@dataclass
class Args:
    # Path to instruction (forward and backward) profile results
    inst_profile: str
    # GPU type
    gpu_type: Literal["A40", "A100"]
    # Directory to output results (should not already exist)
    output_dir: Path
    # Number of microbatchs
    num_mbs: int
    # Number of pipeline stages
    num_stages: int
    # Interval to output heavy results like plots
    plot_interval: int = 100
    # The unit of reduction for each iteration, in seconds
    unit_time: float = 0.001
    # Noise factor for soft Pareto frontier filtering
    noise_factor: float = 0.95
    # Whether to always use noise factor without trying hard Parefo frontier filtering
    always_use_noise_factor: bool = False
    # Path to exponential model initial parameter guesses
    initial_guess: Optional[str] = None
    # Pipeline schedule
    train_schedule: Literal["1f1b"] = "1f1b"


def main(args: Args) -> None:
    """Perseus time-energy tradeoff optimization."""
    # Validate arguments.
    if args.gpu_type not in ["A40", "A100"]:
        raise ValueError(f"Unknown GPU type {args.gpu_type}")

    # Setup logging and output.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "job.log"

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    logger.info("Arguments: %s", args)

    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)

    # P2P communication blocking power consumption.
    # Raw P2P blocking power consumption, in Watts
    # A40: 70.38 W
    # A100: 83.64 W
    p_p2p = 83.64 if args.gpu_type.lower() == "a100" else 70.38
    logger.info("P2P blocking power consumption: %.2fW", p_p2p)

    initial_guess = (
        eval(open(args.initial_guess).read()) if args.initial_guess else None
    )
    op_spec_map: dict[int, dict[Type[Instruction], OperationSpec]] = defaultdict(dict)
    for instruction in [Forward, Backward]:
        inst_name = instruction.__name__
        for stage_id in range(args.num_stages):
            logger.info("Processing %s stage %d", inst_name, stage_id)
            options = []
            _df = inst_df.query(
                f"stage == {stage_id} and instruction == '{inst_name.lower()}'"
            )
            for _, row in _df.iterrows():
                row = row.to_dict()
                options.append(
                    ExecutionOption[int](
                        real_time=row["time"],
                        unit_time=args.unit_time,
                        cost=row["energy"],
                        knob=int(row["frequency"]),
                    )
                )

            # Get the Preto frontier, quantize time, and deduplicate time.
            if args.always_use_noise_factor:
                cand_options = CandidateExecutionOptions[int](options=options, noise_factor=args.noise_factor)
            else:
                cand_options = CandidateExecutionOptions[int](options=options)
                # If we end up with too few options on the frontier, it's probably because
                # measurement was too noisy. In that case, we allow some noise (e.g., 5%).
                # That is, we remove an option only if there is another option that is
                # better by more than `scale_factor * 100` percent.
                if len(cand_options.options) <= 3:
                    cand_options = CandidateExecutionOptions[int](options=options, noise_factor=args.noise_factor)
            logger.info("Candidate execution options: %s", cand_options)

            # Map the cost to be effective computation energy.
            # Everything from now on is in terms of effective energy.
            for option in cand_options.options:
                option.cost -= p_p2p * option.quant_time * option.unit_time

            # Fit the cost model.
            if initial_guess is not None:
                model = ExponentialModel(
                    cand_options, initial_guess=initial_guess[inst_name][stage_id]
                )
            else:
                model = ExponentialModel(cand_options, search_strategy="best")

            # Draw the cost model.
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            model.draw(ax, cand_options)
            fig.savefig(f"{output_dir}/{inst_name.lower()}_{stage_id}.png")

            # Initialize the operation spec.
            op_spec = OperationSpec[int](options=cand_options, cost_model=model)
            op_spec_map[stage_id][instruction] = op_spec

    ####################
    # DAG construction #
    ####################
    dag = nx.DiGraph()

    # Generate and add all instructions to the DAG.
    # Reserve 0 for dummy source and 1 for dummy sink.
    # XXX(JW): Why should all these have a node ID exposed outside?
    node_id = 2
    instructions: list[list[Instruction]] = []
    for stage_id in range(args.num_stages):
        # Generate instructions for each stage.
        stage_insts: list[Instruction] = []
        stage_node_ids: list[int] = []
        if args.train_schedule == "1f1b":
            schedule: PipelineSchedule = Synchronous1F1B(
                num_stages=args.num_stages,
                num_micro_batches=args.num_mbs,
                stage_id=stage_id,
                operation_spec_map=op_spec_map[stage_id],
            )
        else:
            raise NotImplementedError(f"Unknown train schedule {args.train_schedule}")
        for inst in schedule:
            dag.add_node(node_id, op=inst)
            stage_insts.append(inst)
            stage_node_ids.append(node_id)
            node_id += 1
        instructions.append(stage_insts)

        # Add dependencies between adjacent instructions in the same stage.
        for node_id1, node_id2 in zip(stage_node_ids, stage_node_ids[1:]):
            dag.add_edge(node_id1, node_id2)

    # Add dependencies between dependent pipeline instructions.
    insts = dag.nodes(data=True)
    
    if args.train_schedule == "1f1b":
        resolver = DependencyResolver(
            dependency_rules=[forward_dep, backward_dep],
            node_type=Instruction,
        )
        for (id1, data1), (id2, data2) in itertools.product(insts, insts):
            if resolver.is_dependent(data1["op"], data2["op"]):
                dag.add_edge(id1, id2)

    # Add source and sink nodes.
    add_source_node(dag, 0)
    add_sink_node(dag, 1)
    dag.graph["source_node"] = 0
    dag.graph["sink_node"] = 1

    ###################################
    # Time-cost tradeoff optimization #
    ###################################
    def annotation_hook(inst: Instruction) -> str:
        return f"{type(inst).__name__[0]}\n{inst.micro_batch_id}"

    def draw(dag: nx.DiGraph, iteration: int, xlim: int) -> None:
        ANNOTATE_ARGS[Forward]["fontsize"] = 11.0
        ANNOTATE_ARGS[Backward]["fontsize"] = 11.0
        ANNOTATE_ARGS[Forward]["color"] = "black"
        ANNOTATE_ARGS[Backward]["color"] = "black"
        LINE_ARGS["linewidth"] = 3.0

        fig, ax = plt.subplots(figsize=(args.num_mbs, 4), tight_layout=True)

        vis = PipelineVisualizer(dag)
        vis.draw(
            ax,
            draw_time_axis=True,
            p2p_power=p_p2p,
            annotation_hook=annotation_hook,
            power_color="RdBu_r",
            normalizer_range=(-200, 550),
        )
        vis.draw_critical_path(ax)

        # Fix xlim so that we can visually see the pipeline width shrink.
        ax.set_xlim(0.0, xlim)
        ax.set_title(f"Iteration {iteration:4d}")
        fig.savefig(f"{output_dir}/pipeline_{iteration:05d}.png")
        plt.close(fig)


    solver = PhillipsDessouky(dag)

    # Track the iteration time-energy Pareto frontier.
    iteration_time = []
    iteration_energy = []

    max_real_time = None
    iteration = 0
    for iteration, result in enumerate(solver.run()):
        # Maybe draw the pipeline.
        if iteration % args.plot_interval == 0:
            if max_real_time is None:
                max_real_time = int(result.real_time) + 1
            draw(dag, iteration, max_real_time)

        # Write the frequency assignment Python file.
        freqs: list[list[int]] = []
        for stage_id, stage_insts in enumerate(instructions):
            stage_freq = []
            for inst in stage_insts:
                stage_freq.append(inst.assigned_knob)
            freqs.append(stage_freq)

        # Don't flush since IO can overlap with the solver.
        f = open(args.output_dir / f"freqs_pipeline_{iteration:05d}.py", "w")
        f.write("[\n")
        for stage_freq in freqs:
            f.write(f"{stage_freq},\n")
        f.write("]\n")

        iter_str = f"# Iteration {iteration}: "
        real_cost = result.cost + args.num_stages * result.real_time * p_p2p
        f.write(iter_str + f"cost change {result.cost_change} \n")
        f.write(iter_str + f"total cost {result.cost} \n")
        f.write(iter_str + f"refined cost {real_cost} \n")

        # Record iteration time and energy.
        iteration_time.append(result.real_time)
        iteration_energy.append(real_cost)

    # Dump iteration time and energy.
    df = pd.DataFrame({"time": iteration_time, "energy": iteration_energy})
    df.to_csv(output_dir / "iteration_time_energy.csv", index=False)

    if iteration % args.plot_interval != 0:
        assert max_real_time is not None
        draw(dag, iteration + 1, max_real_time)

if __name__ == "__main__":
    args = tyro.cli(Args)

    start_time = time.time()
    main(args)
    logger.info("Total time: %.2fs", time.time() - start_time)
