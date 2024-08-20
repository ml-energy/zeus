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

"""Example script of running EnvPipe frequency assignment."""

from __future__ import annotations

import time
import itertools
import logging
from pathlib import Path
from typing import Literal, Type
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
from lowtime.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    forward_dep,
    backward_dep,
    DependencyResolver,
)
from lowtime.graph_utils import add_source_node, add_sink_node, aoa_to_critical_dag, aon_dag_to_aoa_dag
from lowtime.perseus.schedule import Synchronous1F1B
from lowtime.perseus.visualizer import PipelineVisualizer, ANNOTATE_ARGS, LINE_ARGS

logger = logging.getLogger()


@dataclass
class Args:
    # Path to instruction profile results
    inst_profile: str
    # GPU type
    gpu_type: Literal["A40", "A100"]
    # Directory to output results
    output_dir: Path
    # Number of microbatchs
    num_mbs: int
    # Number of stages
    num_stages: int
    # The unit of reduction for each iteration, in seconds
    unit_time: float = 0.001
    # Pipeline schedule name
    train_schedule: Literal["1f1b"] = "1f1b"


def main(args: Args) -> None:
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

    # P2P communication blocking power consumption.
    # Raw P2P blocking power consumption, in Watts
    # A40: 70.38 W
    # A100: 83.64 W
    p_p2p = 83.64 if args.gpu_type.lower() == "a100" else 70.38

    # Instruction offline profiling results.
    inst_df = pd.read_csv(args.inst_profile)

    # stage_id -> instruction type -> OperationSpec
    op_spec_map: dict[int, dict[Type[Instruction], OperationSpec]] = defaultdict(dict)
    # stage_id -> instruction type -> frequency -> real_time
    freq_option_map: dict[int, dict[Type[Instruction], dict[int, ExecutionOption]]] = defaultdict(
        lambda: defaultdict(dict)
    )
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
                freq_option_map[stage_id][instruction][int(row["frequency"])] = options[-1]

            # Get the Preto frontier, quantize time, and deduplicate time.
            cand_options = CandidateExecutionOptions[int](options=options)

            # Initialize the operation spec.
            op_spec = OperationSpec[int](options=cand_options, cost_model=None)
            print(op_spec.options)
            op_spec_map[stage_id][instruction] = op_spec

    ####################
    # DAG construction #
    ####################
    dag = nx.DiGraph()

    # Generate and add all instructions to the DAG.
    # Reserve 0 for dummy source and 1 for dummy sink.
    node_id = 2
    instructions: list[list[Instruction]] = []
    for stage_id in range(args.num_stages):
        # Generate instructions for each stage.
        stage_insts: list[Instruction] = []
        stage_node_ids: list[int] = []
        schedule = Synchronous1F1B(
            num_stages=args.num_stages,
            num_micro_batches=args.num_mbs,
            stage_id=stage_id,
            operation_spec_map=op_spec_map[stage_id],
        )
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

    ########################
    # Frequency assignment #
    ########################
    def is_outer_envelope(aoa_dag: nx.DiGraph, path: list[tuple[int, int]]) -> bool:
        for u, v in path:
            inst: Instruction = aoa_dag[u][v]["op"]
            if inst.is_dummy:
                continue
            if inst.stage_id == args.num_stages - 1:
                continue
            if inst.micro_batch_id == 0 and isinstance(inst, Forward):
                continue
            if inst.micro_batch_id == args.num_mbs - 1 and isinstance(inst, Backward):
                continue
            return False
        return True

    # The outer envelope always runs with the maximum frequency.
    max_freq: int = inst_df.frequency.max().item()
    aoa_dag = aon_dag_to_aoa_dag(dag, attr_name="op")
    for _, _, edge_attrs in aoa_dag.edges(data=True):
        inst: Instruction = edge_attrs["op"]
        if inst.is_dummy:
            continue
        # Every microbatch on the last stage.
        if inst.stage_id == args.num_stages - 1:
            inst.duration = freq_option_map[inst.stage_id][type(inst)][max_freq].quant_time
            inst.assigned_knob = max_freq
            continue
        # The first forward on each stage.
        if inst.micro_batch_id == 0 and isinstance(inst, Forward):
            inst.duration = freq_option_map[inst.stage_id][type(inst)][max_freq].quant_time
            inst.assigned_knob = max_freq
            continue
        # The last backward on each stage.
        if inst.micro_batch_id == args.num_mbs - 1 and isinstance(inst, Backward):
            inst.duration = freq_option_map[inst.stage_id][type(inst)][max_freq].quant_time
            inst.assigned_knob = max_freq
            continue

    frequency_delta = 15
    max_real_time = None
    reached_outer_envelope = False
    max_iters = 10000
    iteration = 0
    for iteration in range(max_iters):
        logger.info("Iteration %d", iteration)

        # Find the critical paths.
        critical_dag = aoa_to_critical_dag(aoa_dag)
        path = next(nx.all_simple_edge_paths(
            critical_dag,
            critical_dag.graph["source_node"],
            critical_dag.graph["sink_node"],
        ))

        # Keep track of max_real_time
        iter_time = 0.0
        for u, v in path:
            inst: Instruction = aoa_dag[u][v]["op"]
            if inst.is_dummy:
                continue
            iter_time += freq_option_map[inst.stage_id][type(inst)][inst.assigned_knob].real_time
        if max_real_time is None:
            max_real_time = int(iter_time) + 1

        # Reconfigure frequency across one critical path.
        # First, find the minimum frequency along the path.
        min_freq = max_freq
        for u, v in path:
            inst: Instruction = aoa_dag[u][v]["op"]
            if inst.is_dummy:
                continue
            if inst.assigned_knob < min_freq:
                min_freq = inst.assigned_knob

        if min_freq == max_freq:
            logger.info("Minimum frequency along one critical path is the max frequency (%d).", max_freq)
            logger.info("Not going to terminate anyway.")

        # Then, increase the frequency of instructions that run with the minimum frequency.
        for u, v in path:
            inst: Instruction = aoa_dag[u][v]["op"]
            if inst.is_dummy:
                continue
            if inst.assigned_knob == min_freq:
                if inst.assigned_knob + frequency_delta > max_freq:
                    logger.info("%s cannot be sped up further", inst)
                else:
                    prev_inst_str = str(inst)
                    new_freq = inst.assigned_knob + frequency_delta
                    inst.duration = (
                        freq_option_map[inst.stage_id][type(inst)][new_freq].quant_time
                    )
                    inst.assigned_knob = new_freq
                    logger.info("Sped up %s to %s", prev_inst_str, inst)

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

        # Print out cost.
        cost = 0.0
        for stage_id, stage_insts in enumerate(instructions):
            stage_computation_time = 0.0
            stage_computation_cost = 0.0
            for inst in stage_insts:
                freq = inst.assigned_knob
                stage_computation_time += freq_option_map[stage_id][type(inst)][freq].real_time
                stage_computation_cost += freq_option_map[stage_id][type(inst)][freq].cost
            cost += stage_computation_cost + 8 * (iter_time - stage_computation_time) * p_p2p
        logger.info("Iteration time: %f", iter_time)
        logger.info("Cost: %f", cost)

        # Terminate if the outer envelope is part of the critical DAG.
        if is_outer_envelope(aoa_dag, path):
            logger.info("Outer envelope reached. Terminating loop.")
            reached_outer_envelope = True
            break

    # Loop terminated.
    if reached_outer_envelope:
        logger.info("Outer envelope reached after %d iterations.", iteration)
    else:
        logger.info("Iteration limit (%d) reached without reaching the outer envelope.", max_iters)

    # Make sure that the outer envelope is still running with the max frequency.
    for _, _, edge_attrs in aoa_dag.edges(data=True):
        inst: Instruction = edge_attrs["op"]
        if inst.is_dummy:
            continue
        # Every microbatch on the last stage.
        if inst.stage_id == args.num_stages - 1:
            if inst.assigned_knob != max_freq:
                logger.info("Forcing %s to run with max frequency", inst)
                inst.assigned_knob = max_freq
            continue
        # The first forward on each stage.
        if inst.micro_batch_id == 0 and isinstance(inst, Forward):
            if inst.assigned_knob != max_freq:
                logger.info("Forcing %s to run with max frequency", inst)
                inst.assigned_knob = max_freq
            continue
        # The last backward on each stage.
        if inst.micro_batch_id == args.num_mbs - 1 and isinstance(inst, Backward):
            if inst.assigned_knob != max_freq:
                logger.info("Forcing %s to run with max frequency", inst)
                inst.assigned_knob = max_freq
            continue

    # assert max_real_time is not None
    # draw(dag, iteration + 1, max_real_time)

    # Write the frequency assignment Python file.
    freqs: list[list[int]] = []
    for stage_id, stage_insts in enumerate(instructions):
        stage_freq = []
        for inst in stage_insts:
            stage_freq.append(inst.assigned_knob)
        freqs.append(stage_freq)

    # Don't flush since IO can overlap with the solver.
    f = open(args.output_dir / f"freqs_pipeline_{iteration + 1:05d}.py", "w")
    f.write("[\n")
    for stage_freq in freqs:
        f.write(f"{stage_freq},\n")
    f.write("]\n")


if __name__ == "__main__":
    args = tyro.cli(Args)

    start_time = time.time()
    main(args)
    logger.info("Total time: %.2fs", time.time() - start_time)
