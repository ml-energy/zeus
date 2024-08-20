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

"""A time-cost trade-off solver based on the Phillips-Dessouky algorithm."""


from __future__ import annotations

import sys
import logging
from collections import deque
from collections.abc import Generator

import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from attrs import define, field

from lowtime.operation import Operation
from lowtime.graph_utils import (
    aon_dag_to_aoa_dag,
    aoa_to_critical_dag,
    get_critical_aoa_dag_total_time,
    get_total_cost,
)
from lowtime.exceptions import LowtimeFlowError

FP_ERROR = 1e-6

logger = logging.getLogger(__name__)


@define
class IterationResult:
    """Holds results after one PD iteration.

    Attributes:
        iteration: The number of optimization iterations experienced by the DAG.
        cost_change: The increase in cost from reducing the DAG's
            quantized execution time by 1.
        cost: The current total cost of the DAG.
        quant_time: The current quantized total execution time of the DAG.
        unit_time: The unit time used for time quantization.
        real_time: The current real (de-quantized) total execution time of the DAG.
    """

    iteration: int
    cost_change: float
    cost: float
    quant_time: int
    unit_time: float
    real_time: float = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Set `real_time` after initialization."""
        self.real_time = self.quant_time * self.unit_time


class PhillipsDessouky:
    """Implements the Phillips-Dessouky algorithm for the time-cost tradeoff problem."""

    def __init__(self, dag: nx.DiGraph) -> None:
        """Initialize the Phillips-Dessouky solver.

        Assumptions:
            - The graph is a Directed Acyclic Graph (DAG).
            - Computation operations are annotated on the node of the DAG.
            - There is only one source (entry) node. The source node is annotated as
                `dag.graph["source_node"]`.
            - There is only one sink (exit) node. The sink node is annotated as
                `dag.graph["sink_node"]`.
            - The `unit_time` attribute of every operation is the same.

        Args:
            dag: A networkx DiGraph object that represents the computation DAG.
                The aforementioned assumptions should hold for the DAG.
        """
        # Run checks on the DAG and cache some properties.
        # Check: It's a DAG.
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("The graph should be a Directed Acyclic Graph.")

        # Check: Only one source node that matches annotation.
        if (source_node := dag.graph.get("source_node")) is None:
            raise ValueError("The graph should have a `source_node` attribute.")
        source_node_candidates = []
        for node_id, in_degree in dag.in_degree():
            if in_degree == 0:
                source_node_candidates.append(node_id)
        if len(source_node_candidates) == 0:
            raise ValueError(
                "Found zero nodes with in-degree 0. Cannot determine source node."
            )
        if len(source_node_candidates) > 1:
            raise ValueError(
                f"Expecting only one source node, found {source_node_candidates}."
            )
        if (detected_source_node := source_node_candidates[0]) != source_node:
            raise ValueError(
                f"Detected source node ({detected_source_node}) does not match "
                f"the annotated source node ({source_node})."
            )

        # Check: Only one sink node that matches annotation.
        if (sink_node := dag.graph.get("sink_node")) is None:
            raise ValueError("The graph should have a `sink_node` attribute.")
        sink_node_candidates = []
        for node_id, out_degree in dag.out_degree():
            if out_degree == 0:
                sink_node_candidates.append(node_id)
        if len(sink_node_candidates) == 0:
            raise ValueError(
                "Found zero nodes with out-degree 0. Cannot determine sink node."
            )
        if len(sink_node_candidates) > 1:
            raise ValueError(
                f"Expecting only one sink node, found {sink_node_candidates}."
            )
        if (detected_sink_node := sink_node_candidates[0]) != sink_node:
            raise ValueError(
                f"Detected sink node ({detected_sink_node}) does not match "
                f"the annotated sink node ({sink_node})."
            )

        # Check: The `unit_time` attributes of every operation should be the same.
        unit_time_candidates = set[float]()
        for _, node_attr in dag.nodes(data=True):
            if "op" in node_attr:
                op: Operation = node_attr["op"]
                if op.is_dummy:
                    continue
                unit_time_candidates.update(
                    option.unit_time for option in op.spec.options.options
                )
        if len(unit_time_candidates) == 0:
            raise ValueError("Found zero operations in the graph.")
        if len(unit_time_candidates) > 1:
            raise ValueError(
                f"Expecting the same `unit_time` across all operations, "
                f"found {unit_time_candidates}."
            )

        self.aon_dag = dag
        self.unit_time = unit_time_candidates.pop()

    def run(self) -> Generator[IterationResult, None, None]:
        """Run the algorithm and yield a DAG after each iteration.

        The solver will not deepcopy operations on the DAG but rather in-place modify
        them for speed. The caller should deepcopy the operations or the DAG if needed
        before running the next iteration.

        Upon yield, it is guaranteed that the earliest/latest start/finish time values
        of all operations are up to date w.r.t. the `duration` of each operation.
        """
        logger.info("Starting Phillips-Dessouky solver.")

        # Convert the original activity-on-node DAG to activity-on-arc DAG form.
        # AOA DAGs are purely internal. All public input and output of this class
        # should be in AON form.
        aoa_dag = aon_dag_to_aoa_dag(self.aon_dag, attr_name="op")

        # Estimate the minimum execution time of the DAG by setting every operation
        # to run at its minimum duration.
        for _, _, edge_attr in aoa_dag.edges(data=True):
            op: Operation = edge_attr["op"]
            if op.is_dummy:
                continue
            op.duration = op.min_duration
        critical_dag = aoa_to_critical_dag(aoa_dag)
        min_time = get_critical_aoa_dag_total_time(critical_dag)
        logger.info("Expected minimum quantized execution time: %d", min_time)

        # Estimated the maximum execution time of the DAG by setting every operation
        # to run at its maximum duration. This is also our initial start point.
        for _, _, edge_attr in aoa_dag.edges(data=True):
            op: Operation = edge_attr["op"]
            if op.is_dummy:
                continue
            op.duration = op.max_duration
        critical_dag = aoa_to_critical_dag(aoa_dag)
        max_time = get_critical_aoa_dag_total_time(critical_dag)
        logger.info("Expected maximum quantized execution time: %d", max_time)

        num_iters = max_time - min_time + 1
        logger.info("Expected number of PD iterations: %d", num_iters)

        # Iteratively reduce the execution time of the DAG.
        for iteration in range(sys.maxsize):
            logger.info(">>> Beginning iteration %d/%d", iteration + 1, num_iters)

            # At this point, `critical_dag` always exists and is what we want.
            # For the first iteration, the critical DAG is computed before the for
            # loop in order to estimate the number of iterations. For subsequent
            # iterations, the critcal DAG is computed after each iteration in
            # in order to construct `IterationResult`.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Critical DAG:")
                logger.debug("Number of nodes: %d", critical_dag.number_of_nodes())
                logger.debug("Number of edges: %d", critical_dag.number_of_edges())
                non_dummy_ops = [
                    attr["op"]
                    for _, _, attr in critical_dag.edges(data=True)
                    if not attr["op"].is_dummy
                ]
                logger.debug("Number of non-dummy operations: %d", len(non_dummy_ops))
                logger.debug(
                    "Sum of non-dummy durations: %d",
                    sum(op.duration for op in non_dummy_ops),
                )

            self.annotate_capacities(critical_dag)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Capacity DAG:")
                logger.debug(
                    "Total lb value: %f",
                    sum([critical_dag[u][v]["lb"] for u, v in critical_dag.edges]),
                )
                logger.debug(
                    "Total ub value: %f",
                    sum([critical_dag[u][v]["ub"] for u, v in critical_dag.edges]),
                )

            try:
                s_set, t_set = self.find_min_cut(critical_dag)
            except LowtimeFlowError as e:
                logger.info("Could not find minimum cut: %s", e.message)
                logger.info("Terminating PD iteration.")
                break

            cost_change = self.reduce_durations(critical_dag, s_set, t_set)
            if cost_change == float("inf") or abs(cost_change) < FP_ERROR:
                logger.info("No further time reduction possible.")
                logger.info("Terminating PD iteration.")
                break

            # Earliest/latest start/finish times on operations also annotated here.
            critical_dag = aoa_to_critical_dag(aoa_dag)

            # We directly modify operation attributes in the DAG, so after we
            # ran one iteration, the AON DAG holds updated attributes.
            result = IterationResult(
                iteration=iteration + 1,
                cost_change=cost_change,
                cost=get_total_cost(aoa_dag, mode="edge"),
                quant_time=get_critical_aoa_dag_total_time(critical_dag),
                unit_time=self.unit_time,
            )
            logger.info("%s", result)
            yield result

    def reduce_durations(
        self, dag: nx.DiGraph, s_set: set[int], t_set: set[int]
    ) -> float:
        """Modify operation durations to reduce the DAG's execution time by 1."""
        speed_up_edges: list[Operation] = []
        for node_id in s_set:
            for child_id in list(dag.successors(node_id)):
                if child_id in t_set:
                    op: Operation = dag[node_id][child_id]["op"]
                    speed_up_edges.append(op)

        slow_down_edges: list[Operation] = []
        for node_id in t_set:
            for child_id in list(dag.successors(node_id)):
                if child_id in s_set:
                    op: Operation = dag[node_id][child_id]["op"]
                    slow_down_edges.append(op)

        if not speed_up_edges:
            logger.info("No speed up candidate operations.")
            return 0.0

        cost_change = 0.0

        # Reduce the duration of edges (speed up) by quant_time 1.
        for op in speed_up_edges:
            if op.is_dummy:
                logger.info("Cannot speed up dummy operation.")
                return float("inf")
            if op.duration - 1 < op.min_duration:
                logger.info("Operation %s has reached the limit of speed up", op)
                return float("inf")
            cost_change += abs(op.get_cost(op.duration - 1) - op.get_cost(op.duration))
            op_before_str = str(op)
            op.duration -= 1
            logger.info("Sped up %s to %s", op_before_str, op)

        # Increase the duration of edges (slow down) by quant_time 1.
        for op in slow_down_edges:
            # Dummy edges can always be slowed down.
            if op.is_dummy:
                logger.info("Slowed down DummyOperation (didn't really slowdown).")
                # XXX(JW): It seems like we should not actually slow down a dummy
                # operation! This caused a bug in wide-resnet
                # (`results/wide-resnet-dummy` vs `results/wide-resnet-gif-2`).
                # logger.info("Slowed down %s to %d", op, op.duration + 1)
                # op.duration += 1
                continue
            elif op.duration + 1 > op.max_duration:
                logger.info("Operation %s has reached the limit of slow down", op)
                return float("inf")
            cost_change -= abs(op.get_cost(op.duration) - op.get_cost(op.duration + 1))
            before_op_str = str(op)
            op.duration += 1
            logger.info("Slowed down %s to %s", before_op_str, op)

        return cost_change

    def find_min_cut(self, dag: nx.DiGraph) -> tuple[set[int], set[int]]:
        """Find the min cut of the DAG annotated with lower/upper bound flow capacities.

        Assumptions:
            - The capacity DAG is in AOA form.
            - The capacity DAG has been annotated with `lb` and `ub` attributes on edges,
                representing the lower and upper bounds of the flow on the edge.

        Returns:
            A tuple of (s_set, t_set) where s_set is the set of nodes on the source side
            of the min cut and t_set is the set of nodes on the sink side of the min cut.
            Returns None if no feasible flow exists.

        Raises:
            LowtimeFlowError: When no feasible flow exists.
        """
        source_node = dag.graph["source_node"]
        sink_node = dag.graph["sink_node"]

        # In order to solve max flow on edges with both lower and upper bounds,
        # we first need to convert it to another DAG that only has upper bounds.
        unbound_dag = nx.DiGraph(dag)

        # For every edge, capacity = ub - lb.
        for _, _, edge_attrs in unbound_dag.edges(data=True):
            edge_attrs["capacity"] = edge_attrs["ub"] - edge_attrs["lb"]

        # Add a new node s', which will become the new source node.
        # We constructed the AOA DAG, so we know that node IDs are integers.
        node_ids: list[int] = list(unbound_dag.nodes)
        s_prime_id = max(node_ids) + 1
        unbound_dag.add_node(s_prime_id)

        # For every node u in the original graph, add an edge (s', u) with capacity
        # equal to the sum of all lower bounds of u's parents.
        for u in dag.nodes:
            capacity = 0.0
            for pred_id in dag.predecessors(u):
                capacity += dag[pred_id][u]["lb"]
            unbound_dag.add_edge(s_prime_id, u, capacity=capacity)

        # Add a new node t', which will become the new sink node.
        t_prime_id = s_prime_id + 1
        unbound_dag.add_node(t_prime_id)

        # For every node u in the original graph, add an edge (u, t') with capacity
        # equal to the sum of all lower bounds of u's children.
        for u in dag.nodes:
            capacity = 0.0
            for succ_id in dag.successors(u):
                capacity += dag[u][succ_id]["lb"]
            unbound_dag.add_edge(u, t_prime_id, capacity=capacity)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Unbound DAG")
            logger.debug("Number of nodes: %d", unbound_dag.number_of_nodes())
            logger.debug("Number of edges: %d", unbound_dag.number_of_edges())
            logger.debug(
                "Sum of capacities: %f",
                sum(attr["capacity"] for _, _, attr in unbound_dag.edges(data=True)),
            )

        # Add an edge from t to s with infinite capacity.
        unbound_dag.add_edge(
            sink_node,
            source_node,
            capacity=float("inf"),
        )

        # We're done with constructing the DAG with only flow upper bounds.
        # Find the maximum flow on this DAG.
        try:
            _, flow_dict = nx.maximum_flow(
                unbound_dag,
                s_prime_id,
                t_prime_id,
                capacity="capacity",
                flow_func=edmonds_karp,
            )
        except nx.NetworkXUnbounded as e:
            raise LowtimeFlowError("ERROR: Infinite flow for unbounded DAG.") from e

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("After first max flow")
            total_flow = 0.0
            for d in flow_dict.values():
                for flow in d.values():
                    total_flow += flow
            logger.debug("Sum of all flow values: %f", total_flow)

        # Check if residual graph is saturated. If so, we have a feasible flow.
        for u in unbound_dag.successors(s_prime_id):
            if (
                abs(flow_dict[s_prime_id][u] - unbound_dag[s_prime_id][u]["capacity"])
                > FP_ERROR
            ):
                logger.error(
                    "s' -> %s unsaturated (flow: %s, capacity: %s)",
                    u,
                    flow_dict[s_prime_id][u],
                    unbound_dag[s_prime_id][u]["capacity"],
                )
                raise LowtimeFlowError(
                    "ERROR: Max flow on unbounded DAG didn't saturate."
                )
        for u in unbound_dag.predecessors(t_prime_id):
            if (
                abs(flow_dict[u][t_prime_id] - unbound_dag[u][t_prime_id]["capacity"])
                > FP_ERROR
            ):
                logger.error(
                    "%s -> t' unsaturated (flow: %s, capacity: %s)",
                    u,
                    flow_dict[u][t_prime_id],
                    unbound_dag[u][t_prime_id]["capacity"],
                )
                raise LowtimeFlowError(
                    "ERROR: Max flow on unbounded DAG didn't saturate."
                )

        # We have a feasible flow. Construct a new residual graph with the same
        # shape as the capacity DAG so that we can find the min cut.
        # First, retrieve the flow amounts to the original capacity graph, where for
        # each edge u -> v, the flow amount is `flow + lb`.
        for u, v in dag.edges:
            dag[u][v]["flow"] = flow_dict[u][v] + dag[u][v]["lb"]

        # Construct a new residual graph (same shape as capacity DAG) with
        # u -> v capacity `ub - flow` and v -> u capacity `flow - lb`.
        residual_graph = nx.DiGraph(dag)
        for u, v in dag.edges:
            residual_graph[u][v]["capacity"] = (
                residual_graph[u][v]["ub"] - residual_graph[u][v]["flow"]
            )
            capacity = residual_graph[u][v]["flow"] - residual_graph[u][v]["lb"]
            if dag.has_edge(v, u):
                residual_graph[v][u]["capacity"] = capacity
            else:
                residual_graph.add_edge(v, u, capacity=capacity)

        # Run max flow on the new residual graph.
        try:
            _, flow_dict = nx.maximum_flow(
                residual_graph,
                source_node,
                sink_node,
                capacity="capacity",
                flow_func=edmonds_karp,
            )
        except nx.NetworkXUnbounded as e:
            raise LowtimeFlowError(
                "ERROR: Infinite flow on capacity residual graph."
            ) from e

        # Add additional flow we get to the original graph
        for u, v in dag.edges:
            dag[u][v]["flow"] += flow_dict[u][v]
            dag[u][v]["flow"] -= flow_dict[v][u]

        # Construct the new residual graph.
        new_residual = nx.DiGraph(dag)
        for u, v in dag.edges:
            new_residual[u][v]["flow"] = dag[u][v]["ub"] - dag[u][v]["flow"]
            new_residual.add_edge(v, u, flow=dag[u][v]["flow"] - dag[u][v]["lb"])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("New residual graph")
            logger.debug("Number of nodes: %d", new_residual.number_of_nodes())
            logger.debug("Number of edges: %d", new_residual.number_of_edges())
            logger.debug(
                "Sum of flow: %f",
                sum(attr["flow"] for _, _, attr in new_residual.edges(data=True)),
            )

        # Find the s-t cut induced by the second maximum flow above.
        # Only following `flow > 0` edges, find the set of nodes reachable from
        # source node. That's the s-set, and the rest is the t-set.
        s_set = set[int]()
        q: deque[int] = deque()
        q.append(source_node)
        while q:
            cur_id = q.pop()
            s_set.add(cur_id)
            if cur_id == sink_node:
                break
            for child_id in list(new_residual.successors(cur_id)):
                if (
                    child_id not in s_set
                    and abs(new_residual[cur_id][child_id]["flow"]) > FP_ERROR
                ):
                    q.append(child_id)
        t_set = set(new_residual.nodes) - s_set

        return s_set, t_set

    def annotate_capacities(self, critical_dag: nx.DiGraph) -> None:
        """In-place annotate the critical DAG with flow capacities."""
        # XXX(JW): Is this always large enough?
        # It is necessary to monitor the `cost_change` value in `IterationResult`
        # and make sure they are way smaller than this value. Even if the cost
        # change is close or larger than this, users can scale down their cost
        # value in `ExecutionOption`s.
        inf = 10000.0
        for _, _, edge_attr in critical_dag.edges(data=True):
            op: Operation = edge_attr["op"]
            duration = op.duration
            # Dummy operations don't constrain the flow.
            if op.is_dummy:  # noqa: SIM114
                lb, ub = 0.0, inf
            # Cannot be sped up or down.
            elif duration == op.min_duration == op.max_duration:
                lb, ub = 0.0, inf
            # Cannot be sped up.
            elif duration - 1 < op.min_duration:
                lb = abs(op.get_cost(duration) - op.get_cost(duration + 1))
                ub = inf
            # Cannot be slowed down.
            elif duration + 1 > op.max_duration:
                lb = 0.0
                ub = abs(op.get_cost(duration - 1) - op.get_cost(duration))
            else:
                # In case the cost model is almost linear, give this edge some room.
                lb = abs(op.get_cost(duration) - op.get_cost(duration + 1))
                ub = abs(op.get_cost(duration - 1) - op.get_cost(duration)) + FP_ERROR

            # XXX(JW): Why is this rouding needed?
            edge_attr["lb"] = lb // FP_ERROR * FP_ERROR
            edge_attr["ub"] = ub // FP_ERROR * FP_ERROR
