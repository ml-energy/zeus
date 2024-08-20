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

"""Draw the pipeline schedule with matplotlib."""

from __future__ import annotations

from typing import Callable, Any, Type

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter

from lowtime.graph_utils import get_critical_aon_dag_total_time
from lowtime.perseus.instruction import (
    Instruction,
    Forward,
    Backward,
    Recomputation,
    ForwardBackward,
)

# The default arguments for matplotlib.patches.Rectangle.
RECTANGLE_ARGS: dict[Type[Instruction], dict[str, Any]] = {
    Forward: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    Backward: dict(facecolor="#9fc887", edgecolor="#000000", linewidth=1.0),
    Recomputation: dict(facecolor="#2a4b89", edgecolor="#000000", linewidth=1.0),
    ForwardBackward: dict(facecolor="#f542a4", edgecolor="#000000", linewidth=1.0),
}

# The default arguments for matplotlib.axes.Axes.annotate.
ANNOTATE_ARGS: dict[Type[Instruction], dict[str, Any]] = {
    Forward: dict(color="#2a5889", fontsize=10.0, ha="center", va="center"),
    Backward: dict(color="#000000", fontsize=10.0, ha="center", va="center"),
    Recomputation: dict(color="#5b2a89", fontsize=10.0, ha="center", va="center"),
    ForwardBackward: dict(color="#f542a4", fontsize=10.0, ha="center", va="center"),
}

# The default arguments for matplotlib.axes.Axes.plot.
LINE_ARGS: dict[str, Any] = dict(color="#00a6ff", linewidth=4.0)


class PipelineVisualizer:
    """Draws a scheduled pipeline schedule with matplotlib."""

    def __init__(
        self,
        dag: nx.DiGraph,
        rectangle_args: dict[Type[Instruction], dict[str, Any]] = RECTANGLE_ARGS,
        annotate_args: dict[Type[Instruction], dict[str, Any]] = ANNOTATE_ARGS,
        line_args: dict[str, Any] = LINE_ARGS,
    ) -> None:
        """Intialize the visualizer with the DAG.

        Assumptions:
            - The DAG represents a single pipeline schedule.
            - The DAG holds computation operations as nodes.
            - There is only one source (entry) node. The source node is annotated as
                `dag.graph["source_node"]`.
            - There is only one sink (exit) node. The sink node is annotated as
                `dag.graph["sink_node"]`.
            - The `unit_time` attribute of every operation is the same.

        Args:
            dag: The DAG to visualize.
            rectangle_args: Arguments to `matplotlib.patches.Rectangle` for Instructions.
            annotate_args: Arguments to `Axes.annotate` to describe Instructions.
            line_args: Arguments to `Axes.plot` to for the critical path.
        """
        self.dag = dag
        self.rectangle_args = rectangle_args
        self.annotate_args = annotate_args
        self.line_args = line_args

        # Flesh out non-dummy instructions from the DAG.
        instructions: list[Instruction] = []
        for _, node_attr in dag.nodes(data=True):
            inst: Instruction = node_attr["op"]
            if inst.is_dummy:
                continue
            if not isinstance(inst, Instruction):
                raise ValueError("Operations on the DAG must be `Instruction`s.")
            instructions.append(inst)
        if not instructions:
            raise ValueError("The DAG must contain at least one non-dummy instruction.")
        self.instructions = instructions

        # Fetch pipeline schedule parameters.
        stage_ids = set[int]()
        for inst in self.instructions:
            stage_ids.add(inst.stage_id)
        stage_id_list = sorted(stage_ids)
        for left, right in zip(stage_id_list, stage_id_list[1:]):
            if left + 1 != right:
                raise ValueError(
                    "The pipeline schedule must have consecutive stage IDs."
                )
        self.num_stages = len(stage_ids)

        # Fetch `unit_time` from nodes.
        self.unit_time = self.instructions[0].spec.options.options[0].unit_time

        # Construct and cache the CriticalDAG.
        self.critical_dag = nx.DiGraph(dag)
        # First, remove all non-critical nodes.
        for node_id, node_attr in dag.nodes(data=True):
            inst: Instruction = node_attr["op"]
            if inst.earliest_finish != inst.latest_finish:
                self.critical_dag.remove_node(node_id)
        # Then, only leave edges that *back-to-back* connect two critical nodes.
        # There may be edges that connect two critical nodes that are not
        # necessarily planned to execute back-to-back.
        for u, v in list(self.critical_dag.edges):
            u_inst: Instruction = dag.nodes[u]["op"]
            v_inst: Instruction = dag.nodes[v]["op"]
            if u_inst.earliest_finish != v_inst.earliest_start:
                self.critical_dag.remove_edge(u, v)
        self.total_real_time = (
            get_critical_aon_dag_total_time(self.critical_dag) * self.unit_time
        )

    def draw(
        self,
        ax: Axes,
        draw_time_axis: bool = False,
        annotation_hook: Callable[[Instruction], str] | None = None,
        power_color: str | None = "Oranges",
        p2p_power: float = 75.5,
        draw_p2p_power: bool = True,
        normalizer_range: tuple[float, float] = (0.0, 400.0),
    ) -> None:
        """Draw the pipeline schedule on the given Axes object.

        Args:
            ax: The Axes object to draw on.
            draw_time_axis: Whether to draw the time axis on the bottom of the plot.
            annotation_hook: A function that takes an `Instruction` and returns a string,
                which is used as annotation inside the instruction Rectangle. If None,
                the Instruction's `assigned_knob` is used.
            power_color: If None, instruction color is determined by the instruction type.
                Otherwise, this should be a matplotlib colormap name, and the color of each
                instruction is determined by its power consumption (= cost/duration).
            p2p_power: The power consumption during P2P communication.
            draw_p2p_power: Whether to draw the P2P power consumption as a background.
            normalizer_range: The range of the power consumption normalizer.
                By default, power consumption is normalized from [0, 400] to [0, 1].
        """
        # Fill in the background (P2P blocking power consumption) as a Rectangle.
        cmap = get_cmap(power_color) if power_color is not None else None
        normalizer = Normalize(*normalizer_range)
        if cmap is not None and draw_p2p_power:
            bg_color = cmap(normalizer(p2p_power))
            background = Rectangle(
                xy=(0, 0),
                width=self.total_real_time,
                height=self.num_stages,
                facecolor=bg_color,
                edgecolor=bg_color,
            )
            ax.add_patch(background)

        # Draw instruction Rectangles.
        for inst in self.instructions:
            # Location: (start time, stage ID)
            # Width: duration
            # Height: 1
            rectangle_args: dict[str, Any] = dict(
                xy=(inst.earliest_start * self.unit_time, inst.stage_id),
                width=inst.duration * self.unit_time,
                height=1.0,
            )
            rectangle_args.update(self.rectangle_args[type(inst)])
            if cmap is not None:
                cost = inst.get_cost() + p2p_power * inst.duration * self.unit_time
                rectangle_args["facecolor"] = cmap(
                    normalizer(cost / (inst.duration * self.unit_time))
                )
            rect = Rectangle(**rectangle_args)
            ax.add_patch(rect)

            # Place annotations inside the instruction rectangle.
            annotation = (
                annotation_hook(inst) if annotation_hook else str(inst.assigned_knob)
            )

            annotation_args: dict[str, Any] = dict(
                text=annotation,
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + 0.5),
            )
            annotation_args.update(self.annotate_args[type(inst)])
            ax.annotate(**annotation_args)

        if draw_time_axis:
            ax.yaxis.set_visible(False)
            ax.grid(visible=False)

            ax.set_xlabel("Time (seconds)")
            ax.xaxis.set_label_coords(0.5, -0.07)
            
            # ax.set_xlim(0.0, self.total_real_time)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            xticks = [float(t * 5) for t in range(int(self.total_real_time) // 5)]
            xticks.append(self.total_real_time)
            if 0.0 not in xticks:
                xticks.insert(0, 0.0)
            ax.set_xticks(xticks)

            for side in ["top", "left", "right"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_bounds(0.0, self.total_real_time)
        else:
            ax.set_axis_off()

        ax.autoscale()
        ax.invert_yaxis()

    def draw_critical_path(self, ax: Axes) -> None:
        """Draw all critical paths of the DAG on the given Axes object.

        Args:
            ax: The Axes object to draw on.
        """
        for u, v in self.critical_dag.edges:
            u_inst: Instruction = self.dag.nodes[u]["op"]
            v_inst: Instruction = self.dag.nodes[v]["op"]
            if u_inst.is_dummy or v_inst.is_dummy:
                continue
            x1 = (u_inst.earliest_start + u_inst.earliest_finish) * self.unit_time / 2
            x2 = (v_inst.earliest_start + v_inst.earliest_finish) * self.unit_time / 2
            ax.plot(
                [x1, x2],
                [u_inst.stage_id + 0.75, v_inst.stage_id + 0.75],
                **self.line_args,
            )
