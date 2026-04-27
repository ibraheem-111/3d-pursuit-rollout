from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.rescue.testbed import GraphSearchProblem, RescueResult, grid_coordinates


AGENT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]


def plot_rescue_trajectory(
    problem: GraphSearchProblem,
    result: RescueResult,
    output_path=None,
    *,
    show_hidden_lost: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    final_row = result.trajectory[-1]
    _draw_rescue_frame(
        ax=ax,
        problem=problem,
        result=result,
        frame_idx=len(result.trajectory) - 1,
        show_hidden_lost=show_hidden_lost,
        draw_paths=True,
    )
    ax.set_title(
        f"{result.strategy}: step {final_row['step']}, "
        f"remaining {final_row['remaining_unfound']}"
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    return fig, ax


def save_rescue_gif(
    problem: GraphSearchProblem,
    result: RescueResult,
    output_path,
    *,
    show_hidden_lost: bool = False,
    interval_ms: int = 500,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame_idx):
        ax.clear()
        row = result.trajectory[frame_idx]
        _draw_rescue_frame(
            ax=ax,
            problem=problem,
            result=result,
            frame_idx=frame_idx,
            show_hidden_lost=show_hidden_lost,
            draw_paths=False,
        )
        ax.set_title(
            f"{result.strategy}: step {row['step']}, "
            f"remaining {row['remaining_unfound']}"
        )
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(result.trajectory),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    fps = max(1, int(round(1000 / max(interval_ms, 1))))
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def _draw_rescue_frame(
    *,
    ax,
    problem: GraphSearchProblem,
    result: RescueResult,
    frame_idx: int,
    show_hidden_lost: bool,
    draw_paths: bool,
):
    _draw_edges(ax, problem)

    row = result.trajectory[frame_idx]
    explored_nodes = set(row["explored_nodes"])
    found_individuals = set(row["found_individuals"])

    _draw_node_status(ax, problem, explored_nodes)
    _draw_lost_individuals(
        ax=ax,
        problem=problem,
        found_individuals=found_individuals,
        show_hidden_lost=show_hidden_lost,
    )
    if draw_paths:
        _draw_agent_paths(ax, problem, result, frame_idx)
    _draw_agents(ax, problem, row["agent_nodes"])
    _draw_legend(ax)
    _format_axes(ax, problem)


def _node_xy(problem: GraphSearchProblem, node):
    if problem.grid_width is None:
        return float(node), 0.0
    x, y = grid_coordinates(int(node), problem.grid_width)
    return float(x), float(y)


def _draw_edges(ax, problem: GraphSearchProblem):
    seen = set()
    for node, neighbors in problem.adjacency.items():
        x0, y0 = _node_xy(problem, node)
        for neighbor in neighbors:
            key = tuple(sorted((node, neighbor)))
            if key in seen:
                continue
            seen.add(key)
            x1, y1 = _node_xy(problem, neighbor)
            ax.plot([x0, x1], [y0, y1], color="#d0d7de", linewidth=0.8, zorder=1)


def _draw_node_status(ax, problem: GraphSearchProblem, explored_nodes):
    unexplored_xs = []
    unexplored_ys = []
    explored_xs = []
    explored_ys = []
    for node in problem.adjacency:
        x, y = _node_xy(problem, node)
        if node in explored_nodes:
            explored_xs.append(x)
            explored_ys.append(y)
        else:
            unexplored_xs.append(x)
            unexplored_ys.append(y)

    if len(unexplored_xs) > 0:
        ax.scatter(
            unexplored_xs,
            unexplored_ys,
            s=36,
            color="#f6f8fa",
            edgecolor="#8c959f",
            linewidth=0.8,
            zorder=2,
        )
    if len(explored_xs) > 0:
        ax.scatter(
            explored_xs,
            explored_ys,
            s=76,
            color="#b6e3ff",
            edgecolor="#0969da",
            linewidth=1.0,
            zorder=3,
        )


def _draw_lost_individuals(ax, problem: GraphSearchProblem, found_individuals, show_hidden_lost):
    for idx, node in enumerate(problem.lost_individual_nodes):
        if problem.target_knowledge == "unknown" and idx not in found_individuals and not show_hidden_lost:
            continue
        x, y = _node_xy(problem, node)
        color = "tab:red" if idx in found_individuals else "#9a6700"
        marker = "*" if idx in found_individuals else "X"
        ax.scatter([x], [y], s=180, marker=marker, color=color, edgecolor="black", linewidth=0.8, zorder=5)


def _draw_agent_paths(ax, problem: GraphSearchProblem, result: RescueResult, frame_idx: int):
    if len(result.trajectory) == 0:
        return
    num_agents = len(result.trajectory[0]["agent_nodes"])
    for agent_idx in range(num_agents):
        xs = []
        ys = []
        for row in result.trajectory[: frame_idx + 1]:
            x, y = _node_xy(problem, row["agent_nodes"][agent_idx])
            xs.append(x)
            ys.append(y)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9, zorder=3)


def _draw_agents(ax, problem: GraphSearchProblem, agent_nodes):
    for agent_idx, node in enumerate(agent_nodes):
        x, y = _node_xy(problem, node)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        ax.scatter([x], [y], s=130, color=color, edgecolor="black", linewidth=0.9, zorder=6)
        ax.text(x, y, str(agent_idx), color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=7)


def _draw_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#f6f8fa",
            markeredgecolor="#8c959f",
            markersize=7,
            label="unexplored",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#b6e3ff",
            markeredgecolor="#0969da",
            markersize=8,
            label="explored",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            markersize=8,
            label="agent",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="tab:red",
            markeredgecolor="black",
            markersize=10,
            label="found",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=8)


def _format_axes(ax, problem: GraphSearchProblem):
    ax.set_aspect("equal", adjustable="box")
    if problem.grid_width is not None and problem.grid_height is not None:
        ax.set_xlim(-0.75, problem.grid_width - 0.25)
        ax.set_ylim(-0.75, problem.grid_height - 0.25)
        ax.set_xticks(range(problem.grid_width))
        ax.set_yticks(range(problem.grid_height))
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
