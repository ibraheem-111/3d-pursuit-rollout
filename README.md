# Can Learned Signaling Approximate Multi-Agent Rollout in 3D fully observable rollout

# Todo
Fix creating gif leads to out of memory
Make the planner work with multiple agents 
Integrate the planner based simulation with the visualization code

# Project Setup

Install UV
and do ``` uv sync ```

then ``` uv run main.py --config config.yaml ```

## Planner Behavior Notes

### Are all pursuers using infinite-horizon rollout when `--horizon 0`?

Yes for scoring, with an important nuance.

- When `--horizon 0` is passed, the base evaluator uses an infinite-horizon value rollout approximation (it keeps simulating until capture or discounted tail is negligible).
- Every pursuer candidate action is scored through that same evaluator.
- The planner is still **nonautonomous one-agent-at-a-time rollout improvement**, not full joint optimization in one solve.

So each pursuer participates in infinite-horizon scoring, but decisions are improved sequentially (`P0`, then `P1`, then `P2`, ...), which can create order bias.

### Why `P0` can appear to dominate captures

- Pursuer improvements are done in index order.
- Earlier chosen moves are fixed when later pursuers are optimized.
- During execution, if a pursuer captures the evader, the per-step move loop breaks immediately.

This can make `P0` seem strongest even when later pursuers are also being evaluated.

## Planner Flowchart

```mermaid
flowchart TD
	A[CLI args + config] --> B{Planner mode?}
	B -- no --> C[Baseline simulation path]
	B -- yes --> D[planner_run_simulation]

	D --> E[Read horizon and discount_factor]
	E --> F[Build GridModel + BasePolicyEvaluator + NonAutonomousRolloutPlanner]
	F --> G[Create and place evader/pursuers]
	G --> H{capture or max steps?}

	H -- no --> I[Build current GameState]
	I --> J[improve_joint_action]

	J --> K[For each pursuer i in order]
	K --> L[Enumerate valid candidate moves for i]
	L --> M[Assemble provisional joint action]
	M --> N[Compute Q = stage_cost + alpha * cost_to_go]

	N --> O{horizon == 0?}
	O -- yes --> P[Infinite-horizon evaluator loop\nuntil capture or discounted-tail tolerance]
	O -- no --> Q[Finite-horizon evaluator loop\nfor rollout_horizon steps]

	P --> R[Select best move for pursuer i]
	Q --> R
	R --> S[After all i, return joint pursuer move]

	S --> T[Sample evader move]
	T --> U[Apply pursuer moves in order]
	U --> V{capture occurred this step?}
	V -- yes --> W[Break pursuer move loop]
	V -- no --> X[Continue pursuer loop]
	X --> Y[Record snapshots + positions]
	W --> Y
	Y --> H

	H -- yes --> Z[Return result dict\nsnapshots, positions, grid_size, time_steps, capture flags]
```

# Dir structure

Inventory Refreshed: 2026-04-12T15:57:49-0700
Git Baseline: 33b3741 (2026-04-12T03:52:04-07:00)

This project studies 3D pursuit-evasion rollouts on configurable grids, focusing on evader movement and slice-based visualization.
It provides a CLI-driven research workflow to simulate runs, export per-timestep snapshots, and generate GIF artifacts for analysis.

## Directory Structure

```text
3d-pursuit-rollout/
├── .github/
│   └── copilot-instructions.md
├── .gitignore
├── .python-version
├── 3d-pursuit.ipynb
├── cli_slice_smoke.png
├── config.yaml
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
└── src/
	├── __init__.py
	├── agents/
	│   ├── __init__.py
	│   ├── base.py
	│   ├── evader.py
	│   ├── factory.py
	│   └── pursuer.py
	├── data_types/
	│   ├── __init__.py
	│   └── postion.py
	├── grid.py
	├── simulation/
	│   ├── __init__.py
	│   └── simulation.py
	├── utils/
	│   ├── __init__.py
	│   └── constants.py
	└── visualization/
		├── __init__.py
		└── slices.py
```

## File Summaries (3 words each)

| File | Summary |
| --- | --- |
| 3d-pursuit.ipynb | Notebook simulation experiments |
| cli_slice_smoke.png | Sample slice render |
| config.yaml | Simulation default configuration |
| .github/copilot-instructions.md | Repository coding guidelines |
| .gitignore | Ignore generated artifacts |
| main.py | Entrypoint run orchestration |
| pyproject.toml | Project dependency metadata |
| .python-version | Pinned python version |
| README.md | Project overview document |
| src/agents/base.py | Agent base abstraction |
| src/agents/evader.py | Evader policy implementations |
| src/agents/factory.py | Agent factory mapping |
| src/agents/__init__.py | Agents package marker |
| src/agents/pursuer.py | Pursuer policy placeholder |
| src/data_types/__init__.py | Data types package |
| src/data_types/postion.py | Position pydantic model |
| src/grid.py | Three dimensional grid |
| src/__init__.py | Source package marker |
| src/simulation/__init__.py | Simulation package marker |
| src/simulation/simulation.py | Simulation rollout loop |
| src/utils/constants.py | Constants placeholder module |
| src/utils/__init__.py | Utilities package marker |
| src/visualization/__init__.py | Visualization package exports |
| src/visualization/slices.py | Slice plotting utilities |
| uv.lock | Resolved dependency lockfile |
