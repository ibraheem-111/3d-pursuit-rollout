# Can Learned Signaling Approximate Multi-Agent Rollout in 3D fully observable rollout

# Todo
Fix creating gif leads to out of memory
Make the planner work with multiple agents 
Integrate the planner based simulation with the visualization code

# Project Setup

Install UV
and do ``` uv sync ```

then ``` uv run python main.py --config config.yaml ```

Choose the controller with one unified strategy flag:

```bash
uv run python main.py --config config.yaml --strategy greedy
uv run python main.py --config config.yaml --strategy non_autonomous_rollout
uv run python main.py --config config.yaml --strategy autonomous_greedy_signaling
uv run python main.py --config config.yaml --strategy autonomous_learned_signaling --signaling-model models/signaling_kernel.npz
```

Collect an offline kernel-signaling dataset/model from non-autonomous rollout:

```bash
uv run python scripts/collect_signaling_data.py \
  --config config.yaml \
  --output models/signaling_kernel.npz \
  --episodes 25 \
  --seed 0 \
  --k 25 \
  --sigma 5.0
```

## Experimental Table Runs

Use `scripts/run_experiment_table.py` to run repeated experiments for the four table methods:

```bash
uv run python -m compileall -q scripts/run_experiment_table.py
uv run python -m unittest discover -s tests
```

The script evaluates:

```text
greedy
non_autonomous_rollout
autonomous_greedy_signaling
autonomous_learned_signaling
```

It writes one experiment directory under `outputs/experiments/<timestamp>/` containing:

```text
raw_metrics.csv       # one row per run
summary_table.csv     # numeric summary table
summary_table.md      # Markdown table for reports
summary_table.tex     # LaTeX table for the paper
metadata.json         # experiment metadata
config.yaml           # copied config used for the run
```

The summary table columns match the paper table:

```text
Method
Capture Rate
Avg. Capture Time
Avg. Cost
Runtime / Step
```

For `autonomous_learned_signaling`, the runner uses `planner.signaling_model_path` from `config.yaml`. If the model is missing or incompatible with the current grid/features, it automatically collects a fresh kernel model from non-autonomous rollout before running learned-signaling trials. To force a rebuild:

```bash
uv run python scripts/run_experiment_table.py \
  --config config.yaml \
  --runs 50 \
  --base-seed 0 \
  --refresh-signaling-model
```

Useful options:

```bash
--strategies greedy non_autonomous_rollout
--output-dir outputs/experiments/my_run
--signaling-model models/signaling_kernel.npz
--signaling-episodes 50
--capture-time-policy captured_only
--capture-time-policy max_steps_for_failures
```

## Rescue Search Test Bed

The rescue test bed models a team searching a graph for lost individuals. Nodes are locations, edges are feasible moves, agents may move to a neighbor or stay, and each unfound individual adds one unit of cost per time step.

`rescue_config.yaml` uses a larger sparse grid graph by default. `graph.type: "sparse_grid"` starts from a connected corridor backbone and adds only some of the remaining grid edges, so the map looks more like a cave, dense forest, damaged building, or indoor fire scene than a fully traversable lattice. Tune `graph.extra_edge_probability` upward for more open terrain and downward for tighter chokepoints.

Rollout strategies also include `simulation.revisit_penalty`, which adds planning cost when a candidate action sends an agent to an already explored node. This penalty affects non-autonomous and autonomous rollout scoring, but the reported rescue objective remains cumulative unfound-individual cost.

Mathematically, let \(H_t \subseteq V\) be the explored nodes at time \(t\), \(u_t^i\) be agent \(i\)'s proposed next node, and \(\lambda \ge 0\) be `simulation.revisit_penalty`. The rollout-only revisit cost is:

```text
R(s_t, u_t) = lambda * sum_i 1[u_t^i in H_t]
```

The reported rescue objective remains:

```text
J = sum_t gamma^t * (# unfound individuals at time t)
```

but rollout action scoring uses:

```text
Q_rollout(s_t, u_t) =
  (# unfound individuals at time t)
  + R(s_t, u_t)
  + gamma * V_base(s_{t+1})
```

So revisiting explored nodes is discouraged during rollout planning without changing the metric used to compare final search performance.

Run every implemented rescue strategy:

```bash
uv run python scripts/run_rescue_testbed.py --config rescue_config.yaml --strategy all
```

Save static trajectory plots and animations:

```bash
uv run python scripts/run_rescue_testbed.py \
  --config rescue_config.yaml \
  --strategy all \
  --plot \
  --save-gif
```

Collect batch metrics over sampled rescue scenarios:

```bash
uv run python scripts/collect_rescue_data.py \
  --config rescue_config.yaml \
  --output outputs/rescue_metrics.csv \
  --episodes 50 \
  --seed 0 \
  --strategy all
```

Create final-table rescue experiment outputs:

```bash
uv run python scripts/run_rescue_experiment_table.py \
  --config rescue_config.yaml \
  --runs 50 \
  --base-seed 0
```

Available rescue strategies:

```text
greedy
non_autonomous_rollout
autonomous_greedy_signaling
```

For `lost_individuals.knowledge: "unknown"`, greedy moves each agent toward the closest unexplored node, with same-step target reservations to reduce duplicate coverage.

Add multiple evaders by including multiple entries under `evaders:` in the config. Greedy and rollout strategies run until all active evaders are captured or `simulation.time_steps` is reached.

## Planner Behavior Notes

### Are all pursuers using infinite-horizon rollout?

Yes. Planner scoring uses an infinite-horizon discounted value rollout approximation, with an important nuance.

- The base evaluator keeps simulating until capture or until the remaining discounted tail is negligible.
- Every pursuer candidate action is scored through that same evaluator.
- The default planner is still **nonautonomous one-agent-at-a-time rollout improvement**, not full joint optimization in one solve.

So each pursuer participates in infinite-horizon scoring, but decisions are improved sequentially (`P0`, then `P1`, then `P2`, ...), which can create order bias.

### Why `P0` can appear to dominate captures

- Pursuer improvements are done in index order.
- Earlier chosen moves are fixed when later pursuers are optimized.
- During execution, if a pursuer captures the evader, the per-step move loop breaks immediately.

This can make `P0` seem strongest even when later pursuers are also being evaluated.

## Strategy Flowchart

```mermaid
flowchart TD
	A[CLI args + config] --> B{strategy}
	B -- greedy --> C[Greedy pursuer simulation path]
	B -- non_autonomous_rollout --> D[planner_run_simulation]
	B -- autonomous_greedy_signaling --> D
	B -- autonomous_learned_signaling --> D

	D --> E[Read strategy and discount_factor]
	E --> F[Build GridModel + BasePolicyEvaluator + selected rollout planner]
	F --> G[Create and place evader/pursuers]
	G --> H{capture or max steps?}

	H -- no --> I[Build current GameState]
	I --> J[improve_joint_action]

	J --> K[For each pursuer i in order]
	K --> L[Enumerate valid candidate moves for i]
	L --> M[Assemble provisional joint action]
	M --> N[Compute Q = stage_cost + alpha * cost_to_go]

	N --> P[Infinite-horizon evaluator loop\nuntil capture or discounted-tail tolerance]

	P --> R[Select best move for pursuer i]
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
