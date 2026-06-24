# nashville-sim

> Episode 1 of building a budget-constrained resource-allocation engine in public.

A small reinforcement-learning environment + PPO trainer. The agent plays a
Nashville city planner spending a yearly capital budget across growth corridors
(zoning, transit, safety, complete streets) over a 10-year horizon, trying to
relieve housing/jobs pressure, grow transit access, and cut crash risk while
keeping growth spread across corridors instead of piling into one.

## Results so far

A PPO agent beats a hand-written greedy planner by a wide margin, and many
different hyperparameter settings converge to the same score — a real environment
ceiling.

| policy | mean return |
|---|---|
| do-nothing (no_op) | -0.99 |
| random | -0.88 |
| greedy heuristic | -0.62 |
| **PPO champion** | **-0.31** |

See `runs/sweep/dashboard.html` for the full sweep dashboard (open it in a
browser).

## Status / history

The original source was found **deleted** — only `env.py`'s compiled bytecode and
a virtualenv survived, and the folder was never committed to git. The environment
was **reconstructed from that bytecode**: field names, the 5-action set,
observation shape, and the recovered numeric constants match the original; the
surrounding dynamics were rebuilt to be internally consistent and learnable. The
trainer, sweep, and seed corridor data are new (they were also missing). This repo
exists so it never gets lost again.

## Layout

```
nashville_sim/env.py     # NashvilleGrowthEnv + Corridor + loaders
data/seed_corridors.json # 8 real Nashville corridors, normalized [0,1] values
train.py                 # self-contained PPO (PyTorch, GPU), one run
sweep.py                 # time-budgeted random hyperparameter search
make_dashboard.py        # renders runs/sweep/ into a static dashboard.html
test_env.py              # env sanity check + baseline returns
```

## Env design

- **Observation** (`2 + 6*N` floats, N=corridors): years_remaining, budget_pressure,
  then per corridor [housing_pressure, jobs_pressure, transit_access, crash_risk,
  density, political_friction], all in [0,1].
- **Action** (`N*5` discrete): `corridor_idx * 5 + action_type`, where action_type is
  {no_op, upzone, transit_priority, safety_work, complete_street}. Each costs budget
  scaled by the corridor's political friction.
- **Reward**: 1.5 x year-over-year change in a weighted livability score
  (housing_relief, jobs_relief, transit, safety, minus growth_concentration and
  over-budget), minus a penalty if over budget.
- **Budget**: `annual_refill=True` (default) gives `annual_budget` each year, unspent
  budget accumulates up to `max_budget_multiple`. Set `annual_refill=False` for the
  strict single-pool model recovered from the bytecode.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                # torch + numpy

python test_env.py                    # sanity + baselines
python train.py --total-steps 800000  # one PPO run -> runs/<name>/
python sweep.py --budget-hours 8       # overnight hyperparameter sweep
python make_dashboard.py               # render runs/sweep/dashboard.html
```

The trainer puts the policy on the GPU (CUDA); the env itself is pure-Python and
CPU-bound, so GPU draw is light. A numpy-vectorized env is the obvious next
throughput win.

## License

MIT. See [LICENSE](LICENSE).
