# nashville-sim

> Building a budget-constrained resource-allocation engine in public. One RL
> engine, two skins.

A small reinforcement-learning engine for **multi-objective sequential resource
allocation under a budget**, with a self-contained PPO trainer. The same
`train.py` / `sweep.py` drive two environments via `--env`:

- **`nashville`** (Episode 1): a city planner spends a yearly capital budget across
  growth corridors (zoning, transit, safety, complete streets) over a 10-year
  horizon to improve a livability score.
- **`screening`** (Episode 2): the *same engine* spends a cancer-screening budget
  across Tennessee counties to maximize population-weighted early-stage detection,
  with an equity penalty for leaving the worst-covered counties behind. This is the
  "where should limited healthcare dollars go to do the most good" question.

## Results

### Episode 1 — city corridors (`--env nashville`)

A PPO agent beats a hand-written greedy planner by ~50%, and many different
hyperparameter settings converge to the same score — a real environment ceiling.

| policy | mean return |
|---|---|
| do-nothing (no_op) | -0.99 |
| random | -0.88 |
| greedy | -0.62 |
| **PPO champion** | **-0.31** |

### Episode 2 — Tennessee cancer-screening allocation (`--env screening`)

Here `greedy` is a strong 1-step-lookahead planner ("spend where it helps most
this year"). PPO has to learn multi-year strategy to win — and it does, reaching a
*positive* return (it improves population-weighted early detection over the decade)
while every baseline loses ground. Tuning matters far more than in the city env:
the best config and the median trial are far apart.

| policy | mean return |
|---|---|
| do-nothing (no_op) | -0.50 |
| random | -0.43 |
| greedy (1-step lookahead) | -0.10 |
| **PPO champion** | **+0.15** |

Dashboards (open in a browser, or via the GitHub Pages link):
`runs/sweep/dashboard.html` and `runs/sweep_screening/dashboard.html`.

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
nashville_sim/env.py     # NashvilleGrowthEnv (Episode 1 — city corridors)
screening_sim/env.py     # EarlyDetectionEnv (Episode 2 — TN screening allocation)
data/seed_corridors.json # 8 Nashville corridors, normalized [0,1] values
data/tn_counties.json    # 12 Tennessee counties (illustrative; see data note)
train.py                 # self-contained PPO (PyTorch, GPU); pick env with --env
sweep.py                 # time-budgeted random hyperparameter search (--env)
make_dashboard.py        # renders a sweep into a static dashboard.html (--env)
test_env.py / test_screening.py  # env sanity checks + baseline returns
```

Both envs expose the identical interface (`reset/step/observation_size/
action_size/sample_action/greedy_action`), so the trainer is env-agnostic. The
screening env's `greedy` baseline is a strong 1-step-lookahead planner, so PPO has
to learn multi-year strategy to beat it.

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

python test_env.py                        # city env sanity + baselines
python test_screening.py                  # screening env sanity + baselines
python train.py --total-steps 800000      # train the city env -> runs/<name>/
python train.py --env screening --total-steps 800000   # train the screening env
python sweep.py --env screening --budget-hours 0.3      # sweep the screening env
python make_dashboard.py --env screening  # render runs/sweep_screening/dashboard.html
```

The trainer puts the policy on the GPU (CUDA); the env itself is pure-Python and
CPU-bound, so GPU draw is light. A numpy-vectorized env is the obvious next
throughput win.

## License

MIT. See [LICENSE](LICENSE).
