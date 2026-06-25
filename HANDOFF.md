# Handoff — nashville-sim (RL allocation engine, building in public)

For the next agent (Grok) picking this up. Read this top to bottom once, then
skim `README.md` for commands and `GOLDMINE_research.md` (gitignored, local only)
for the strategy. This doc is the "why + how + gotchas + roadmap" that isn't in
the README.

## What this is, in one breath

One small RL engine for **multi-objective sequential resource allocation under a
budget**, with three interchangeable environments (skins) and a self-contained
PyTorch PPO trainer. It is built **in public** as a series of episodes, each one a
GPU experiment + an open repo + a LinkedIn post, laddering toward a real,
healthcare-adjacent product. The fun and the point: watch RL learn live (on the
`rldash` terminal dashboard), push the bounds, and aim the compute at healthcare —
specifically oncology drug + screening problems.

Owner: Patrick Hughes (BMD PAT LLC). GitHub `bmdhodl`. Hardware: RTX 5090 in WSL.

## The strategic arc (why these envs exist)

A 4-agent research pass (saved in `GOLDMINE_research.md`, local/gitignored) found
the goldmine for this compute is **healthcare-adjacent allocation**, and the
loudest finding was: *the RL/GPU technique is never the bottleneck — buyer access
and willingness-to-pay are.* Two sequenced bets came out of it:

- **Act 1 (revenue now): buy-and-bill drug margin-leakage** — sell a
  vendor-neutral wastage/ordering optimizer to community oncology practices on
  share-of-recovered-dollars. Patrick's OneOncology connection is the unfair
  advantage (data access + a design partner). This is the `drug` env.
- **Act 2 (bigger, compute-heavy): cancer-screening deployment-ROI** — sell a
  GPU microsimulation "where to spend screening dollars for max early-stage
  detection" tool to the MCED/pharma HEOR buyers. This is the `screening` env.

The envs are deliberately toy + illustrative-data right now. The product step is
**wiring real data** and validating one paying buyer. Don't lose that thread:
every episode should ladder toward "this is sellable," not just "this is a neat
benchmark."

## Environments (the three skins)

| env (`--env`) | unit | budget | reward | episode | data |
|---|---|---|---|---|---|
| `nashville` | road corridors | yearly capital | livability score delta | 10 yrs | `data/seed_corridors.json` |
| `screening` | TN counties | screening dollars | pop-weighted early detection − equity penalty | 10 yrs | `data/tn_counties.json` |
| `drug` | oncology drugs | (cash, implicit) | margin − wastage − stockouts | 24 weeks | `data/oncology_drugs.json` |

Results so far (PPO best vs the strongest scripted baseline):
- nashville: PPO -0.31 vs greedy -0.62 (deterministic env, flat ceiling — many
  configs converge to the same score).
- screening: PPO **+0.15** vs lookahead-greedy -0.10 (+0.25 uplift; tuning matters
  a lot here).
- drug: PPO **-11.6** vs base-stock greedy -17.4 (+5.8, ~33% less leakage;
  stochastic env, learns fast then plateaus ~-12).

All three are healthcare-adjacent except nashville (the origin toy). `drug` and
`screening` are the ones that matter for the product.

## Architecture (how it fits together)

The whole thing is ~6 small files. The key idea is **one trainer, swappable envs
via a registry.**

- **Env interface contract** — every env exposes exactly this, nothing more:
  `__init__(units=None, years, annual_budget, seed, annual_refill, max_budget_multiple)`,
  `reset(seed) -> (obs:list[float], info:dict)`, `step(action:int) -> (obs, reward, terminated, truncated, info)`,
  `observation_size` (property), `action_size` (property), `sample_action() -> int`,
  `greedy_action() -> int`. Obs is a flat `list[float]`; actions are a single
  discrete int (flattened `unit_idx * n_action_types + action_type`). The trainer
  knows nothing else about the env.
- **`train.py`** — self-contained PPO (PyTorch, GPU). `ENV_REGISTRY` maps an env
  name to `(EnvClass, loader_fn)`; `Config.env` picks it. Full-episode rollouts
  across `num_envs` parallel copies, GAE, PPO clip. Eval is reported two ways:
  `argmax` and `sample` — on deterministic envs the argmax can collapse to a
  do-nothing policy while sampling is good, so the objective is `max(argmax,
  sample)`. Writes `runs/<name>/{config.json, metrics.csv, train.log, best.pt,
  result.json}`. `train(config) -> dict` is importable.
- **`sweep.py`** — time-budgeted random hyperparameter search; in-process so CUDA
  inits once. `--env`, `--years`, `--budget-hours`, `--steps-per-trial`. Writes
  `runs/sweep[_<env>]/{sweep_results.jsonl, leaderboard.txt, champion.json,
  champion.pt}` and copies the best checkpoint to `champion.pt`. Robust per-trial
  try/except; incremental writes survive a kill.
- **`make_dashboard.py`** — renders a sweep into a self-contained static
  `dashboard.html` (inline SVG, no JS/CDN). `--env` selects labels + paths via
  `ENV_META`. GitHub Pages serves it (root `index.html` redirects to the city
  dashboard).
- **`rldash` logging** — the trainer prints one rldash-format line per update to
  `runs/<name>/train.log`:
  `upd N/Ntot step N SPS N ep_ret N ep_len N rps N v_loss N EV N ent N`. Watch any
  run live with `python rldash.py --log "runs/*/train.log"` (rldash =
  github.com/bmdhodl/rldash, Patrick's other repo; it auto-follows the newest run
  and shows a GPU gauge).

## How to add a new environment (the pattern)

1. `mkdir <name>_sim`, write `<name>_sim/env.py` with a `<Thing>Env` class that
   satisfies the interface contract above, plus a `load_<things>()` loader and a
   `data/<name>.json` seed file. Mirror `screening_sim/env.py` — it's the cleanest
   template (deterministic; copy `drug_sim` if you want stochastic dynamics).
2. `<name>_sim/__init__.py` re-exports the env + loader.
3. Register in `train.py` `ENV_REGISTRY`: `"<name>": (<Thing>Env, load_<things>)`.
4. Add a `"<name>"` block to `make_dashboard.py` `ENV_META`.
5. Write `test_<name>.py` (copy `test_drug.py`): print baselines, assert obs size.
   **Run it first** — you want `no_op` clearly worst, `greedy` clearly best
   scripted policy, and a wide-enough band that PPO has room. If greedy ties no_op
   or random wins, the reward is mis-shaped — fix the dynamics before training.
6. Smoke-train (`train.py --env <name> --total-steps 250000 --log-every-updates 999`)
   to confirm PPO learns, then sweep.

Design the `greedy_action` as a *strong* baseline (1-step lookahead for
deterministic envs, a domain heuristic like base-stock for stochastic ones). A
weak baseline makes a fake win; PPO beating a strong baseline is the real result.

## Environment / running (the setup)

- Code + git live on the Windows K: drive: `K:\bmdpat\experiments\nashville-sim`.
  This is its **own** git repo (remote `github.com/bmdhodl/nashville-sim`), nested
  inside the larger `bmdpat` working tree — they're independent.
- Training runs in **WSL** (`Ubuntu-22.04`), where the venv + CUDA torch live:
  `.venv/bin/python` (torch 2.11+cu128, numpy; RTX 5090, sm_120). From WSL the
  drive is `/mnt/k/bmdpat/experiments/nashville-sim`.
- Edit files with Windows tools; run training via
  `wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd /mnt/k/... && .venv/bin/python ...'`.
- `test_*.py` and `make_dashboard.py` are pure stdlib — run them with **Windows**
  python (`python test_drug.py`) without WSL.
- Rebuild the venv if needed: `uv venv --python 3.12 && uv pip install torch
  --index-url https://download.pytorch.org/whl/cu128 && uv pip install numpy`.

## Gotchas (hard-won — do not relearn these)

- **Commit to GitHub immediately.** This folder was once wiped overnight (untracked
  files, likely a cleanup/`git clean`); everything not pushed was lost. The repo
  exists so it can't happen again. Push after every working change.
- **Keep the machine awake** during long runs. Two overnight/long runs died when
  Windows slept. `fullautoresearch/scripts/keep_awake.py` (Patrick's other repo)
  exists for this; or just don't let it sleep.
- **The `drug` env needs `--years 24`.** `Config.years` defaults to 10 (right for
  nashville/screening). The trainer/sweep pass `--years`; the sweep needs it too
  (already wired). Forgetting it silently trains a 10-week drug env.
- **WSL `/mnt/k` can go stale** after a sleep/restart — if a path "doesn't exist"
  but you know it does, the mount is stale; the files are fine on the Windows side.
- **argmax vs sample eval** — see `evaluate_policy`. Objective is the better of the
  two. Don't "fix" a low argmax by assuming the policy failed; check sample.
- **LF→CRLF git warnings on Windows are harmless.** Ignore them.
- **rldash `--once` shows the final frame only**; live (no `--once`) shows the
  climbing sparkline. Don't conclude it's broken from a `--once` snapshot.
- Drug training **plateaus fast** (~1.5M steps); more steps mostly just make a
  longer watch. Don't burn 12M steps expecting more learning.

## Conventions

- **Build-in-public posts** (LinkedIn is primary): builder-to-builder voice, short
  sentences, concrete numbers. **No em dashes.** Never use: harness, leverage,
  streamline, delve, landscape, cutting-edge, game-changer, revolutionary,
  seamless, robust, holistic, synergy, ecosystem. End with the AgentGuard CTA:
  `https://bmdpat.com/tools/agentguard`. Don't auto-post to Patrick's accounts —
  draft, he posts. Blog posts go through the QA-gated pipeline (don't fabricate QA
  provenance).
- Keep illustrative data **labeled as illustrative**. Don't present made-up numbers
  as real CDC/SEER/ASP data. Honesty is the moat for a healthcare-adjacent tool.
- `GOLDMINE_research.md` is **gitignored** (internal strategy). Read it; don't
  publish it.

## Roadmap (prioritized — pick up here)

1. **Drug sweep** (in flight as of this handoff) → commit `runs/sweep_drug/` +
   `dashboard.html`, and see if any config breaks the ~-11.6 ceiling. The
   one-reorder-per-week constraint is likely what caps it — try relaxing it
   (order a small basket per week, or a continuous-ish order vector) and see if
   PPO pulls further ahead of base-stock. That's a strong "RL beats the heuristic
   by even more" result.
2. **Real data.** This is the step that makes it sellable. For `drug`: real
   ASP/wastage/shelf-life from a practice (Patrick's OneOncology path) → the
   margin-leakage number becomes credible. For `screening`: wire CDC PLACES
   (county screening rates) + SEER/State Cancer Profiles (stage-at-dx) + CDC SVI —
   all public, no PHI. A `data/*_live.json` drops in via the existing loader
   fallback (`default_*_path` already checks for a `_live` file first).
3. **Posts.** Episode 3 (drug) post is the obvious next one: "I taught an RL agent
   to run an oncology drug desk and it beat the textbook heuristic by 33%." Then a
   post on watching it live on rldash.
4. **Throughput** (only if needed): the envs are pure-Python and CPU-bound. A
   numpy- or torch-vectorized env would unlock much bigger/faster runs (and make
   the GPU the actual bottleneck). PufferLib is the other option but adds API risk;
   plain vectorization first.
5. **Validation, not just code.** The research is blunt: the buyer is the risk, not
   the model. The highest-value non-code move is one community-oncology admin (via
   OneOncology) agreeing to share a month of billing data for the drug tool, or one
   MCED/pharma market-access lead saying they'd pay for the screening tool.

## File map

```
nashville_sim/  screening_sim/  drug_sim/   # the three env packages (env.py + __init__.py)
data/           seed_corridors.json  tn_counties.json  oncology_drugs.json
train.py        # PPO trainer, env-agnostic via ENV_REGISTRY; emits rldash train.log
sweep.py        # hyperparameter sweep (--env, --years)
make_dashboard.py  # sweep -> static dashboard.html (--env, ENV_META)
test_env.py / test_screening.py / test_drug.py   # baseline sanity checks
champ_config.json / champ_screening_config.json  # champion configs (drug one is generated)
runs/           # outputs (mostly gitignored; curated sweep results + dashboards committed)
README.md       # usage + results;  GOLDMINE_research.md = strategy (gitignored)
index.html      # GitHub Pages redirect to the dashboard
```

Live: https://github.com/bmdhodl/nashville-sim · dashboard
https://bmdhodl.github.io/nashville-sim/ · watch with rldash
(github.com/bmdhodl/rldash).
