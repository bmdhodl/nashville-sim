"""Overnight PPO hyperparameter sweep. Works on any registered env (--env).

Time-budgeted random search. Runs all trials in one process so the CUDA context
initializes once. Each trial trains a PPO agent with sampled hyperparameters and
records its best deployable return (max of argmax / sampled eval). Results stream
to disk after every trial, a leaderboard is kept sorted, and the best agent's
checkpoint is copied to champion.pt.

Output dir is runs/sweep for the city env and runs/sweep_<env> otherwise, so a
screening sweep never clobbers the nashville results.

Safe to run unattended:
  * hard wall-clock budget (default 8h); stops launching trials near the limit
  * every trial wrapped in try/except — one bad config never kills the run
  * incremental writes, so a crash/kill still leaves a usable leaderboard
  * Ctrl-C exits cleanly with a summary

Usage:
  python sweep.py --env screening --budget-hours 0.3
  python sweep.py --budget-hours 0.1 --steps-per-trial 100000   # quick test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import time
import traceback
from pathlib import Path

import torch

from train import Config, train

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Hyperparameter search space. Each entry is a list of choices; the sampler
# picks uniformly. Kept to PPO ranges that are sane for a small discrete env.
SEARCH_SPACE = {
    "lr": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    "gamma": [0.9, 0.93, 0.95, 0.97, 0.99],
    "gae_lambda": [0.9, 0.95, 0.98],
    "clip_coef": [0.1, 0.2, 0.3],
    "ent_coef": [0.0, 0.001, 0.003, 0.01, 0.03],
    "vf_coef": [0.25, 0.5, 1.0],
    "update_epochs": [2, 4, 8],
    "num_minibatches": [2, 4, 8],
    "hidden_size": [64, 128, 256],
    "num_envs": [128, 256, 512],
    "anneal_lr": [True, False],
}


def sample_config(rng: random.Random, trial_idx: int, steps_per_trial: int,
                  env: str, sweep_name: str, years: int | None = None) -> Config:
    chosen = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    kw = dict(
        run_name=f"{sweep_name}/trials/trial_{trial_idx:04d}",
        env=env,
        seed=5090 + trial_idx,
        total_steps=steps_per_trial,
        eval_episodes=96,
        log_every_updates=10_000,  # only the final eval matters for the sweep
        **chosen,
    )
    if years is not None:
        kw["years"] = years  # drug env runs a 24-week horizon
    return Config(**kw)


def write_leaderboard(results: list[dict], path: Path, top: int = 25) -> None:
    ranked = sorted(results, key=lambda r: r["best_eval_return"], reverse=True)
    lines = [
        "rank  best_eval  uplift_vs_greedy  argmax   sample   lr       gamma  ent     "
        "epochs  mb   hidden  envs  anneal  trial",
    ]
    for i, r in enumerate(ranked[:top], 1):
        c = r["config"]
        lines.append(
            f"{i:>4}  {r['best_eval_return']:>9.4f}  {r['uplift_vs_greedy']:>16.4f}  "
            f"{r['final_eval_argmax']:>7.3f}  {r['final_eval_sample']:>7.3f}  "
            f"{c['lr']:<7}  {c['gamma']:<5}  {c['ent_coef']:<6}  "
            f"{c['update_epochs']:<6}  {c['num_minibatches']:<3}  {c['hidden_size']:<6}  "
            f"{c['num_envs']:<4}  {str(c['anneal_lr']):<6}  {r['run_name']}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="nashville")
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--steps-per-trial", type=int, default=800_000)
    ap.add_argument("--years", type=int, default=None, help="episode horizon override (drug env: 24)")
    ap.add_argument("--seed", type=int, default=5090)
    args = ap.parse_args()

    sweep_name = "sweep" if args.env == "nashville" else f"sweep_{args.env}"
    sweep_dir = RUNS_DIR / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_path = sweep_dir / "sweep_results.jsonl"
    leaderboard_path = sweep_dir / "leaderboard.txt"
    champion_path = sweep_dir / "champion.pt"
    champion_meta_path = sweep_dir / "champion.json"

    budget_s = args.budget_hours * 3600.0
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[sweep] env={args.env} device={device} budget={args.budget_hours}h "
          f"steps/trial={args.steps_per_trial:,}")
    print(f"[sweep] results -> {results_path}")

    results: list[dict] = []
    best_overall = -float("inf")
    start = time.time()
    trial_idx = 0
    durations: list[float] = []

    results_file = results_path.open("w")
    try:
        while True:
            elapsed = time.time() - start
            est_trial = (sum(durations) / len(durations)) if durations else 90.0
            if elapsed + est_trial > budget_s:
                print(f"[sweep] stopping: {elapsed/3600:.2f}h elapsed, "
                      f"next trial (~{est_trial:.0f}s) would exceed budget.")
                break

            cfg = sample_config(rng, trial_idx, args.steps_per_trial, args.env, sweep_name, args.years)
            t0 = time.time()
            try:
                result = train(cfg)
            except Exception as exc:  # noqa: BLE001 — one trial must not kill the run
                print(f"[sweep] trial {trial_idx:04d} FAILED: {exc}")
                (sweep_dir / f"trial_{trial_idx:04d}_error.txt").write_text(
                    traceback.format_exc()
                )
                trial_idx += 1
                continue
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()
            dt = time.time() - t0
            durations.append(dt)
            durations[:] = durations[-20:]  # rolling estimate

            result["config"] = dataclasses.asdict(cfg)
            result["trial"] = trial_idx
            results.append(result)
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()

            tag = ""
            if result["best_eval_return"] > best_overall:
                best_overall = result["best_eval_return"]
                src = RUNS_DIR / cfg.run_name / "best.pt"
                if src.exists():
                    shutil.copy(src, champion_path)
                champion_meta_path.write_text(json.dumps(result, indent=2))
                tag = "  <-- NEW BEST"

            write_leaderboard(results, leaderboard_path)
            print(
                f"[sweep] trial {trial_idx:04d} | {dt:5.1f}s | "
                f"best_eval={result['best_eval_return']:+.4f} "
                f"(argmax={result['final_eval_argmax']:+.3f} "
                f"sample={result['final_eval_sample']:+.3f}) "
                f"uplift={result['uplift_vs_greedy']:+.4f} | "
                f"trials={len(results)} elapsed={elapsed/3600:.2f}h{tag}"
            )
            trial_idx += 1
    except KeyboardInterrupt:
        print("\n[sweep] interrupted — writing final summary.")
    finally:
        results_file.close()

    if results:
        ranked = sorted(results, key=lambda r: r["best_eval_return"], reverse=True)
        best = ranked[0]
        print("\n" + "=" * 70)
        print(f"[sweep] DONE. {len(results)} trials in {(time.time()-start)/3600:.2f}h")
        print(f"[sweep] champion best_eval = {best['best_eval_return']:+.4f}  "
              f"(greedy baseline {best['baseline_greedy']:+.4f}, "
              f"uplift {best['uplift_vs_greedy']:+.4f})")
        print(f"[sweep] champion config: {json.dumps(best['config'])}")
        print(f"[sweep] leaderboard -> {leaderboard_path}")
        print(f"[sweep] champion checkpoint -> {champion_path}")
    else:
        print("[sweep] no completed trials.")


if __name__ == "__main__":
    main()
