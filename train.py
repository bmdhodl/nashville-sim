"""Self-contained PPO trainer for NashvilleGrowthEnv.

Pure PyTorch + NumPy, GPU-aware. The env is tiny and deterministic (10-step
episodes), so we collect full-episode rollouts across many parallel env copies
and run standard PPO with GAE on the GPU policy.

Usage:
    python train.py                          # default config
    python train.py --config trial.json      # config from file
    python train.py --total-steps 500000 --lr 3e-4 --run-name demo

`train(config) -> dict` is also importable (the sweep calls it in-process).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from nashville_sim import NashvilleGrowthEnv
from nashville_sim.env import load_corridors
from screening_sim import EarlyDetectionEnv
from screening_sim.env import load_counties

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Selectable environments — same interface, different skin. Pick via Config.env.
ENV_REGISTRY = {
    "nashville": (NashvilleGrowthEnv, load_corridors),
    "screening": (EarlyDetectionEnv, load_counties),
}


@dataclass
class Config:
    run_name: str = "ppo"
    seed: int = 5090
    # env
    env: str = "nashville"
    years: int = 10
    annual_budget: float = 1.0
    annual_refill: bool = True
    # ppo
    total_steps: int = 400_000
    num_envs: int = 256
    lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    hidden_size: int = 128
    anneal_lr: bool = True
    # logging / eval
    eval_episodes: int = 256
    log_every_updates: int = 10
    device: str = "cuda"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        # Orthogonal init (PPO standard).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


class VecEnv:
    """Synchronous vectorized env. Full-episode rollouts: reset all, step
    `years` times, every env terminates together on the last step."""

    def __init__(self, num_envs: int, env_cls, units, cfg: Config, base_seed: int):
        self.cfg = cfg
        self.envs = [
            env_cls(
                units,
                years=cfg.years,
                annual_budget=cfg.annual_budget,
                annual_refill=cfg.annual_refill,
                seed=base_seed + i,
            )
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.obs_dim = self.envs[0].observation_size
        self.act_dim = self.envs[0].action_size
        self._rng = np.random.default_rng(base_seed)

    def reset(self) -> np.ndarray:
        seeds = self._rng.integers(0, 2**31 - 1, size=self.num_envs)
        out = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=int(seeds[i]))
            out[i] = obs
        return out

    def step(self, actions: np.ndarray):
        obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.empty(self.num_envs, dtype=np.float32)
        for i, env in enumerate(self.envs):
            o, r, terminated, truncated, _ = env.step(int(actions[i]))
            obs[i] = o
            rewards[i] = r
            dones[i] = 1.0 if (terminated or truncated) else 0.0
        return obs, rewards, dones


def evaluate_baselines(env_cls, units, cfg: Config, episodes: int) -> dict[str, float]:
    """Mean episode return for the scripted baselines, fixed eval seeds."""
    results: dict[str, float] = {}
    for policy in ("no_op", "random", "greedy"):
        total = 0.0
        for ep in range(episodes):
            env = env_cls(
                units,
                years=cfg.years,
                annual_budget=cfg.annual_budget,
                annual_refill=cfg.annual_refill,
                seed=1_000_000 + ep,
            )
            env.reset(seed=1_000_000 + ep)
            ep_return, done = 0.0, False
            while not done:
                if policy == "no_op":
                    a = 0
                elif policy == "random":
                    a = env.sample_action()
                else:
                    a = env.greedy_action()
                _, r, terminated, truncated, _ = env.step(a)
                ep_return += r
                done = terminated or truncated
            total += ep_return
        results[policy] = total / episodes
    return results


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic, env_cls, units, cfg: Config, episodes: int, device, mode: str = "argmax"
) -> float:
    """Eval the learned policy on fixed eval seeds.

    mode="argmax": deterministic greedy action (mode of the policy).
    mode="sample": sample from the policy (reflects stochastic deployment).

    The env is deterministic, so many good action sequences score similarly and
    the argmax can collapse onto one head while sampling spreads usefully; we
    report both and let the objective take the better one.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 777)
    total = 0.0
    for ep in range(episodes):
        env = env_cls(
            units,
            years=cfg.years,
            annual_budget=cfg.annual_budget,
            annual_refill=cfg.annual_refill,
            seed=1_000_000 + ep,
        )
        obs, _ = env.reset(seed=1_000_000 + ep)
        ep_return, done = 0.0, False
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(x)
            if mode == "argmax":
                a = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs, 1, generator=gen).item())
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            done = terminated or truncated
        total += ep_return
    return total / episodes


def train(config: Config | dict[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, Config) else Config(**config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    env_cls, loader = ENV_REGISTRY[cfg.env]
    units = loader()
    vec = VecEnv(cfg.num_envs, env_cls, units, cfg, base_seed=cfg.seed)
    obs_dim, act_dim = vec.obs_dim, vec.act_dim

    model = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    steps_per_rollout = cfg.years  # one full episode per env per update
    batch_per_update = cfg.num_envs * steps_per_rollout
    num_updates = max(1, cfg.total_steps // batch_per_update)
    minibatch_size = max(1, batch_per_update // cfg.num_minibatches)

    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log_path = run_dir / "metrics.csv"
    log_fields = [
        "update", "global_step", "elapsed_s", "train_return",
        "eval_argmax", "eval_sample", "policy_loss", "value_loss",
        "entropy", "approx_kl", "lr",
    ]
    log_file = log_path.open("w", newline="")
    logger = csv.DictWriter(log_file, fieldnames=log_fields)
    logger.writeheader()

    start = time.time()
    best_eval = -float("inf")
    global_step = 0

    # rollout buffers
    B, T = cfg.num_envs, steps_per_rollout
    obs_buf = torch.zeros((T, B, obs_dim), device=device)
    act_buf = torch.zeros((T, B), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, B), device=device)
    rew_buf = torch.zeros((T, B), device=device)
    done_buf = torch.zeros((T, B), device=device)
    val_buf = torch.zeros((T, B), device=device)

    last_metrics: dict[str, Any] = {}
    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.lr

        next_obs = torch.tensor(vec.reset(), dtype=torch.float32, device=device)
        ep_return = torch.zeros(B, device=device)
        for t in range(T):
            obs_buf[t] = next_obs
            action, logp, value = model.act(next_obs)
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value
            obs_np, rew_np, done_np = vec.step(action.cpu().numpy())
            rew_buf[t] = torch.tensor(rew_np, device=device)
            done_buf[t] = torch.tensor(done_np, device=device)
            ep_return += rew_buf[t]
            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            global_step += B

        # GAE (episodes end at t=T-1, so no bootstrap past the end).
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = val_buf[t + 1] if t + 1 < T else torch.zeros(B, device=device)
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idx = np.arange(batch_per_update)
        approx_kl = pg_loss = v_loss = ent = torch.tensor(0.0)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch_per_update, minibatch_size):
                mb = idx[s : s + minibatch_size]
                logits, newval = model(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                newlogp = dist.log_prob(b_act[mb])
                ent = dist.entropy().mean()
                ratio = (newlogp - b_logp[mb]).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (newlogp - b_logp[mb])).mean()
                mb_adv = b_adv[mb]
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        train_return = ep_return.mean().item()
        do_eval = (update % cfg.log_every_updates == 0) or (update == num_updates)
        eval_argmax = eval_sample = float("nan")
        if do_eval:
            eval_argmax = evaluate_policy(model, env_cls, units, cfg, cfg.eval_episodes, device, "argmax")
            eval_sample = evaluate_policy(model, env_cls, units, cfg, cfg.eval_episodes, device, "sample")
            best_now = max(eval_argmax, eval_sample)
            if best_now > best_eval:
                best_eval = best_now
                torch.save(
                    {"model": model.state_dict(), "config": asdict(cfg),
                     "eval_argmax": eval_argmax, "eval_sample": eval_sample,
                     "global_step": global_step},
                    run_dir / "best.pt",
                )
        last_metrics = {
            "update": update,
            "global_step": global_step,
            "elapsed_s": round(time.time() - start, 1),
            "train_return": round(train_return, 4),
            "eval_argmax": round(eval_argmax, 4) if do_eval else "",
            "eval_sample": round(eval_sample, 4) if do_eval else "",
            "policy_loss": round(pg_loss.item(), 5),
            "value_loss": round(v_loss.item(), 5),
            "entropy": round(ent.item(), 5),
            "approx_kl": round(approx_kl.item(), 5),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
        }
        logger.writerow(last_metrics)
        log_file.flush()

    log_file.close()
    baselines = evaluate_baselines(env_cls, units, cfg, cfg.eval_episodes)
    final_argmax = evaluate_policy(model, env_cls, units, cfg, cfg.eval_episodes, device, "argmax")
    final_sample = evaluate_policy(model, env_cls, units, cfg, cfg.eval_episodes, device, "sample")
    best_eval = max(best_eval, final_argmax, final_sample)
    result = {
        "run_name": cfg.run_name,
        "env": cfg.env,
        "best_eval_return": round(best_eval, 4),
        "final_eval_argmax": round(final_argmax, 4),
        "final_eval_sample": round(final_sample, 4),
        "baseline_no_op": round(baselines["no_op"], 4),
        "baseline_random": round(baselines["random"], 4),
        "baseline_greedy": round(baselines["greedy"], 4),
        "uplift_vs_greedy": round(best_eval - baselines["greedy"], 4),
        "elapsed_s": round(time.time() - start, 1),
        "global_step": global_step,
        "num_updates": num_updates,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    for f in dataclasses.fields(Config):
        if f.type == "bool" or isinstance(f.default, bool):
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                           type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                           type=type(f.default), default=None)
    args = p.parse_args()
    base: dict[str, Any] = {}
    if args.config:
        base = json.loads(Path(args.config).read_text())
    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    base.update(overrides)
    return Config(**base)


if __name__ == "__main__":
    cfg = _parse_args()
    res = train(cfg)
    print(json.dumps(res, indent=2))
