"""DrugInventoryEnv — buy-and-bill oncology drug ordering, same engine, new skin.

The agent runs the drug-buying desk at a community oncology practice. Expensive
infusion drugs are bought up front (buy-and-bill), perish after a few weeks, and
patient demand is random. Each week the agent picks ONE drug to reorder and how
many vials. Order too much and vials expire (you eat the full cost — margins are
razor thin). Order too little and a patient's drug is not on the shelf (stockout).

This is the "margin leakage / wastage" problem from the research, as a textbook
perishable-newsvendor / stochastic control task. Unlike the deterministic city
and screening envs, demand here is STOCHASTIC (Poisson per drug, seeded so eval
is reproducible), so the agent must learn to buffer under uncertainty — harder,
and more interesting to watch train.

Same interface as the other envs (reset/step/observation_size/action_size/
sample_action/greedy_action), so train.py / sweep.py / rldash drive it unchanged.

DATA NOTE: data/oncology_drugs.json values are normalized and illustrative (thin
margins, expensive waste, short shelf life — the real buy-and-bill shape), not a
specific price file.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Max vials orderable for the chosen drug in one week: levels 0..ORDER_MAX-1.
ORDER_MAX = 6
# Normalization caps for the observation.
INV_CAP = 10.0
DEMAND_CAP = 6.0

# Reward weights.
STOCKOUT_PENALTY = 0.3   # per unmet vial of demand (lost margin + delayed patient)
HOLDING_COST = 0.01      # per vial held per week


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_drug_path() -> Path:
    live = experiment_root() / "data" / "oncology_drugs_live.json"
    if live.exists():
        return live
    return experiment_root() / "data" / "oncology_drugs.json"


@dataclass(frozen=True)
class Drug:
    name: str
    vial_cost: float       # cash out to buy one vial (normalized)
    reimbursement: float   # cash in when one vial is administered (ASP + markup)
    shelf_life: int        # weeks until an unused vial expires
    demand_lambda: float   # mean weekly vials needed


def load_drugs(path: str | Path | None = None) -> list[Drug]:
    source = Path(path) if path is not None else default_drug_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    drugs: list[Drug] = []
    for row in payload["drugs"]:
        drugs.append(
            Drug(
                name=row["name"],
                vial_cost=float(row["vial_cost"]),
                reimbursement=float(row["reimbursement"]),
                shelf_life=int(row["shelf_life"]),
                demand_lambda=float(row["demand_lambda"]),
            )
        )
    if len(drugs) < 2:
        raise ValueError("DrugInventoryEnv needs at least two drugs")
    return drugs


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's sampler — fine for the small lambdas here."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


class DrugInventoryEnv:
    """Perishable multi-drug inventory control. Reorder one drug per week."""

    def __init__(
        self,
        drugs: list[Drug] | None = None,
        years: int = 24,            # "years" = weeks here; kept for trainer compat
        annual_budget: float = 1.0,  # unused; kept for a uniform constructor
        seed: int = 5090,
        annual_refill: bool = True,
        max_budget_multiple: float = 3.0,
    ) -> None:
        self.drugs = drugs if drugs is not None else load_drugs()
        self.years = years          # number of weeks in an episode
        self.rng = random.Random(seed)
        self.week = 0
        # inventory[d] = list over remaining shelf life; index 0 expires this week.
        self.inv: list[list[int]] = []
        self.last_demand: list[int] = []
        self.last_stockout: list[int] = []
        self.last_info: dict[str, Any] = {}
        self.reset(seed=seed)

    @property
    def observation_size(self) -> int:
        return 1 + 4 * len(self.drugs)

    @property
    def action_size(self) -> int:
        return len(self.drugs) * ORDER_MAX

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.week = 0
        # Start with a couple weeks of average stock so week 0 is not a guaranteed
        # stockout — the agent's job is to keep it balanced, not bootstrap it.
        self.inv = []
        for d in self.drugs:
            vials = max(1, round(d.demand_lambda))
            shelf = [0] * d.shelf_life
            shelf[-1] = vials  # freshest slot
            self.inv.append(shelf)
        self.last_demand = [0] * len(self.drugs)
        self.last_stockout = [0] * len(self.drugs)
        self.last_info = {"margin": 0.0, "wastage": 0.0, "stockouts": 0, "score": 0.0}
        return self._observation(), self.last_info

    def _on_hand(self, d: int) -> int:
        return sum(self.inv[d])

    def sample_action(self) -> int:
        return self.rng.randrange(self.action_size)

    def greedy_action(self) -> int:
        """Base-stock heuristic: reorder the drug furthest below a 2-week target."""
        best_drug, best_gap = 0, -1.0
        for d, drug in enumerate(self.drugs):
            target = math.ceil(drug.demand_lambda * 2)
            gap = target - self._on_hand(d)
            if gap > best_gap:
                best_gap, best_drug = gap, d
        level = int(clamp(round(best_gap), 0, ORDER_MAX - 1))
        return best_drug * ORDER_MAX + level

    def step(self, action: int) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        if not (0 <= action < self.action_size):
            raise ValueError(f"action must be in [0, {self.action_size - 1}]")
        drug_idx = action // ORDER_MAX
        order_level = action % ORDER_MAX

        margin = wastage = holding = 0.0
        stockouts = 0

        # 1) place the order — fresh vials enter the freshest shelf slot.
        drug = self.drugs[drug_idx]
        self.inv[drug_idx][-1] += order_level
        margin -= drug.vial_cost * order_level  # cash out now

        # 2) realize stochastic demand for every drug, fulfilled oldest-first.
        for d, dr in enumerate(self.drugs):
            demand = _poisson(self.rng, dr.demand_lambda)
            self.last_demand[d] = demand
            need = demand
            for age in range(dr.shelf_life):  # oldest (index 0) first
                if need <= 0:
                    break
                take = min(self.inv[d][age], need)
                self.inv[d][age] -= take
                need -= take
                margin += dr.reimbursement * take
            self.last_stockout[d] = need
            stockouts += need

            # 3) age the shelf: index 0 expires (wastage), everything shifts down.
            expired = self.inv[d][0]
            wastage += dr.vial_cost * expired
            for age in range(dr.shelf_life - 1):
                self.inv[d][age] = self.inv[d][age + 1]
            self.inv[d][dr.shelf_life - 1] = 0
            holding += HOLDING_COST * self._on_hand(d)

        self.week += 1
        reward = round(
            margin - wastage - STOCKOUT_PENALTY * stockouts - holding, 4
        )
        terminated = self.week >= self.years
        truncated = False
        self.last_info = {
            "margin": round(margin, 4),
            "wastage": round(wastage, 4),
            "stockouts": stockouts,
            "ordered": f"{order_level}x {drug.name}",
            "score": reward,
            "week": self.week,
        }
        return self._observation(), reward, terminated, truncated, self.last_info

    def _observation(self) -> list[float]:
        obs: list[float] = [clamp((self.years - self.week) / max(1, self.years), 0.0, 1.0)]
        for d, dr in enumerate(self.drugs):
            on_hand = self._on_hand(d)
            expiring = self.inv[d][0]  # expires next week if unused
            obs.extend([
                round(clamp(on_hand / INV_CAP, 0.0, 1.0), 4),
                round(clamp(expiring / INV_CAP, 0.0, 1.0), 4),
                round(clamp(self.last_demand[d] / DEMAND_CAP, 0.0, 1.0), 4),
                round(clamp(self.last_stockout[d] / DEMAND_CAP, 0.0, 1.0), 4),
            ])
        return obs
