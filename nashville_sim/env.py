"""NashvilleGrowthEnv — small city-builder environment for fast RL prototyping.

The agent plays a Nashville city planner. Each year it spends a limited budget
on one corridor-level intervention (upzone, transit priority, safety work, or a
complete street), trying to relieve housing/jobs pressure, grow transit access,
and cut crash risk while keeping growth spread across corridors instead of
concentrated in one place.

The env follows the Gymnasium reset/step shape without importing Gymnasium, so
it stays dependency-light for smoke tests.

NOTE: this file was reconstructed from the compiled `env.cpython-313.pyc` after
the original source was lost. Field names, action set, observation shape, and
the recovered numeric constants match the bytecode; the surrounding dynamics are
rebuilt to be internally consistent and learnable.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Action indices. The flat action space is corridor_idx * len(ACTION_NAMES) + action_type.
ACTION_NAMES = ("no_op", "upzone", "transit_priority", "safety_work", "complete_street")

# Per-corridor pressure fields, in fixed order. Each is normalized to [0, 1].
PRESSURE_FIELDS = (
    "housing_pressure",
    "jobs_pressure",
    "transit_access",
    "crash_risk",
    "density",
    "political_friction",
)

# Base budget cost per action type (index-aligned with ACTION_NAMES), before
# the political-friction surcharge. Recovered from the bytecode constants.
ACTION_BASE_COST = (0.0, 0.18, 0.22, 0.16, 0.32)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_corridor_path() -> Path:
    live_path = experiment_root() / "data" / "nashville_corridors.json"
    if live_path.exists():
        return live_path
    return experiment_root() / "data" / "seed_corridors.json"


@dataclass(frozen=True)
class Corridor:
    name: str
    housing_pressure: float
    jobs_pressure: float
    transit_access: float
    crash_risk: float
    density: float
    political_friction: float
    geometry_points: list[list[float]] = field(default_factory=list)


def load_corridors(path: str | Path | None = None) -> list[Corridor]:
    source = Path(path) if path is not None else default_corridor_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    corridors: list[Corridor] = []
    for row in payload["corridors"]:
        corridors.append(
            Corridor(
                name=row["name"],
                housing_pressure=float(row.get("housing_pressure", 0.0)),
                jobs_pressure=float(row.get("jobs_pressure", 0.0)),
                transit_access=float(row.get("transit_access", 0.0)),
                crash_risk=float(row.get("crash_risk", 0.0)),
                density=float(row.get("density", 0.0)),
                political_friction=float(row.get("political_friction", 0.0)),
                geometry_points=row.get("geometry_points", []),
            )
        )
    if len(corridors) < 2:
        raise ValueError("NashvilleSim needs at least two corridors")
    return corridors


class NashvilleGrowthEnv:
    """City-growth MDP. Gymnasium reset/step shape, no Gymnasium dependency."""

    def __init__(
        self,
        corridors: list[Corridor] | None = None,
        years: int = 10,
        annual_budget: float = 1.0,
        seed: int = 5090,
        annual_refill: bool = True,
        max_budget_multiple: float = 3.0,
    ) -> None:
        self.corridors = corridors if corridors is not None else load_corridors()
        self.years = years
        self.annual_budget = annual_budget
        # When True the planner receives `annual_budget` each year (unspent
        # budget accumulates up to max_budget_multiple * annual_budget, so
        # saving for a big project is a real choice). When False the whole
        # episode shares one `annual_budget` pool, matching the strict
        # single-pool model recovered from the original bytecode.
        self.annual_refill = annual_refill
        self.max_budget_multiple = max_budget_multiple
        self.rng = random.Random(seed)
        self.year = 0
        self.remaining_budget = annual_budget
        self.state: list[dict[str, float]] = []
        self.last_info: dict[str, Any] = {}
        self.reset(seed=seed)

    # ----- spaces -------------------------------------------------------
    @property
    def observation_size(self) -> int:
        # 2 globals (years_remaining, budget_pressure) + 6 fields per corridor.
        return 2 + len(PRESSURE_FIELDS) * len(self.corridors)

    @property
    def action_size(self) -> int:
        return len(self.corridors) * len(ACTION_NAMES)

    # ----- core loop ----------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.year = 0
        self.remaining_budget = self.annual_budget
        self.state = []
        for c in self.corridors:
            self.state.append(
                {
                    "housing_pressure": c.housing_pressure,
                    "jobs_pressure": c.jobs_pressure,
                    "transit_access": c.transit_access,
                    "crash_risk": c.crash_risk,
                    "density": c.density,
                    "political_friction": c.political_friction,
                }
            )
        self.last_info = self._score_info()
        return self._observation(), self.last_info

    def sample_action(self) -> int:
        return self.rng.randrange(self.action_size)

    def greedy_action(self) -> int:
        """Heuristic baseline: pick the corridor+intervention with the highest
        affordable need. Used only as a benchmark policy, never for training."""
        best_idx = 0
        best_action_type = 0
        best_score = -math.inf
        for idx, row in enumerate(self.state):
            growth_need = row["housing_pressure"] + row["jobs_pressure"] - row["density"]
            risk_need = row["crash_risk"]
            transit_need = 1.0 - row["transit_access"]

            # Candidate interventions and their rough payoff for this corridor.
            candidates = {
                1: growth_need * 0.68,        # upzone relieves housing/jobs
                2: transit_need * 0.62,       # transit_priority
                3: risk_need * 0.58,          # safety_work
                4: (transit_need + risk_need) * 0.4,  # complete_street
            }
            for action_type, payoff in candidates.items():
                cost = self._estimated_action_cost(idx, action_type)
                if cost > self.remaining_budget:
                    continue
                # Prefer higher payoff per dollar.
                score = payoff - 0.5 * cost
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_action_type = action_type
        if best_score <= 0.0:
            best_action_type = 0  # nothing worth paying for -> no_op
        return best_idx * len(ACTION_NAMES) + best_action_type

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        if not (0 <= action < self.action_size):
            raise ValueError(f"action must be in [0, {self.action_size - 1}]")

        # Annual capital allotment for every year after the first (the first
        # year is funded by reset). Unspent budget accumulates up to a cap.
        if self.annual_refill and self.year >= 1:
            self.remaining_budget = round(
                min(
                    self.remaining_budget + self.annual_budget,
                    self.annual_budget * self.max_budget_multiple,
                ),
                4,
            )

        before = self._score_info()["score"]
        corridor_idx = action // len(ACTION_NAMES)
        action_type = action % len(ACTION_NAMES)

        self._apply_action(corridor_idx, action_type)
        for row in self.state:
            self._advance_pressure(row)
        self.year += 1

        after = self._score_info()
        reward = (after["score"] - before) * 1.5
        if self.remaining_budget < 0.0:
            reward += -0.25  # over-budget penalty
        reward = round(reward, 3)

        terminated = self.year >= self.years
        truncated = False
        self.last_info = {
            **after,
            "year": self.year,
            "action": ACTION_NAMES[action_type],
            "corridor": self.corridors[corridor_idx].name,
            "remaining_budget": round(self.remaining_budget, 3),
        }
        return self._observation(), reward, terminated, truncated, self.last_info

    # ----- dynamics -----------------------------------------------------
    def _estimated_action_cost(self, corridor_idx: int, action_type: int) -> float:
        if not (0 <= action_type < len(ACTION_BASE_COST)):
            raise ValueError(f"unknown action type {action_type}")
        row = self.state[corridor_idx]
        friction = row["political_friction"]
        base = ACTION_BASE_COST[action_type]
        # Friction makes interventions more expensive to push through.
        return round(base * (1.0 + 0.5 * friction), 4)

    def _apply_action(self, corridor_idx: int, action_type: int) -> None:
        row = self.state[corridor_idx]
        friction = row["political_friction"]
        cost = self._estimated_action_cost(corridor_idx, action_type)

        if action_type == 0:  # no_op
            return
        elif action_type == 1:  # upzone
            row["density"] = clamp(row["density"] + 0.12, 0.0, 1.0)
            row["housing_pressure"] = clamp(row["housing_pressure"] - 0.08, 0.0, 1.0)
            row["jobs_pressure"] = clamp(row["jobs_pressure"] + 0.03, 0.0, 1.0)
            row["crash_risk"] = clamp(row["crash_risk"] + 0.025, 0.0, 1.0)
            row["political_friction"] = clamp(friction + 0.04, 0.0, 1.0)
        elif action_type == 2:  # transit_priority
            row["transit_access"] = clamp(row["transit_access"] + 0.14, 0.0, 1.0)
            row["crash_risk"] = clamp(row["crash_risk"] - 0.015, 0.0, 1.0)
            row["jobs_pressure"] = clamp(row["jobs_pressure"] - 0.02, 0.0, 1.0)
        elif action_type == 3:  # safety_work
            row["crash_risk"] = clamp(row["crash_risk"] - 0.16, 0.0, 1.0)
            row["political_friction"] = clamp(friction - 0.02, 0.0, 1.0)
        elif action_type == 4:  # complete_street
            row["transit_access"] = clamp(row["transit_access"] + 0.07, 0.0, 1.0)
            row["crash_risk"] = clamp(row["crash_risk"] - 0.05, 0.0, 1.0)
            row["density"] = clamp(row["density"] + 0.05, 0.0, 1.0)
            row["housing_pressure"] = clamp(row["housing_pressure"] - 0.03, 0.0, 1.0)
        else:
            raise ValueError(f"unknown action type {action_type}")

        self.remaining_budget = round(self.remaining_budget - cost, 4)

    def _advance_pressure(self, row: dict[str, float]) -> None:
        # Natural year-over-year drift. Growth pressures rise, transit access
        # erodes with congestion, crash risk creeps up, friction relaxes.
        density = row["density"]
        # Denser corridors absorb a bit more growth pressure before it shows.
        absorb = 1.0 - 0.65 * density
        row["housing_pressure"] = clamp(row["housing_pressure"] + 0.025 * absorb, 0.0, 1.0)
        row["jobs_pressure"] = clamp(row["jobs_pressure"] + 0.018 * absorb, 0.0, 1.0)
        row["transit_access"] = clamp(row["transit_access"] - 0.15 * density * 0.1, 0.0, 1.0)
        row["crash_risk"] = clamp(row["crash_risk"] + 0.02, 0.0, 1.0)
        row["political_friction"] = clamp(
            max(0.0, row["political_friction"] - 0.01), 0.0, 1.0
        )

    # ----- observation & scoring ---------------------------------------
    def _observation(self) -> list[float]:
        years_remaining = clamp((self.years - self.year) / max(1, self.years), 0.0, 1.0)
        budget_pressure = clamp(self.remaining_budget / self.annual_budget, 0.0, 1.0)
        obs: list[float] = [round(years_remaining, 4), round(budget_pressure, 4)]
        for row in self.state:
            obs.extend(round(row[field_name], 4) for field_name in PRESSURE_FIELDS)
        return obs

    def _score_info(self) -> dict[str, Any]:
        n = len(self.state)
        housing_relief = sum(1.0 - row["housing_pressure"] for row in self.state) / n
        jobs_relief = sum(1.0 - row["jobs_pressure"] for row in self.state) / n
        transit = sum(row["transit_access"] for row in self.state) / n
        safety = sum(1.0 - row["crash_risk"] for row in self.state) / n

        density_values = [row["density"] for row in self.state]
        growth_concentration = max(density_values) - min(density_values)

        over_budget = max(0.0, -self.remaining_budget)

        score = (
            1.4 * housing_relief
            + 0.9 * jobs_relief
            + 0.8 * transit
            + 1.2 * safety
            - 1.7 * growth_concentration
            - over_budget
        )
        return {
            "score": round(score, 4),
            "housing_relief": round(housing_relief, 4),
            "jobs_relief": round(jobs_relief, 4),
            "transit": round(transit, 4),
            "safety": round(safety, 4),
            "growth_concentration": round(growth_concentration, 4),
            "over_budget": round(over_budget, 4),
        }
