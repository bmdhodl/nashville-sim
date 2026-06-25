"""EarlyDetectionEnv — cancer-screening budget allocation, same engine as the
Nashville city-builder, new skin.

The agent is a state cancer-control / value-based-care planner. Each year it
spends a limited screening + outreach budget across Tennessee counties, choosing
how to lift screening uptake and shift cancer detection earlier (lower late-stage
share). The objective is population-weighted early-stage detection per dollar,
with an equity penalty for leaving the worst-covered counties behind.

This is the Act-2 framing from the research: "given $X of screening budget,
where do we spend it across populations to maximize early-stage detections per
dollar." It reuses train.py / sweep.py unchanged via the same env interface as
NashvilleGrowthEnv.

DATA NOTE: data/tn_counties.json ships with *illustrative* values calibrated to
the well-documented urban/rural and Social-Vulnerability disparity in Tennessee
cancer outcomes. They are NOT pulled from a source yet. The loader reads a fixed
schema so real inputs drop in cleanly:
  - screening_rate  <- CDC PLACES (county colorectal/breast/cervical screening)
  - late_stage_share<- SEER / NCI State Cancer Profiles (stage at diagnosis)
  - incidence       <- State Cancer Profiles / USCS
  - svi             <- CDC/ATSDR Social Vulnerability Index
  - population      <- Census
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Interventions. Flat action space is county_idx * len(ACTION_NAMES) + action_type.
ACTION_NAMES = ("no_op", "mailer_outreach", "mobile_screening", "patient_navigation", "provider_activation")

# Per-county mutable fields, in fixed order. Each normalized to [0, 1].
SCREENING_FIELDS = (
    "screening_rate",     # share of eligible population up to date on screening
    "late_stage_share",   # share of cancers caught at a late stage (lower is better)
    "incidence",          # cancer incidence / burden
    "svi",                # social vulnerability / access barrier (raises cost)
    "awareness",          # population awareness of screening
    "capacity",           # local screening + follow-up capacity
)

# Base budget cost per action type (index-aligned with ACTION_NAMES).
ACTION_BASE_COST = (0.0, 0.12, 0.26, 0.18, 0.30)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_county_path() -> Path:
    live_path = experiment_root() / "data" / "tn_counties_live.json"
    if live_path.exists():
        return live_path
    return experiment_root() / "data" / "tn_counties.json"


@dataclass(frozen=True)
class County:
    name: str
    population: float          # raw population, used to weight impact
    screening_rate: float
    late_stage_share: float
    incidence: float
    svi: float
    awareness: float
    capacity: float


def load_counties(path: str | Path | None = None) -> list[County]:
    source = Path(path) if path is not None else default_county_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    counties: list[County] = []
    for row in payload["counties"]:
        counties.append(
            County(
                name=row["name"],
                population=float(row.get("population", 1.0)),
                screening_rate=float(row.get("screening_rate", 0.0)),
                late_stage_share=float(row.get("late_stage_share", 0.0)),
                incidence=float(row.get("incidence", 0.0)),
                svi=float(row.get("svi", 0.0)),
                awareness=float(row.get("awareness", 0.0)),
                capacity=float(row.get("capacity", 0.0)),
            )
        )
    if len(counties) < 2:
        raise ValueError("EarlyDetectionEnv needs at least two counties")
    return counties


class EarlyDetectionEnv:
    """Screening-allocation MDP. Same reset/step interface as NashvilleGrowthEnv."""

    def __init__(
        self,
        counties: list[County] | None = None,
        years: int = 10,
        annual_budget: float = 1.0,
        seed: int = 5090,
        annual_refill: bool = True,
        max_budget_multiple: float = 3.0,
    ) -> None:
        self.counties = counties if counties is not None else load_counties()
        self.years = years
        self.annual_budget = annual_budget
        self.annual_refill = annual_refill
        self.max_budget_multiple = max_budget_multiple
        self.rng = random.Random(seed)
        # Population weights (sum to 1) — how much each county counts toward impact.
        total_pop = sum(c.population for c in self.counties) or 1.0
        self.pop_weight = [c.population / total_pop for c in self.counties]
        self.year = 0
        self.remaining_budget = annual_budget
        self.state: list[dict[str, float]] = []
        self.last_info: dict[str, Any] = {}
        self.reset(seed=seed)

    # ----- spaces -------------------------------------------------------
    @property
    def observation_size(self) -> int:
        return 2 + len(SCREENING_FIELDS) * len(self.counties)

    @property
    def action_size(self) -> int:
        return len(self.counties) * len(ACTION_NAMES)

    # ----- core loop ----------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.year = 0
        self.remaining_budget = self.annual_budget
        self.state = []
        for c in self.counties:
            self.state.append(
                {
                    "screening_rate": c.screening_rate,
                    "late_stage_share": c.late_stage_share,
                    "incidence": c.incidence,
                    "svi": c.svi,
                    "awareness": c.awareness,
                    "capacity": c.capacity,
                }
            )
        self.last_info = self._score_info()
        return self._observation(), self.last_info

    def sample_action(self) -> int:
        return self.rng.randrange(self.action_size)

    def greedy_action(self) -> int:
        """One-step-lookahead baseline: try every affordable intervention, keep
        the one that most improves this year's score. A strong, objective-aligned
        benchmark — PPO has to beat it by planning across years, not just locally.
        Benchmark only, never trained."""
        base = self._score_info()["score"]
        saved_state = [dict(r) for r in self.state]
        saved_budget = self.remaining_budget
        best_action, best_gain = 0, 0.0
        for a in range(self.action_size):
            cidx, atype = divmod(a, len(ACTION_NAMES))
            if atype == 0:
                continue
            if self._estimated_action_cost(cidx, atype) > saved_budget:
                continue
            self.state = [dict(r) for r in saved_state]
            self.remaining_budget = saved_budget
            self._apply_action(cidx, atype)
            gain = self._score_info()["score"] - base
            if gain > best_gain:
                best_gain, best_action = gain, a
        self.state = saved_state
        self.remaining_budget = saved_budget
        return best_action

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        if not (0 <= action < self.action_size):
            raise ValueError(f"action must be in [0, {self.action_size - 1}]")

        if self.annual_refill and self.year >= 1:
            self.remaining_budget = round(
                min(
                    self.remaining_budget + self.annual_budget,
                    self.annual_budget * self.max_budget_multiple,
                ),
                4,
            )

        before = self._score_info()["score"]
        county_idx = action // len(ACTION_NAMES)
        action_type = action % len(ACTION_NAMES)

        self._apply_action(county_idx, action_type)
        for row in self.state:
            self._advance(row)
        self.year += 1

        after = self._score_info()
        reward = (after["score"] - before) * 1.5
        if self.remaining_budget < 0.0:
            reward += -0.25
        reward = round(reward, 3)

        terminated = self.year >= self.years
        truncated = False
        self.last_info = {
            **after,
            "year": self.year,
            "action": ACTION_NAMES[action_type],
            "county": self.counties[county_idx].name,
            "remaining_budget": round(self.remaining_budget, 3),
        }
        return self._observation(), reward, terminated, truncated, self.last_info

    # ----- dynamics -----------------------------------------------------
    def _estimated_action_cost(self, county_idx: int, action_type: int) -> float:
        if not (0 <= action_type < len(ACTION_BASE_COST)):
            raise ValueError(f"unknown action type {action_type}")
        svi = self.state[county_idx]["svi"]
        base = ACTION_BASE_COST[action_type]
        # Higher social vulnerability makes interventions costlier to run.
        return round(base * (1.0 + 0.5 * svi), 4)

    def _apply_action(self, county_idx: int, action_type: int) -> None:
        row = self.state[county_idx]
        svi = row["svi"]
        cost = self._estimated_action_cost(county_idx, action_type)

        if action_type == 0:  # no_op
            return
        elif action_type == 1:  # mailer_outreach — cheap reach, weaker in high-svi
            row["screening_rate"] = clamp(row["screening_rate"] + 0.10 * (1.0 - 0.4 * svi), 0.0, 1.0)
            row["awareness"] = clamp(row["awareness"] + 0.06, 0.0, 1.0)
        elif action_type == 2:  # mobile_screening — strong, better in high-svi
            row["screening_rate"] = clamp(row["screening_rate"] + 0.14 + 0.08 * svi, 0.0, 1.0)
            row["late_stage_share"] = clamp(row["late_stage_share"] - 0.03, 0.0, 1.0)
        elif action_type == 3:  # patient_navigation — completes follow-up, cuts late stage
            row["late_stage_share"] = clamp(row["late_stage_share"] - 0.14, 0.0, 1.0)
            row["screening_rate"] = clamp(row["screening_rate"] + 0.04, 0.0, 1.0)
        elif action_type == 4:  # provider_activation — durable capacity, lowers barrier
            row["capacity"] = clamp(row["capacity"] + 0.18, 0.0, 1.0)
            row["screening_rate"] = clamp(row["screening_rate"] + 0.07, 0.0, 1.0)
            row["svi"] = clamp(svi - 0.04, 0.0, 1.0)
        else:
            raise ValueError(f"unknown action type {action_type}")

        self.remaining_budget = round(self.remaining_budget - cost, 4)

    def _advance(self, row: dict[str, float]) -> None:
        # Natural year-over-year drift. Uptake attrites unless capacity supports it;
        # late-stage share is pulled toward a target set by current screening_rate
        # (more screening -> earlier detection over time); awareness fades.
        row["screening_rate"] = clamp(
            row["screening_rate"] - 0.03 + 0.02 * row["capacity"], 0.0, 1.0
        )
        late_target = clamp(0.7 - 0.5 * row["screening_rate"] + 0.1 * row["incidence"], 0.0, 1.0)
        row["late_stage_share"] = clamp(
            row["late_stage_share"] + 0.25 * (late_target - row["late_stage_share"]), 0.0, 1.0
        )
        row["awareness"] = clamp(row["awareness"] - 0.02, 0.0, 1.0)
        row["incidence"] = clamp(row["incidence"] + 0.005, 0.0, 1.0)
        row["capacity"] = clamp(row["capacity"] - 0.012, 0.0, 1.0)

    # ----- observation & scoring ---------------------------------------
    def _observation(self) -> list[float]:
        years_remaining = clamp((self.years - self.year) / max(1, self.years), 0.0, 1.0)
        budget_pressure = clamp(self.remaining_budget / self.annual_budget, 0.0, 1.0)
        obs: list[float] = [round(years_remaining, 4), round(budget_pressure, 4)]
        for row in self.state:
            obs.extend(round(row[field_name], 4) for field_name in SCREENING_FIELDS)
        return obs

    def _score_info(self) -> dict[str, Any]:
        # Population-weighted early-stage detection value: counties with more
        # cancer burden and more population count more, and detecting earlier
        # (low late_stage_share) is the win.
        early_value = sum(
            self.pop_weight[i] * row["incidence"] * (1.0 - row["late_stage_share"])
            for i, row in enumerate(self.state)
        )
        coverage = sum(
            self.pop_weight[i] * row["screening_rate"] for i, row in enumerate(self.state)
        )
        rates = [row["screening_rate"] for row in self.state]
        disparity = max(rates) - min(rates)  # equity penalty
        over_budget = max(0.0, -self.remaining_budget)

        score = (
            2.0 * early_value
            + 1.0 * coverage
            - 0.8 * disparity
            - over_budget
        )
        return {
            "score": round(score, 4),
            "early_value": round(early_value, 4),
            "coverage": round(coverage, 4),
            "disparity": round(disparity, 4),
            "over_budget": round(over_budget, 4),
        }
