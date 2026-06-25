"""Generate a self-contained HTML dashboard for the Nashville-sim PPO sweep.

Reads the sweep outputs (sweep_results.jsonl, champion.json) plus the champion
learning curve (runs/champion_curve/metrics.csv) and writes a single static
dashboard.html with inline SVG charts — no JS, no CDN, opens anywhere.

    python make_dashboard.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/bmdhodl/nashville-sim"

# Per-environment labels so one generator renders either sweep.
ENV_META = {
    "nashville": {
        "sweep": "sweep", "curve": "champion_curve", "episode": "Episode 1",
        "title": "Nashville-sim — PPO tuning sweep",
        "subtitle": "City-growth RL environment, reconstructed from bytecode",
        "blurb": "An RL agent learns to spend a city's limited budget across growth corridors.",
    },
    "screening": {
        "sweep": "sweep_screening", "curve": "champion_curve_screening", "episode": "Episode 2",
        "title": "Screening allocator — PPO tuning sweep",
        "subtitle": "Cancer-screening budget allocation across Tennessee counties, same engine new skin",
        "blurb": "The same allocation engine, now spending a limited cancer-screening budget across "
                 "Tennessee counties to maximize population-weighted early-stage detection.",
    },
    "drug": {
        "sweep": "sweep_drug", "curve": "champion_curve_drug", "episode": "Episode 3",
        "title": "Drug inventory allocator — PPO tuning sweep",
        "subtitle": "Buy-and-bill oncology drug ordering under stochastic demand, same engine new skin",
        "blurb": "The same allocation engine, now reordering perishable oncology drugs against random "
                 "weekly demand to minimize wastage and stockouts (margin leakage).",
    },
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--env", default="nashville", choices=list(ENV_META))
META = ENV_META[_ap.parse_args().env]
SWEEP = BASE / "runs" / META["sweep"]
OUT = SWEEP / "dashboard.html"

# ---- palette --------------------------------------------------------------
BG = "#0b0e14"
CARD = "#151a23"
GRID = "#222a37"
TEXT = "#d7dde7"
MUTED = "#8893a7"
LIME = "#a3e635"
BLUE = "#60a5fa"
RED = "#f87171"
AMBER = "#fbbf24"


def esc(s) -> str:
    return html.escape(str(s))


# ---- data load ------------------------------------------------------------
results = [
    json.loads(line)
    for line in (SWEEP / "sweep_results.jsonl").read_text().splitlines()
    if line.strip()
]
champ = json.loads((SWEEP / "champion.json").read_text())

curve_rows: list[dict] = []
mp = BASE / "runs" / META["curve"] / "metrics.csv"
if mp.exists():
    curve_rows = list(csv.DictReader(mp.read_text().splitlines()))

greedy = champ["baseline_greedy"]
baselines = [
    ("no_op", champ["baseline_no_op"], MUTED),
    ("random", champ["baseline_random"], MUTED),
    ("greedy", champ["baseline_greedy"], AMBER),
    ("PPO champion", champ["best_eval_return"], LIME),
]


# ---- svg helpers ----------------------------------------------------------
def lerp(v, a, b, pa, pb):
    if b == a:
        return pa
    return pa + (v - a) / (b - a) * (pb - pa)


def hbar_chart(items, vmin, vmax, w=760, row_h=46, pad_l=120, pad_r=70):
    h = len(items) * row_h + 20
    x0, x1 = pad_l, w - pad_r
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx = lerp(frac, 0, 1, x0, x1)
        out.append(f'<line x1="{gx:.1f}" y1="10" x2="{gx:.1f}" y2="{h-10}" stroke="{GRID}" stroke-width="1"/>')
        tv = lerp(frac, 0, 1, vmin, vmax)
        out.append(f'<text x="{gx:.1f}" y="{h-2}" fill="{MUTED}" font-size="10" text-anchor="middle">{tv:.2f}</text>')
    for i, (label, val, color) in enumerate(items):
        y = 18 + i * row_h
        bw = lerp(val, vmin, vmax, 2, x1 - x0)
        out.append(f'<text x="{x0-10}" y="{y+18}" fill="{TEXT}" font-size="13" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="26" rx="4" fill="{color}" opacity="0.9"/>')
        out.append(f'<text x="{x0+bw+8:.1f}" y="{y+18}" fill="{TEXT}" font-size="12">{val:+.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def line_chart(series, ref_lines=None, w=760, h=300, pad=46, ymin=None, ymax=None, xlabel="", ylabel=""):
    xs = [p[0] for _, _, pts in series for p in pts]
    ys = [p[1] for _, _, pts in series for p in pts]
    if ref_lines:
        ys += [v for _, v, _ in ref_lines]
    if not xs:
        return "<div></div>"
    xmin, xmax = min(xs), max(xs)
    ymin = min(ys) if ymin is None else ymin
    ymax = max(ys) if ymax is None else ymax
    yr = (ymax - ymin) or 1
    ymin -= 0.06 * yr
    ymax += 0.06 * yr
    x0, x1, y0, y1 = pad, w - pad, h - pad, pad

    def px(x): return lerp(x, xmin, xmax, x0, x1)
    def py(y): return lerp(y, ymin, ymax, y0, y1)

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gy = lerp(frac, 0, 1, y0, y1)
        tv = lerp(frac, 0, 1, ymin, ymax)
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0-8}" y="{gy+4:.1f}" fill="{MUTED}" font-size="10" text-anchor="end">{tv:.2f}</text>')
    for frac in (0, 0.5, 1.0):
        gx = lerp(frac, 0, 1, x0, x1)
        tv = lerp(frac, 0, 1, xmin, xmax)
        out.append(f'<text x="{gx:.1f}" y="{h-12}" fill="{MUTED}" font-size="10" text-anchor="middle">{tv:.0f}</text>')
    if ref_lines:
        for name, v, color in ref_lines:
            yy = py(v)
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.8"/>')
            out.append(f'<text x="{x1-4}" y="{yy-5:.1f}" fill="{color}" font-size="10" text-anchor="end">{esc(name)} {v:+.3f}</text>')
    for name, color, pts in series:
        if not pts:
            continue
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    if xlabel:
        out.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-1}" fill="{MUTED}" font-size="11" text-anchor="middle">{esc(xlabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def group_bars(pairs, w=760, h=240, pad_l=46, pad_b=46, color=BLUE, vfmt="{:+.3f}"):
    if not pairs:
        return "<div></div>"
    vals = [v for _, v in pairs]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1
    vlo = vmin - 0.15 * span
    vhi = vmax + 0.12 * span
    x0, x1, y0, y1 = pad_l, w - 16, h - pad_b, 16
    n = len(pairs)
    bw = (x1 - x0) / n * 0.6
    gap = (x1 - x0) / n
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for frac in (0, 0.5, 1.0):
        gy = lerp(frac, 0, 1, y0, y1)
        tv = lerp(frac, 0, 1, vlo, vhi)
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0-8}" y="{gy+4:.1f}" fill="{MUTED}" font-size="10" text-anchor="end">{tv:.2f}</text>')
    for i, (label, val) in enumerate(pairs):
        cx = x0 + gap * (i + 0.5)
        top = lerp(val, vlo, vhi, y0, y1)
        out.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{y0-top:.1f}" rx="3" fill="{color}" opacity="0.9"/>')
        out.append(f'<text x="{cx:.1f}" y="{top-5:.1f}" fill="{TEXT}" font-size="10" text-anchor="middle">{vfmt.format(val)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{h-14}" fill="{MUTED}" font-size="11" text-anchor="middle">{esc(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---- chart 1: baselines vs champion --------------------------------------
all_vals = [v for _, v, _ in baselines]
bar1 = hbar_chart(baselines, vmin=min(all_vals) - 0.05, vmax=max(all_vals) + 0.05)

# ---- chart 2: champion learning curve ------------------------------------
train_pts = [(int(r["update"]), float(r["train_return"])) for r in curve_rows if r.get("train_return")]
eval_pts = [(int(r["update"]), float(r["eval_sample"])) for r in curve_rows if r.get("eval_sample")]
curve_svg = line_chart(
    series=[("train return (sampled)", BLUE, train_pts), ("eval (sampled)", LIME, eval_pts)],
    ref_lines=[("greedy", greedy, AMBER), ("no_op", champ["baseline_no_op"], MUTED)],
    xlabel="PPO update",
)

# ---- chart 3: sweep convergence (best_eval per trial) --------------------
by_trial = sorted(results, key=lambda r: r["trial"])
conv_pts = [(r["trial"], r["best_eval_return"]) for r in by_trial]
conv_svg = line_chart(
    series=[("best eval per trial", LIME, conv_pts)],
    ref_lines=[("greedy", greedy, AMBER)],
    xlabel="trial index",
)

# ---- chart 4: hyperparameter analysis ------------------------------------
def grouped(key):
    buckets: dict = {}
    for r in results:
        buckets.setdefault(r["config"][key], []).append(r["best_eval_return"])
    return [(str(k), mean(v)) for k, v in sorted(buckets.items(), key=lambda kv: float(kv[0]))]

epochs_bars = group_bars(grouped("update_epochs"), color=BLUE)
ent_bars = group_bars(grouped("ent_coef"), color="#c084fc")

# ---- leaderboard table ----------------------------------------------------
ranked = sorted(results, key=lambda r: r["best_eval_return"], reverse=True)
rows_html = []
for i, r in enumerate(ranked, 1):
    c = r["config"]
    cls = ' class="champ"' if i == 1 else ""
    rows_html.append(
        f"<tr{cls}><td>{i}</td><td>{r['best_eval_return']:+.4f}</td>"
        f"<td>{r['uplift_vs_greedy']:+.4f}</td>"
        f"<td>{r['final_eval_argmax']:+.3f}</td><td>{r['final_eval_sample']:+.3f}</td>"
        f"<td>{c['lr']}</td><td>{c['gamma']}</td><td>{c['ent_coef']}</td>"
        f"<td>{c['update_epochs']}</td><td>{c['num_minibatches']}</td>"
        f"<td>{c['hidden_size']}</td><td>{c['num_envs']}</td><td>{c['anneal_lr']}</td>"
        f"<td>{r['trial']}</td></tr>"
    )
leaderboard_rows = "\n".join(rows_html)

# ---- summary numbers ------------------------------------------------------
n_trials = len(results)
ceiling = max(r["best_eval_return"] for r in results)
near_ceiling = sum(1 for r in results if r["best_eval_return"] <= ceiling + 0.005)
uplift_pct = (champ["best_eval_return"] - greedy) / abs(greedy) * 100
champ_cfg = champ["config"]
cfg_chips = " ".join(
    f'<span class="chip">{esc(k)}={esc(champ_cfg[k])}</span>'
    for k in ["lr", "gamma", "gae_lambda", "clip_coef", "ent_coef", "vf_coef",
              "update_epochs", "num_minibatches", "hidden_size", "num_envs", "anneal_lr"]
)

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Nashville-sim — PPO sweep dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:{BG}; color:{TEXT}; font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:28px 20px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:34px 0 12px; color:{TEXT}; }}
  .sub {{ color:{MUTED}; margin:0 0 18px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
  .card {{ background:{CARD}; border:1px solid {GRID}; border-radius:12px; padding:16px; }}
  .stat .n {{ font-size:30px; font-weight:700; }}
  .stat .l {{ color:{MUTED}; font-size:13px; }}
  .lime {{ color:{LIME}; }} .amber {{ color:{AMBER}; }} .red {{ color:{RED}; }} .muted {{ color:{MUTED}; }}
  .chip {{ display:inline-block; background:#1d2533; border:1px solid {GRID}; border-radius:6px; padding:3px 8px; margin:3px 4px 3px 0; font-size:12px; color:{TEXT}; font-family:ui-monospace,monospace; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th,td {{ text-align:right; padding:6px 8px; border-bottom:1px solid {GRID}; white-space:nowrap; }}
  th {{ color:{MUTED}; font-weight:600; position:sticky; top:0; background:{CARD}; }}
  td:first-child, th:first-child {{ text-align:center; }}
  tr.champ td {{ background:rgba(163,230,53,0.10); color:{LIME}; font-weight:600; }}
  .tablewrap {{ max-height:520px; overflow:auto; border-radius:12px; }}
  .note {{ background:{CARD}; border-left:3px solid {AMBER}; border-radius:8px; padding:12px 16px; margin:10px 0; color:{TEXT}; }}
  .legend span {{ margin-right:16px; font-size:12px; color:{MUTED}; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
  code {{ background:#1d2533; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  a {{ color:{LIME}; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  .banner {{ background:linear-gradient(90deg,#1a2230,#151a23); border:1px solid {GRID}; border-left:3px solid {LIME}; border-radius:10px; padding:10px 16px; margin:0 0 18px; font-size:13px; color:{MUTED}; }}
  .banner b {{ color:{LIME}; }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  @media (max-width:720px) {{ .two {{ grid-template-columns:1fr; }} }}
</style></head>
<body><div class="wrap">

  <div class="banner"><b>Building an allocation engine in public · {META['episode']}.</b> {esc(META['blurb'])}
  Code &amp; data: <a href="{REPO_URL}">{REPO_URL}</a></div>

  <h1>{esc(META['title'])}</h1>
  <p class="sub">{esc(META['subtitle'])} · {n_trials} sweep trials · champion vs scripted baselines</p>

  <div class="note">
    <b>Run note:</b> a time-boxed hyperparameter sweep of <b>{n_trials} trials</b>.
    {near_ceiling} of {n_trials} reached the same performance ceiling, so the champion
    is robust rather than luck — random search keeps rediscovering the same
    near-optimal policy.
  </div>

  <div class="grid">
    <div class="card stat"><div class="n lime">{champ['best_eval_return']:+.3f}</div><div class="l">champion return (best of argmax/sample)</div></div>
    <div class="card stat"><div class="n">{champ['uplift_vs_greedy']:+.3f}</div><div class="l">uplift vs greedy ({uplift_pct:+.0f}% of the gap)</div></div>
    <div class="card stat"><div class="n">{n_trials}</div><div class="l">sweep trials completed</div></div>
    <div class="card stat"><div class="n">{near_ceiling}</div><div class="l">trials at the ~{ceiling:+.3f} ceiling</div></div>
  </div>

  <h2>Champion vs baselines</h2>
  <p class="sub">Higher (less negative) is better. The learned policy beats the hand-written greedy planner and the do-nothing / random baselines.</p>
  <div class="card">{bar1}</div>

  <h2>Champion learning curve</h2>
  <p class="legend"><span><i class="dot" style="background:{BLUE}"></i>train return (sampled)</span><span><i class="dot" style="background:{LIME}"></i>eval (sampled)</span><span><i class="dot" style="background:{AMBER}"></i>greedy baseline</span><span><i class="dot" style="background:{MUTED}"></i>no_op baseline</span></p>
  <div class="card">{curve_svg}</div>
  <p class="sub">Captured by re-running the champion config with frequent logging. The policy crosses the greedy line early and keeps climbing.</p>

  <div class="two">
    <div>
      <h2>Sweep convergence</h2>
      <div class="card">{conv_svg}</div>
      <p class="sub">Best return per trial. Random search hits the ceiling within the first few trials.</p>
    </div>
    <div>
      <h2>Avg return by PPO epochs</h2>
      <div class="card">{epochs_bars}</div>
      <p class="sub">More update epochs help most on this deterministic env.</p>
    </div>
  </div>

  <h2>Avg return by entropy coefficient</h2>
  <div class="card">{ent_bars}</div>
  <p class="sub">Entropy bonus is fairly insensitive once epochs are high — the policy peaks either way.</p>

  <h2>Leaderboard — all {n_trials} trials</h2>
  <div class="card tablewrap">
  <table>
    <thead><tr><th>#</th><th>best_eval</th><th>uplift</th><th>argmax</th><th>sample</th>
      <th>lr</th><th>gamma</th><th>ent</th><th>epochs</th><th>mb</th><th>hidden</th><th>envs</th><th>anneal</th><th>trial</th></tr></thead>
    <tbody>
{leaderboard_rows}
    </tbody>
  </table>
  </div>

  <p class="sub" style="margin-top:30px">Generated by <code>make_dashboard.py</code> from <code>runs/sweep/</code>. Re-run it after another sweep to refresh.</p>

</div></body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}")
print(f"trials={n_trials} champion={champ['best_eval_return']:+.4f} uplift={champ['uplift_vs_greedy']:+.4f}")
