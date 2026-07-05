"""
testing/evaluate_policy.py — OFAT sensitivity sweeps for the trained policy.

Self-contained testing harness for the multi-drone search policy. It measures
how a *single* trained checkpoint behaves as we vary one environment / context
factor at a time (One-Factor-At-A-Time, OFAT), holding the others at a fixed
reference operating point. This avoids the exponential blow-up of a full grid
search while still mapping every axis the policy was randomised over.

The harness is **import-only** with respect to the rest of the codebase — it
reuses the project's helpers and never modifies them:
  * ``training.train``        : build_ctx_norm, sample_ctx_params, ctx_to_numpy
  * ``env.grid_env``          : DroneSearchEnv (info["known_cells"] → coverage)
  * ``env.utils``             : bfs_reachable (imported; SPL uses the local
                                bfs_distance helper below)
  * ``agents.dqn_agent``      : DQNAgent (greedy rollout, epsilon = 0)

Outputs (under ``testing_results/csv/`` by default):
  * ``detail_<axis>.csv``     : one row per (config-point, seed) episode.
  * ``agg_<axis>.csv``        : one row per config-point (success_rate + Wilson
                                95% CI, mean reward ± std, coverage, coverage
                                efficiency, steps-to-success, SPL, cooperation).
  * ``heatmap_<pair>.csv``    : 2-D interaction grids (axis = "heatmaps").

Metrics
-------
success_rate    fraction of episodes that reached the target (greedy policy).
mean_reward     mean / std of episode return.
coverage        known free cells / total free cells at episode end.
coverage_eff    coverage divided by agent-steps (exploration speed).
ttf             steps-to-success, **averaged over successful episodes only**
                (averaging timeouts in is misleading — they cap at the budget).
spl             Success weighted by Path Length (Anderson et al., 2018):
                S · d / max(d, p), d = BFS optimal spawn→target distance,
                p = moves taken by the drone that found the target.
collisions      mean number of same-cell overlaps entered per episode.
revisits        mean per-agent cell re-entries per episode (redundancy).
overlap         fraction of team-visited cells visited by more than one drone.

Usage
-----
  python testing/evaluate_policy.py --axis all --seeds 500
  python testing/evaluate_policy.py --axis vision --seeds 1000 --checkpoint checkpoints/Phase3/best.pt
  python testing/evaluate_policy.py --axis epoch --seeds 300
  python testing/evaluate_policy.py --axis heatmaps --seeds 300
  python testing/evaluate_policy.py --axis density --no-ood
"""

import argparse
import copy
import csv
import glob
import math
import os
import re
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch
import yaml

# Make the project root importable when run as ``python testing/evaluate_policy.py``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse existing helpers (imported, never modified).
from training.train import build_ctx_norm, ctx_to_numpy          # noqa: E402
from env.utils import bfs_reachable                              # noqa: E402  (kept for parity / sanity checks)
from env.grid_env import DroneSearchEnv                          # noqa: E402
from agents.dqn_agent import DQNAgent                            # noqa: E402

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

# Reference operating point — every OFAT sweep holds the non-swept axes here.
# Matches the training operating point (phase2: density 0.2, max_steps 200);
# vision/comm/n_agents at the centre of their DR ranges.
REFERENCE = {
    "vision_radius":    3,
    "comm_range":       5,
    "n_agents":         3,
    "obstacle_density": 0.2,
    "max_steps":        200,
}

# kind="ctx"  → varied via set_domain_params + the context vector (in DR space)
# kind="env"  → varied via the env config (NOT randomised during training)
SWEEPS = {
    "n_agents":  {"kind": "ctx", "param": "n_agents",         "id": [1, 2, 3, 4],
                  "ood": [5, 6]},
    "vision":    {"kind": "ctx", "param": "vision_radius",    "id": [1, 2, 3, 4, 5, 6],
                  "ood": [7, 8]},
    "comm":      {"kind": "ctx", "param": "comm_range",       "id": [2, 4, 6, 8, 10, 12],
                  "ood": [0, 14, 16]},
    "density":   {"kind": "env", "param": "obstacle_density", "id": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
                  "ood": [0.35, 0.4]},
    "max_steps": {"kind": "env", "param": "max_steps",        "id": [50, 100, 150, 200, 300],
                  "ood": []},
}

# Targeted 2-D interaction grids (small, not the full product).
HEATMAPS = {
    "nagents_comm":   {"x": ("n_agents",     [1, 2, 3, 4]),
                       "y": ("comm_range",   [2, 5, 8, 12])},
    "vision_density": {"x": ("vision_radius", [1, 2, 4, 6]),
                       "y": ("obstacle_density", [0.1, 0.2, 0.3, 0.4])},
}


# ---------------------------------------------------------------------------
# SPL helper — multi-target-free BFS distance map (local, keeps env.utils intact)
# ---------------------------------------------------------------------------

def bfs_distance(grid: np.ndarray, source: tuple) -> np.ndarray:
    """4-connected BFS shortest-path distance (in moves) from ``source``.

    Returns an (H, W) int array; unreachable / obstacle cells are -1. Used to
    compute the optimal spawn→target path length for SPL. Mirrors the traversal
    rule of ``env.utils.bfs_reachable`` (0 = free, 1 = obstacle) but accumulates
    hop counts instead of a reachable set.
    """
    H, W = grid.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    sr, sc = source
    if grid[sr, sc] == 1:
        return dist
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion (robust near 0/1)."""
    if n == 0:
        return (0.0, 0.0)
    p      = k / n
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def make_env(base_config: dict, env_overrides: dict) -> DroneSearchEnv:
    """Construct a DroneSearchEnv from a deep-copied config with env overrides."""
    cfg = copy.deepcopy(base_config)
    cfg["env"].update(env_overrides)
    return DroneSearchEnv(cfg, render_mode=None)


def rollout_episode(env, agent, ctx_params, ctx_norm, grid_size, device, seed):
    """Run one greedy episode and return its per-episode metrics.

    ``ctx_params`` = {vision_radius, comm_range, n_agents}; it is applied to the
    env via ``set_domain_params`` BEFORE ``reset`` (Invariant 5) so observations
    match the context vector handed to the network.
    """
    env.set_domain_params(**ctx_params)
    obs, info = env.reset(seed=seed)
    n_agents  = env.n_agents
    start_pos = env.agent_pos[0]          # all drones share the spawn corner
    target    = env.target_pos

    ep_reward  = 0.0
    collisions = 0
    moves      = [0] * n_agents           # per-agent move counter (for SPL path length)
    found_by   = -1
    done       = False

    while not done:
        for i in range(n_agents):
            if done:
                break
            ctx_np = ctx_to_numpy(ctx_params, ctx_norm)
            ctx_t  = torch.tensor(ctx_np, dtype=torch.float32, device=device).unsqueeze(0)
            mask   = env._get_action_mask(i)

            action = agent.select_action(
                obs[i]["global"], obs[i]["local"], ctx_t,
                greedy=True, action_mask=mask,
            )
            obs_i, r, terminated, truncated, info = env.step_agent(i, action)
            obs[i]    = obs_i
            ep_reward += r
            moves[i]  += 1

            # Same-cell overlap entered on this move (mirrors env collision logic).
            here = env.agent_pos[i]
            collisions += sum(1 for k in range(n_agents) if k != i and env.agent_pos[k] == here)

            if terminated:
                found_by = i
            done = terminated or truncated

    success = bool(info["found"])
    steps   = info["step"]                                  # total agent-steps

    free_cells = int((env.grid == 0).sum())
    coverage   = info["known_cells"] / free_cells if free_cells else 0.0
    cov_eff    = coverage / steps if steps else 0.0

    # SPL: optimal spawn→target distance vs the finder's own move count.
    spl = 0.0
    if success:
        d = int(bfs_distance(env.grid, start_pos)[target[0], target[1]])
        d = max(d, 1)                                       # guard degenerate d=0
        p = max(moves[found_by], 1)
        spl = d / max(d, p)

    # Cooperation / redundancy from the per-agent trajectory counts.
    traj          = env.agent_trajectory[:, :env.H, :env.W]
    revisits      = float(np.maximum(traj - 1.0, 0.0).sum()) / n_agents
    visited_count = (traj > 0).sum(axis=0)                  # #agents visiting each cell
    team_cells    = int((visited_count > 0).sum())
    overlap       = float((visited_count > 1).sum()) / team_cells if team_cells else 0.0

    return {
        "success":      int(success),
        "reward":       ep_reward,
        "steps":        steps,
        "coverage":     coverage,
        "coverage_eff": cov_eff,
        "ttf":          steps if success else None,
        "spl":          spl,
        "collisions":   collisions,
        "revisits":     revisits,
        "overlap":      overlap,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_DETAIL_FIELDS = ["axis", "value", "in_dist", "seed", "vision", "comm", "n_agents",
                  "density", "max_steps", "success", "reward", "steps", "coverage",
                  "coverage_eff", "ttf", "spl", "collisions", "revisits", "overlap"]

_AGG_FIELDS = ["axis", "value", "in_dist", "n_seeds", "success_rate",
               "wilson_lo", "wilson_hi", "mean_reward", "std_reward",
               "coverage", "coverage_eff", "ttf", "ttf_n", "spl",
               "collisions", "revisits", "overlap"]


def aggregate(axis, value, in_dist, rows):
    """Collapse a config-point's per-episode rows into one summary row."""
    n        = len(rows)
    succ     = sum(r["success"] for r in rows)
    rewards  = np.array([r["reward"] for r in rows], dtype=np.float64)
    lo, hi   = wilson_interval(succ, n)
    ttf_vals = [r["ttf"] for r in rows if r["ttf"] is not None]

    def m(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    return {
        "axis":         axis,
        "value":        value,
        "in_dist":      int(in_dist),
        "n_seeds":      n,
        "success_rate": succ / n if n else 0.0,
        "wilson_lo":    lo,
        "wilson_hi":    hi,
        "mean_reward":  float(rewards.mean()) if n else 0.0,
        "std_reward":   float(rewards.std())  if n else 0.0,
        "coverage":     m("coverage"),
        "coverage_eff": m("coverage_eff"),
        "ttf":          float(np.mean(ttf_vals)) if ttf_vals else float("nan"),
        "ttf_n":        len(ttf_vals),
        "spl":          m("spl"),
        "collisions":   m("collisions"),
        "revisits":     m("revisits"),
        "overlap":      m("overlap"),
    }


# ---------------------------------------------------------------------------
# Config-point builders
# ---------------------------------------------------------------------------

def point_setup(base_config, spec, value):
    """Return (env_overrides, ctx_params) for one swept value.

    ``ctx_params`` always reflects the *actual* {vision, comm, n_agents} so the
    context vector stays coherent with what the env is configured to do.
    """
    ctx = {
        "vision_radius": REFERENCE["vision_radius"],
        "comm_range":    REFERENCE["comm_range"],
        "n_agents":      REFERENCE["n_agents"],
    }
    env_over = {
        "obstacle_density": REFERENCE["obstacle_density"],
        "max_steps":        REFERENCE["max_steps"],
        "n_agents":         REFERENCE["n_agents"],
    }
    param = spec["param"]
    if spec["kind"] == "ctx":
        ctx[param] = value
        if param == "n_agents":
            env_over["n_agents"] = value      # reset reallocates from env.n_agents
    else:                                     # kind == "env"
        env_over[param] = value
    return env_over, ctx


def run_sweep(axis, base_config, agent, ctx_norm, grid_size, device, seeds, use_ood):
    """Run one OFAT axis; return (detail_rows, agg_rows)."""
    spec   = SWEEPS[axis]
    values = list(spec["id"]) + (list(spec["ood"]) if use_ood else [])
    detail, agg = [], []

    for value in values:
        in_dist = value in spec["id"]
        env_over, ctx = point_setup(base_config, spec, value)
        env  = make_env(base_config, env_over)
        rows = []

        it = tqdm(seeds, desc=f"  {axis}={value!s:<6} {'(OOD)' if not in_dist else '     '}",
                  ncols=88, leave=False) if TQDM else seeds
        for seed in it:
            m = rollout_episode(env, agent, ctx, ctx_norm, grid_size, device, seed)
            rows.append(m)
            detail.append({
                "axis": axis, "value": value, "in_dist": int(in_dist), "seed": seed,
                "vision": ctx["vision_radius"], "comm": ctx["comm_range"],
                "n_agents": ctx["n_agents"], "density": env_over["obstacle_density"],
                "max_steps": env_over["max_steps"],
                "success": m["success"], "reward": round(m["reward"], 4),
                "steps": m["steps"], "coverage": round(m["coverage"], 5),
                "coverage_eff": round(m["coverage_eff"], 7),
                "ttf": m["ttf"] if m["ttf"] is not None else "",
                "spl": round(m["spl"], 5), "collisions": m["collisions"],
                "revisits": round(m["revisits"], 3), "overlap": round(m["overlap"], 4),
            })
        env.close()
        a = aggregate(axis, value, in_dist, rows)
        agg.append(a)
        print(f"  {axis:<10}={value!s:<6} {'OOD' if not in_dist else 'ID '}  "
              f"succ={a['success_rate']*100:5.1f}% [{a['wilson_lo']*100:4.1f},{a['wilson_hi']*100:4.1f}]  "
              f"R={a['mean_reward']:7.2f}  cov={a['coverage']*100:5.1f}%  "
              f"SPL={a['spl']:.3f}  ttf={a['ttf']:6.1f}")
    return detail, agg


def run_epoch(base_config, agent_factory, ckpt_glob, ctx_norm, grid_size, device, seeds):
    """Learning curve: every checkpoint evaluated at the reference operating point."""
    ckpts = sorted(glob.glob(ckpt_glob),
                   key=lambda p: int(re.search(r"_ep(\d+)\.pt$", p).group(1)))
    if not ckpts:
        print(f"[epoch] no checkpoints match: {ckpt_glob}")
        return [], []

    env_over, ctx = point_setup(base_config, {"kind": "ctx", "param": "vision_radius"},
                                REFERENCE["vision_radius"])
    detail, agg = [], []
    for ckpt in ckpts:
        ep    = int(re.search(r"_ep(\d+)\.pt$", ckpt).group(1))
        agent = agent_factory()
        agent.load(ckpt)
        agent.epsilon = 0.0
        env  = make_env(base_config, env_over)
        rows = []
        it = tqdm(seeds, desc=f"  ep{ep:<7}", ncols=88, leave=False) if TQDM else seeds
        for seed in it:
            m = rollout_episode(env, agent, ctx, ctx_norm, grid_size, device, seed)
            rows.append(m)
            detail.append({
                "axis": "epoch", "value": ep, "in_dist": 1, "seed": seed,
                "vision": ctx["vision_radius"], "comm": ctx["comm_range"],
                "n_agents": ctx["n_agents"], "density": env_over["obstacle_density"],
                "max_steps": env_over["max_steps"],
                "success": m["success"], "reward": round(m["reward"], 4),
                "steps": m["steps"], "coverage": round(m["coverage"], 5),
                "coverage_eff": round(m["coverage_eff"], 7),
                "ttf": m["ttf"] if m["ttf"] is not None else "",
                "spl": round(m["spl"], 5), "collisions": m["collisions"],
                "revisits": round(m["revisits"], 3), "overlap": round(m["overlap"], 4),
            })
        env.close()
        del agent
        if str(device) == "cuda":
            torch.cuda.empty_cache()
        a = aggregate("epoch", ep, True, rows)
        agg.append(a)
        print(f"  ep{ep:<7}  succ={a['success_rate']*100:5.1f}%  R={a['mean_reward']:7.2f}  "
              f"cov={a['coverage']*100:5.1f}%  SPL={a['spl']:.3f}")
    return detail, agg


def run_heatmaps(base_config, agent, ctx_norm, grid_size, device, seeds, out_dir):
    """Two small 2-D interaction grids; each saved to its own heatmap CSV."""
    fields = ["x_name", "x", "y_name", "y", "n_seeds", "success_rate",
              "wilson_lo", "wilson_hi", "mean_reward", "coverage", "spl"]
    for name, hm in HEATMAPS.items():
        xn, xs = hm["x"]
        yn, ys = hm["y"]
        out_rows = []
        for xv in xs:
            for yv in ys:
                ctx = {"vision_radius": REFERENCE["vision_radius"],
                       "comm_range":    REFERENCE["comm_range"],
                       "n_agents":      REFERENCE["n_agents"]}
                env_over = {"obstacle_density": REFERENCE["obstacle_density"],
                            "max_steps": REFERENCE["max_steps"],
                            "n_agents": REFERENCE["n_agents"]}
                for nm, val in ((xn, xv), (yn, yv)):
                    if nm in ctx:
                        ctx[nm] = val
                        if nm == "n_agents":
                            env_over["n_agents"] = val
                    else:
                        env_over[nm] = val
                env  = make_env(base_config, env_over)
                rows = [rollout_episode(env, agent, ctx, ctx_norm, grid_size, device, s)
                        for s in seeds]
                env.close()
                a  = aggregate(name, f"{xv}x{yv}", True, rows)
                out_rows.append({"x_name": xn, "x": xv, "y_name": yn, "y": yv,
                                 "n_seeds": a["n_seeds"], "success_rate": a["success_rate"],
                                 "wilson_lo": a["wilson_lo"], "wilson_hi": a["wilson_hi"],
                                 "mean_reward": a["mean_reward"], "coverage": a["coverage"],
                                 "spl": a["spl"]})
                print(f"  [{name}] {xn}={xv}, {yn}={yv}: succ={a['success_rate']*100:5.1f}%")
        path = os.path.join(out_dir, f"heatmap_{name}.csv")
        _write_csv(path, fields, out_rows)
        print(f"  -> {path}")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="OFAT sensitivity sweeps for the trained policy.")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/Phase3/best.pt")
    p.add_argument("--ckpt-glob",  default="checkpoints/Phase3/phase2_ConvNeXT_10k_dr_ep*000.pt",
                   help="Glob for the epoch (learning-curve) axis.")
    p.add_argument("--axis", default="all",
                   choices=["all", "n_agents", "vision", "comm", "density",
                            "max_steps", "epoch", "heatmaps"],
                   help="Which sweep(s) to run. 'all' = the 5 OFAT axes.")
    p.add_argument("--seeds",      type=int, default=500, help="Episodes per config point.")
    p.add_argument("--seed-start", type=int, default=2000, help="First seed in the bank.")
    p.add_argument("--ood", dest="ood", action="store_true",  default=True,
                   help="Include out-of-distribution points (default: on).")
    p.add_argument("--no-ood", dest="ood", action="store_false",
                   help="In-distribution points only.")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    p.add_argument("--out-dir", default="testing_results", help="Output root.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but unavailable -> cpu")
        device = "cpu"
    torch_device = torch.device(device)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    grid_size = base_config["env"]["grid_size"]
    ctx_norm  = build_ctx_norm(base_config)
    seeds     = list(range(args.seed_start, args.seed_start + args.seeds))
    csv_dir   = os.path.join(args.out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    def agent_factory():
        return DQNAgent(grid_size=grid_size, n_actions=4, config=base_config, device=device)

    print(f"\n{'=' * 70}")
    print(f"  POLICY TESTING - OFAT sweeps")
    print(f"{'=' * 70}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}   grid: {grid_size}x{grid_size} (fixed)")
    print(f"  seeds      : {len(seeds)}  ({seeds[0]}..{seeds[-1]})  greedy (eps=0)")
    print(f"  OOD        : {'on' if args.ood else 'off'}")
    print(f"  ctx norm   : {ctx_norm}")
    print(f"  reference  : {REFERENCE}")
    print(f"  out        : {csv_dir}/\n")

    # Epoch and heatmap axes manage their own agent lifecycle / loop.
    if args.axis == "epoch":
        detail, agg = run_epoch(base_config, agent_factory, args.ckpt_glob,
                                ctx_norm, grid_size, torch_device, seeds)
        if agg:
            _write_csv(os.path.join(csv_dir, "detail_epoch.csv"), _DETAIL_FIELDS, detail)
            _write_csv(os.path.join(csv_dir, "agg_epoch.csv"),    _AGG_FIELDS,    agg)
            print(f"\n  saved: {csv_dir}/detail_epoch.csv , agg_epoch.csv")
        return

    # All other axes share one loaded agent.
    agent = agent_factory()
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    if args.axis == "heatmaps":
        run_heatmaps(base_config, agent, ctx_norm, grid_size, torch_device, seeds, csv_dir)
        return

    axes = list(SWEEPS.keys()) if args.axis == "all" else [args.axis]
    for axis in axes:
        print(f"\n--- sweep: {axis} ---")
        detail, agg = run_sweep(axis, base_config, agent, ctx_norm, grid_size,
                                torch_device, seeds, args.ood)
        _write_csv(os.path.join(csv_dir, f"detail_{axis}.csv"), _DETAIL_FIELDS, detail)
        _write_csv(os.path.join(csv_dir, f"agg_{axis}.csv"),    _AGG_FIELDS,    agg)
        print(f"  saved: {csv_dir}/detail_{axis}.csv , agg_{axis}.csv")

    with open(os.path.join(csv_dir, "_run_meta.txt"), "w") as f:
        f.write(f"timestamp   : {datetime.now().isoformat()}\n")
        f.write(f"checkpoint  : {args.checkpoint}\n")
        f.write(f"seeds       : {len(seeds)} ({seeds[0]}..{seeds[-1]})\n")
        f.write(f"ood         : {args.ood}\n")
        f.write(f"reference   : {REFERENCE}\n")
        f.write(f"ctx_norm    : {ctx_norm}\n")
    print(f"\n  done.\n")


if __name__ == "__main__":
    main()
