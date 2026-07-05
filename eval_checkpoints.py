"""
eval_checkpoints.py — Robust checkpoint evaluation.

Each checkpoint is evaluated on the SAME N instances (fixed seeds + domain
randomisation sampled deterministically per seed), guaranteeing a fair
comparison while still testing varied {map, vision, comm, n_agents}. Results
are saved to CSV and printed as a final ranking.

Two modes:
  * batch  (default) — evaluate the checkpoints matching ``--pattern`` once,
    then print/save the ranking.
  * watch  (``--watch``) — poll ``--ckpt_dir`` and evaluate each new checkpoint
    as soon as the training process writes it, appending its summary to
    ``watch_summary.csv`` incrementally. When the training process writes the
    sentinel ``--done_file``, evaluate any stragglers, then read the summary CSV
    back and print/save the final ranking. Typically spawned automatically by
    ``training/train.py`` when ``training.eval_during_training`` is true.

Usage:
  python eval_checkpoints.py
  python eval_checkpoints.py --episodes 200 --last_n 10
  python eval_checkpoints.py --phase 5 --last_n 20
  python eval_checkpoints.py --watch --pattern "phase2_..._ep*.pt" --device cuda
"""

import argparse
import csv
import glob
import os
import re
import time
from datetime import datetime

import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--ckpt_dir",  default="checkpoints")
    parser.add_argument("--pattern",   default="mas_10k_dr_faults_ep*.pt")
    parser.add_argument("--episodes",  type=int, default=100)
    parser.add_argument("--last_n",    type=int, default=None,
                        help="Evaluate only the last N checkpoints (sorted by ep).")
    parser.add_argument("--phase",     type=int, default=-1,
                        help="Curriculum phase index (default: last = phase7)")
    parser.add_argument("--out_dir",   default="eval_results",
                        help="Directory to save CSV and logs.")
    parser.add_argument("--device",    default=None,
                        help="Force eval device: 'cuda' or 'cpu' (default: auto).")
    parser.add_argument("--no_dr",     action="store_true",
                        help="Disable domain randomisation (use fixed config context).")
    parser.add_argument("--fault_prob", type=float, default=None,
                        help="Force a fixed per-step drone fault probability for every "
                             "episode (overrides the DR-sampled / config value).")
    parser.add_argument("--auto_nav",  action="store_true",
                        help="Drones that know the target follow the BFS shortest path "
                             "on their known map instead of the network policy.")
    # --- watch mode (parallel evaluation during training) -----------------
    parser.add_argument("--watch",     action="store_true",
                        help="Poll ckpt_dir and evaluate new checkpoints as they appear.")
    parser.add_argument("--done_file", default=None,
                        help="Sentinel file whose existence signals training is finished "
                             "(watch mode). Default: <ckpt_dir>/.training_done")
    parser.add_argument("--poll_interval", type=float, default=10.0,
                        help="Seconds between checkpoint-dir polls (watch mode).")
    return parser.parse_args()


def build_eval_config(config: dict, phase_idx: int = -1) -> dict:
    import copy
    cfg   = copy.deepcopy(config)
    phase = cfg["curriculum"][phase_idx]
    e     = cfg["env"]
    # grid_size stays unchanged: single fixed global constant
    e["n_agents"]         = phase["n_agents"]
    e["obstacle_density"] = phase.get("obstacle_density", e["obstacle_density"])
    e["max_steps"]        = phase.get("max_steps",        e["max_steps"])
    return cfg


def evaluate_checkpoint(
    ckpt_path:  str,
    config:     dict,
    phase_idx:  int,
    seeds:      list,
    device:     str,
    out_dir:    str,
    ctx_norm:   dict,
    grid_size:  int,
    dr_cfg:     dict = None,
    fault_prob: float = None,
    auto_nav:   bool = False,
) -> dict:
    """Evaluate one checkpoint over ``seeds`` greedy episodes.

    When ``dr_cfg`` is given, the context {vision_radius, comm_range, n_agents}
    (plus obstacle_density / fault_prob when their ranges are present) is
    sampled deterministically from a per-seed RNG (the same mechanism as
    ``training.train.evaluate``): identical instances across checkpoints, varied
    across episodes. When ``dr_cfg`` is ``None`` a fixed context from the config
    is used for every episode.

    ``fault_prob`` (when not None) overrides the sampled/config value with a
    fixed per-step drone fault probability. ``auto_nav`` switches a drone that
    knows the target to the deterministic BFS shortest path on its known map.
    """
    import random
    from agents.dqn_agent import DQNAgent
    from env.grid_env import DroneSearchEnv
    from env.utils import auto_nav_action
    from training.train import ctx_to_numpy, sample_ctx_params

    n_episodes = len(seeds)
    eval_cfg   = build_eval_config(config, phase_idx)
    env        = DroneSearchEnv(eval_cfg, render_mode=None)

    # Fixed-context fallback (used only when DR is disabled).
    fixed_ctx = {
        "vision_radius": config["env"]["vision_radius"],
        "comm_range":    config["env"]["comm_range"],
        "n_agents":      eval_cfg["env"]["n_agents"],
    }
    if dr_cfg is None:
        env.set_domain_params(**fixed_ctx)

    agent = DQNAgent(
        grid_size = grid_size,
        n_actions = 4,
        config    = config,
        device    = device,
    )
    agent.load(ckpt_path)
    agent.epsilon = 0.0

    ckpt_name    = os.path.basename(ckpt_path)
    episode_rows = []

    if TQDM_AVAILABLE:
        ep_iter = tqdm(
            range(n_episodes),
            desc       = f"  {ckpt_name:<38}",
            ncols      = 92,
            unit       = "ep",
            leave      = True,
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} ep  [{elapsed}<{remaining}  {rate_fmt}]",
        )
    else:
        ep_iter = range(n_episodes)

    total_rewards = []
    found_count   = 0
    step_list     = []
    broken_list   = []

    for ep in ep_iter:
        # Sample the DR context deterministically from the (fixed) seed so every
        # checkpoint is tested on the same {map, vision, comm, n_agents}, and
        # apply it to the env so observations match the context.
        if dr_cfg is not None:
            ctx_params = sample_ctx_params(dr_cfg, rng=random.Random(seeds[ep]))
            env.set_domain_params(**ctx_params)
        else:
            ctx_params = fixed_ctx
        # An explicit --fault_prob overrides whatever DR/config set.
        if fault_prob is not None:
            env.set_domain_params(fault_prob=fault_prob)

        obs, info = env.reset(seed=seeds[ep])
        ep_reward = 0.0
        done      = False

        while not done:
            for i in range(env.n_agents):
                if done:
                    break

                if not env.agent_alive[i]:
                    # Broken drone: no-op turn, the clock still ticks.
                    obs_i, r, terminated, truncated, info = env.step_agent(i, 0)
                    obs[i] = obs_i
                    done   = terminated or truncated
                    continue

                action = None
                if auto_nav:
                    action, _ = auto_nav_action(env, i)
                if action is None:
                    ctx_np = ctx_to_numpy(
                        ctx_params, ctx_norm, i, env.n_alive_belief(i)
                    )
                    ctx_t  = torch.tensor(
                        ctx_np, dtype=torch.float32, device=torch.device(device)
                    ).unsqueeze(0)

                    mask   = env._get_action_mask(i)
                    action = agent.select_action(
                        obs[i]["global"], obs[i]["local"],
                        ctx_t,
                        greedy=True, action_mask=mask,
                    )
                obs_i, r, terminated, truncated, info = env.step_agent(i, action)
                obs[i]    = obs_i
                ep_reward += r
                done       = terminated or truncated

        total_rewards.append(ep_reward)
        found    = info["found"]
        steps    = info["step"]
        n_broken = env.n_agents - int(env.agent_alive.sum())
        if found:
            found_count += 1
        step_list.append(steps)
        broken_list.append(n_broken)

        episode_rows.append({
            "checkpoint": ckpt_name,
            "episode":    ep,
            "seed":       seeds[ep],
            "reward":     round(ep_reward, 4),
            "found":      int(found),
            "steps":      steps,
            "vision":     ctx_params["vision_radius"],
            "comm":       ctx_params["comm_range"],
            "n_agents":   ctx_params["n_agents"],
            "fault_prob": round(float(env.fault_prob), 5),
            "n_broken":   n_broken,
        })

        if TQDM_AVAILABLE:
            ep_iter.set_postfix({
                "found":  f"{found_count}/{ep+1}",
                "mean_r": f"{np.mean(total_rewards):.2f}",
            }, refresh=False)

    env.close()

    rewards_arr = np.array(total_rewards)
    summary = {
        "ckpt":          ckpt_name,
        "sum_reward":    float(rewards_arr.sum()),
        "mean_reward":   float(rewards_arr.mean()),
        "std_reward":    float(rewards_arr.std()),
        "median_reward": float(np.median(rewards_arr)),
        "min_reward":    float(rewards_arr.min()),
        "max_reward":    float(rewards_arr.max()),
        "found_rate":    found_count / n_episodes,
        "mean_steps":    float(np.mean(step_list)),
        "mean_broken":   float(np.mean(broken_list)),
    }

    ep_num   = re.search(r"_ep(\d+)\.pt$", ckpt_name).group(1)
    csv_path = os.path.join(out_dir, f"detail_ep{ep_num}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    print(
        f"    -> sum={summary['sum_reward']:>9.2f}  "
        f"mean={summary['mean_reward']:>7.3f} +/- {summary['std_reward']:.3f}  "
        f"found={summary['found_rate']*100:>5.1f}%  "
        f"steps={summary['mean_steps']:>6.1f}  "
        f"[saved: {os.path.basename(csv_path)}]"
    )

    # Release the agent's GPU memory before the next checkpoint (matters when the
    # evaluator shares the GPU with a running training process).
    del agent
    if str(device) == "cuda":
        torch.cuda.empty_cache()

    return summary


def save_summary_csv(results: list, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fields = ["rank", "ckpt", "sum_reward", "mean_reward", "std_reward",
              "median_reward", "min_reward", "max_reward", "found_rate",
              "mean_steps", "mean_broken"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, res in enumerate(results, 1):
            row = {"rank": rank}
            row.update(res)
            writer.writerow(row)
    return path


# Incremental per-checkpoint summary written during watch mode (one row each).
WATCH_SUMMARY = "watch_summary.csv"
_SUMMARY_NUM_FIELDS = ("sum_reward", "mean_reward", "std_reward", "median_reward",
                       "min_reward", "max_reward", "found_rate", "mean_steps",
                       "mean_broken")


def append_watch_summary(summary: dict, out_dir: str, ep_num: int):
    """Append one checkpoint's summary to the incremental watch CSV.

    Writing a row as soon as each checkpoint is evaluated means a crash or an
    interruption never loses the work already done — the final ranking is read
    back from this file.
    """
    path   = os.path.join(out_dir, WATCH_SUMMARY)
    fields = ["ckpt", "ep", *_SUMMARY_NUM_FIELDS, "timestamp"]
    is_new = not os.path.exists(path)
    row    = {"ckpt": summary["ckpt"], "ep": ep_num,
              "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update({k: summary[k] for k in _SUMMARY_NUM_FIELDS})
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return path


def read_watch_summary(out_dir: str) -> list:
    """Read the incremental watch CSV back into a list of summary dicts."""
    path = os.path.join(out_dir, WATCH_SUMMARY)
    if not os.path.exists(path):
        return []
    results = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            res = {"ckpt": r["ckpt"]}
            for k in _SUMMARY_NUM_FIELDS:
                res[k] = float(r[k])
            results.append(res)
    return results


def print_and_save_ranking(results: list, config: dict, phase_idx: int,
                           episodes: int, out_dir: str):
    """Sort the per-checkpoint summaries by sum_reward, print and save the ranking."""
    results_sorted = sorted(results, key=lambda x: x["sum_reward"], reverse=True)
    phase_name     = config["curriculum"][phase_idx]["name"]
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 90}")
    print(f"  RANKING - {phase_name}, {episodes} greedy ep, fixed seeds 0..{episodes-1}".center(90))
    print(f"{'=' * 90}")
    print(
        f"  {'#':<4} {'Checkpoint':<38} "
        f"{'Sum':>9}  {'Mean':>7}  {'Std':>7}  "
        f"{'Median':>7}  {'Found':>6}  {'Steps':>6}"
    )
    print(f"  {'-' * 86}")
    for rank, res in enumerate(results_sorted, 1):
        marker = "  <= BEST" if rank == 1 else ""
        print(
            f"  {rank:<4} {res['ckpt']:<38} "
            f"{res['sum_reward']:>9.2f}  "
            f"{res['mean_reward']:>7.3f}  "
            f"{res['std_reward']:>7.3f}  "
            f"{res['median_reward']:>7.3f}  "
            f"{res['found_rate']*100:>5.1f}%  "
            f"{res['mean_steps']:>6.1f}"
            f"{marker}"
        )
    print(f"{'=' * 90}")

    best = results_sorted[0]
    print(f"\n  *  Best checkpoint : {best['ckpt']}")
    print(f"     Sum reward     : {best['sum_reward']:.2f}")
    print(f"     Found rate     : {best['found_rate']*100:.1f}%")
    print(f"     Mean steps     : {best['mean_steps']:.1f}")
    print(f"     Mean reward/ep : {best['mean_reward']:.3f} +/- {best['std_reward']:.3f}")
    print(f"     Median reward  : {best['median_reward']:.3f}\n")

    summary_file = save_summary_csv(results_sorted, out_dir, f"ranking_{timestamp}.csv")
    print(f"  CSV ranking  -> {summary_file}")
    print(f"  CSV detail   -> {out_dir}/detail_ep*.csv")
    print()
    return summary_file


def _ckpt_ep(path: str) -> int:
    return int(re.search(r"_ep(\d+)\.pt$", path).group(1))


def _resolve_device(arg_device) -> str:
    if arg_device:
        if arg_device == "cuda" and not torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA is unavailable -> using cpu")
            return "cpu"
        return arg_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dr_cfg(config: dict, phase_idx: int, no_dr: bool):
    if no_dr:
        return None
    from training.train import get_dr_cfg
    return get_dr_cfg(config, config["curriculum"][phase_idx])


def watch_loop(args, config, ctx_norm, grid_size, device, dr_cfg):
    """Evaluate new checkpoints as the training process writes them.

    Polls ``args.ckpt_dir`` for files matching ``args.pattern``; each new one is
    evaluated and its summary appended to ``watch_summary.csv``. When the
    training process creates ``args.done_file``, a final sweep catches any
    stragglers, then the ranking is built from the accumulated CSV.
    """
    done_file  = args.done_file or os.path.join(args.ckpt_dir, ".training_done")
    pattern    = os.path.join(args.ckpt_dir, args.pattern)
    eval_seeds = list(range(args.episodes))

    # Resume safety: skip checkpoints already present in the incremental CSV.
    evaluated = {r["ckpt"] for r in read_watch_summary(args.out_dir)}
    if evaluated:
        print(f"[watch] resuming - {len(evaluated)} checkpoint(s) already in {WATCH_SUMMARY}")

    print(f"[watch] pattern   : {pattern}")
    print(f"[watch] done-file : {done_file}")
    print(f"[watch] {args.episodes} greedy ep/ckpt  |  DR={'on' if dr_cfg else 'off'}  |  device={device}\n")

    def sweep() -> int:
        ckpts = sorted(glob.glob(pattern), key=_ckpt_ep)
        n = 0
        for c in ckpts:
            name = os.path.basename(c)
            if name in evaluated:
                continue
            try:
                summary = evaluate_checkpoint(
                    ckpt_path=c, config=config, phase_idx=args.phase,
                    seeds=eval_seeds, device=device, out_dir=args.out_dir,
                    ctx_norm=ctx_norm, grid_size=grid_size, dr_cfg=dr_cfg,
                    fault_prob=args.fault_prob, auto_nav=args.auto_nav,
                )
                append_watch_summary(summary, args.out_dir, _ckpt_ep(c))
                evaluated.add(name)
                n += 1
            except Exception as e:   # keep watching even if one checkpoint fails
                print(f"[watch] FAILED on {name}: {e}  - retry next sweep")
                if str(device) == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(args.poll_interval)
        return n

    while True:
        sweep()
        if os.path.exists(done_file):
            # Training finished: one more sweep to catch checkpoints written
            # between our last glob and the sentinel, then stop when none remain.
            if sweep() == 0:
                break
        else:
            time.sleep(args.poll_interval)

    results = read_watch_summary(args.out_dir)
    if not results:
        print("[watch] no checkpoints were evaluated.")
        return
    print(f"[watch] training finished - evaluated {len(results)} checkpoint(s). Ranking:")
    print_and_save_ranking(results, config, args.phase, args.episodes, args.out_dir)


def run_batch(args, config, ctx_norm, grid_size, device, dr_cfg):
    """Evaluate the checkpoints matching --pattern once, then rank them."""
    eval_seeds = list(range(args.episodes))
    print(f"Eval seeds   : 0 .. {args.episodes - 1}  (fixed for all checkpoints)\n")

    pattern    = os.path.join(args.ckpt_dir, args.pattern)
    ckpt_files = sorted(glob.glob(pattern), key=_ckpt_ep)

    if not ckpt_files:
        print(f"[ERROR] No checkpoints found with pattern: {pattern}")
        return

    if args.last_n is not None:
        skipped    = max(0, len(ckpt_files) - args.last_n)
        ckpt_files = ckpt_files[-args.last_n:]
        ep_skip    = skipped * 50
        print(f"Total checkpoints : {skipped + len(ckpt_files)}  "
              f"|  Evaluating last {len(ckpt_files)}  "
              f"(skipped ep50 -> ep{ep_skip})\n")
    else:
        print(f"Checkpoints found : {len(ckpt_files)}  |  Evaluating all\n")

    print(f"{args.episodes} greedy episodes per checkpoint  |  DR={'on' if dr_cfg else 'off'}\n")
    print("-" * 92)

    results = []
    for ckpt in ckpt_files:
        res = evaluate_checkpoint(
            ckpt_path  = ckpt,
            config     = config,
            phase_idx  = args.phase,
            seeds      = eval_seeds,
            device     = device,
            out_dir    = args.out_dir,
            ctx_norm   = ctx_norm,
            grid_size  = grid_size,
            dr_cfg     = dr_cfg,
            fault_prob = args.fault_prob,
            auto_nav   = args.auto_nav,
        )
        results.append(res)

    print_and_save_ranking(results, config, args.phase, args.episodes, args.out_dir)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Single source of truth: both values derived once from config at startup
    from training.train import build_ctx_norm
    grid_size = config["env"]["grid_size"]
    ctx_norm  = build_ctx_norm(config)

    os.makedirs(args.out_dir, exist_ok=True)
    device = _resolve_device(args.device)
    dr_cfg = _resolve_dr_cfg(config, args.phase, args.no_dr)

    print(f"\nDevice       : {device}")
    print(f"Grid size    : {grid_size}x{grid_size}  (fixed)")
    print(f"CTX norm     : {ctx_norm}")
    print(f"Domain rand. : {'on (per-seed)' if dr_cfg else 'off (fixed context)'}")
    print(f"Output dir   : {args.out_dir}/")
    if not TQDM_AVAILABLE:
        print("[WARN] tqdm not installed -> pip install tqdm\n")

    if args.watch:
        watch_loop(args, config, ctx_norm, grid_size, device, dr_cfg)
    else:
        run_batch(args, config, ctx_norm, grid_size, device, dr_cfg)


if __name__ == "__main__":
    main()
