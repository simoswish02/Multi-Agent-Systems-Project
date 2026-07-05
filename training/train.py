import os
import sys
import glob
import copy
import random
import subprocess
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env.grid_env import DroneSearchEnv
from agents.dqn_agent import DQNAgent
from agents.nstep import NStepBuffer


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ctx_norm(config: dict) -> dict:
    """
    Derive normalisation denominators from domain_randomization max values.
    Denominator = max_value so the normalised value is exactly 1.0 when the
    sampled value equals the maximum (full range [min/max, 1.0]).

    This is the single source of truth: changing the DR ranges in
    default.yaml automatically updates normalisation without touching
    any other file.
    """
    dr = config["domain_randomization"]
    return {
        "vision_radius": float(dr["vision_radius"][1]),
        "comm_range":    float(dr["comm_range"][1]),
        "n_agents":      float(dr["n_agents"][1]),
    }


def sample_ctx_params(dr_cfg: dict, rng=None) -> dict:
    """Sample context parameters uniformly from domain-randomisation ranges.

    Args:
        dr_cfg: Dict with ``vision_radius`` / ``comm_range`` / ``n_agents``
            ``[min, max]`` ranges.
        rng: Optional ``random.Random`` instance for deterministic sampling
            (used by evaluation so a fixed seed yields a fixed config). Defaults
            to the global ``random`` module.

    Returns:
        Dict with sampled integer ``vision_radius`` / ``comm_range`` /
        ``n_agents``. These must be applied to the env via
        ``DroneSearchEnv.set_domain_params`` so observations match the context.
    """
    rnd = rng if rng is not None else random
    return {
        "vision_radius": rnd.randint(*dr_cfg["vision_radius"]),
        "comm_range":    rnd.randint(*dr_cfg["comm_range"]),
        "n_agents":      rnd.randint(*dr_cfg["n_agents"]),
    }


def ctx_to_numpy(ctx_params: dict, ctx_norm: dict) -> np.ndarray:
    """
    Normalise a context-parameter dict to a (CTX_DIM=3,) float32 array:
    [vision_radius, comm_range, n_agents].

    ctx_norm is passed explicitly so there are no module-level constants; all
    denominators come from the config file (domain_randomization max values).
    The drone position is no longer part of the context — it travels as the
    Own_Position global observation channel, from which the network recovers
    the crop coordinates. The context is therefore constant within an episode.
    """
    return np.array(
        [
            ctx_params["vision_radius"] / ctx_norm["vision_radius"],
            ctx_params["comm_range"]    / ctx_norm["comm_range"],
            ctx_params["n_agents"]      / ctx_norm["n_agents"],
        ],
        dtype=np.float32,
    )


def make_phase_config(base_config: dict, phase: dict, max_agents: int) -> dict:
    cfg = copy.deepcopy(base_config)
    e   = cfg["env"]
    e["n_agents"]         = phase["n_agents"]
    e["obstacle_density"] = phase.get("obstacle_density", e["obstacle_density"])
    e["max_steps"]        = phase.get("max_steps",        e["max_steps"])
    # grid_size is fixed globally in config["env"]["grid_size"]; not set per-phase
    return cfg


def get_dr_cfg(config: dict, phase_meta: dict) -> dict:
    """Return the DR ranges to use: phase dr_override if present, else global."""
    return phase_meta.get("dr_override", config["domain_randomization"])


def count_total_episodes(config: dict) -> int:
    return sum(phase.get("n_episodes", 0) for phase in config["curriculum"])


# Fixed seeds used for in-training evaluation across all phases.
# Using a constant set guarantees that every evaluate() call tests the
# agent on the exact same maps, making the metric comparable across epochs.
_EVAL_SEEDS = list(range(100, 120))  # 20 fixed maps: seed 100..119


def evaluate(
    env,
    agent,
    n_episodes:   int   = 20,
    eval_epsilon: float = 0.05,
    dr_cfg:       dict  = None,
    device:       torch.device = None,
    ctx_norm:     dict  = None,
    grid_size:    int   = None,
    eval_seeds:   list  = None,
):
    """
    Evaluate the agent on a fixed set of maps.

    eval_seeds: list of integer seeds, one per episode.  When provided,
    each episode resets the env with the corresponding seed so that the
    same maps are used regardless of when evaluate() is called during
    training.  Defaults to _EVAL_SEEDS.
    """
    if eval_seeds is None:
        eval_seeds = _EVAL_SEEDS

    # Clamp n_episodes to the number of available fixed seeds
    n_episodes = min(n_episodes, len(eval_seeds))

    rewards, successes = [], []
    saved_eps     = agent.epsilon
    agent.epsilon = eval_epsilon

    for ep_idx in range(n_episodes):
        # Sample the DR config deterministically from the (fixed) eval seed so
        # every evaluate() call tests the same {map, vision, comm, n_agents},
        # and apply it to the env so observations match the context.
        if dr_cfg is not None:
            ctx_params = sample_ctx_params(dr_cfg, rng=random.Random(eval_seeds[ep_idx]))
            env.set_domain_params(**ctx_params)
        else:
            ctx_params = {
                "vision_radius": env.vision_radius,
                "comm_range":    env.comm_range,
                "n_agents":      env.n_agents,
            }

        # Fixed seed: same map layout at every call to evaluate()
        obs, info = env.reset(seed=eval_seeds[ep_idx])
        n_agents  = env.n_agents
        # Context is constant within an episode (position now lives in the obs).
        ctx_np    = ctx_to_numpy(ctx_params, ctx_norm)
        ep_r = 0.0
        done = False

        while not done:
            masks   = [env._get_action_mask(i) for i in range(n_agents)]
            actions = agent.select_actions_batch(obs, [ctx_np] * n_agents, masks)

            for i in range(n_agents):
                if done:
                    break
                obs_i, r, terminated, truncated, info = env.step_agent(i, actions[i])
                obs[i]  = obs_i
                ep_r   += r
                done    = terminated or truncated

        rewards.append(ep_r)
        successes.append(float(info["found"]))

    agent.epsilon = saved_eps
    return float(np.mean(rewards)), float(np.mean(successes))


def _is_new_best(success_rate, mean_r, best_success, best_reward):
    if success_rate > best_success:
        return True
    if success_rate == 1.0 and best_success == 1.0 and mean_r > best_reward:
        return True
    return False


def run_phase(
    phase_cfg, phase_meta, agent, writer, global_episode,
    best_success, best_reward, train_cfg, dr_cfg, device,
    ctx_norm, grid_size,
    phase_idx=1, start_ep=1, apply_eps_reset=True,
):
    name         = phase_meta["name"]
    n_episodes   = phase_meta["n_episodes"]
    n_agents     = phase_cfg["env"]["n_agents"]
    eval_eps     = train_cfg.get("eval_epsilon", 0.05)
    update_every = train_cfg.get("update_every", 4)

    # n-step returns + cooperative terminal-reward variant.
    gamma         = phase_cfg["agent"]["gamma"]
    n_step        = phase_cfg["agent"].get("n_step", 1)
    target_reward = phase_cfg["env"]["target_reward"]
    shared_term   = phase_cfg["env"].get("shared_target_reward", False)
    if shared_term and n_step < 2:
        tqdm.write(
            f"[{name}] WARNING: shared_target_reward needs n_step >= 2 to credit "
            f"the team (n_step={n_step}); the shared bonus will have no effect."
        )

    # On a fresh phase, reset epsilon to the phase value. When resuming into a
    # partially-completed phase, keep the epsilon restored from the checkpoint.
    eps_reset = phase_meta.get("epsilon_reset", None)
    if apply_eps_reset and eps_reset is not None:
        agent.epsilon = eps_reset

    env = DroneSearchEnv(phase_cfg)

    recent_rewards   = []
    recent_successes = []
    step_counter     = 0

    def _save_last(ep):
        """Save the resumable checkpoint (latest training state)."""
        agent.save(
            os.path.join(train_cfg["checkpoint_dir"], "last.pt"),
            training_state={
                "global_episode": global_episode,
                "phase_idx":      phase_idx,
                "ep_in_phase":    ep,
                "best_success":   best_success,
                "best_reward":    best_reward,
            },
        )

    pbar = tqdm(
        range(start_ep, n_episodes + 1),
        desc=name,
        unit="ep",
        initial=start_ep - 1,
        total=n_episodes,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    for ep in pbar:
        ctx_params = sample_ctx_params(dr_cfg)

        # Apply the sampled DR params to the env BEFORE reset so the drone's
        # actual field of view, comm range and team size match the context
        # vector fed to the network (coherent partial observability).
        env.set_domain_params(**ctx_params)
        obs, info = env.reset()
        n_agents  = env.n_agents
        # Context is constant within an episode now that position rides in the
        # Own_Position obs channel — same vector for every agent and timestep.
        ctx_np    = ctx_to_numpy(ctx_params, ctx_norm)
        # Per-agent n-step return accumulator (terminated, not truncated, ends a
        # window — time-limit truncation must still bootstrap, Pardo et al. 2018).
        nstep     = NStepBuffer(n_agents, n_step, gamma)
        finder    = None
        ep_reward = 0.0
        done      = False

        def _consume(transitions):
            """Push emitted n-step transitions and update on the step cadence."""
            for tr in transitions:
                agent.push_transition(*tr)

        while not done:
            masks = [env._get_action_mask(i) for i in range(n_agents)]

            actions = agent.select_actions_batch(obs, [ctx_np] * n_agents, masks)

            obs_before_g = [obs[i]["global"] for i in range(n_agents)]
            obs_before_l = [obs[i]["local"]  for i in range(n_agents)]

            for i in range(n_agents):
                if done:
                    break

                obs_i, reward, terminated, truncated, info = env.step_agent(i, actions[i])
                obs[i] = obs_i
                done   = terminated or truncated
                if terminated:
                    finder = i

                # Context is identical before/after the step (constant within
                # the episode), so the same ctx_np is the state and next-state ctx.
                _consume(nstep.push(
                    i,
                    obs_before_g[i], obs_before_l[i], ctx_np,
                    actions[i], reward,
                    obs_i["global"], obs_i["local"], ctx_np,
                    terminated,
                ))

                step_counter += 1
                if step_counter % update_every == 0:
                    agent.update()

                ep_reward += reward

        # Episode over: optionally share the terminal reward across the team,
        # then flush each agent's pending n-step windows into the buffer.
        if shared_term and finder is not None:
            nstep.apply_team_terminal(finder, target_reward)
        for i in range(n_agents):
            _consume(nstep.flush(i))

        agent.step_episode_scheduler()
        agent.decay_epsilon()
        global_episode += 1

        recent_rewards.append(ep_reward)
        recent_successes.append(float(info["found"]))
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            recent_successes.pop(0)

        current_lr = agent.current_lr()

        pbar.set_postfix({
            "r50":     f"{np.mean(recent_rewards):.2f}",
            "suc50":   f"{np.mean(recent_successes):.0%}",
            "eps":     f"{agent.epsilon:.3f}",
            "lr":      f"{current_lr:.2e}",
            "best_sr": f"{best_success:.0%}",
            "dr_v":    str(ctx_params["vision_radius"]),
        }, refresh=False)

        writer.add_scalar(f"{name}/episode_reward",  ep_reward,            global_episode)
        writer.add_scalar(f"{name}/epsilon",          agent.epsilon,        global_episode)
        writer.add_scalar(f"{name}/episode_length",   info["step"],         global_episode)
        writer.add_scalar(f"{name}/found_target",     float(info["found"]), global_episode)
        writer.add_scalar(f"{name}/lr",               current_lr,           global_episode)
        writer.add_scalar(f"{name}/dr_vision_radius", ctx_params["vision_radius"], global_episode)
        writer.add_scalar(f"{name}/dr_comm_range",    ctx_params["comm_range"],    global_episode)
        writer.add_scalar(f"{name}/dr_n_agents",      ctx_params["n_agents"],      global_episode)
        writer.add_scalar("train/episode_reward",     ep_reward,            global_episode)
        writer.add_scalar("train/epsilon",            agent.epsilon,        global_episode)
        writer.add_scalar("train/lr",                 current_lr,           global_episode)

        if ep % train_cfg["eval_every"] == 0:
            mean_r, success_rate = evaluate(
                env, agent, train_cfg["eval_episodes"], eval_eps,
                dr_cfg=dr_cfg, device=device,
                ctx_norm=ctx_norm, grid_size=grid_size,
                # eval_seeds defaults to _EVAL_SEEDS (fixed, shared across all phases)
            )
            writer.add_scalar(f"{name}/eval_success_rate", success_rate, global_episode)
            writer.add_scalar(f"{name}/eval_mean_reward",  mean_r,       global_episode)
            writer.add_scalar("eval/success_rate",         success_rate, global_episode)
            writer.add_scalar("eval/mean_reward",          mean_r,       global_episode)

            tqdm.write(
                f"  [{name} {ep:4d}/{n_episodes}] "
                f"eval_r={mean_r:7.3f}  "
                f"success={success_rate:.1%}  "
                f"eps={agent.epsilon:.3f}  "
                f"lr={current_lr:.2e}"
            )

            if _is_new_best(success_rate, mean_r, best_success, best_reward):
                best_success = success_rate
                best_reward  = mean_r
                path = os.path.join(train_cfg["checkpoint_dir"], "best.pt")
                agent.save(path)
                tqdm.write(
                    f"  --> New best: success={best_success:.1%}  "
                    f"eval_r={best_reward:.3f}  saved to {path}"
                )
                pbar.set_postfix({
                    "r50":     f"{np.mean(recent_rewards):.2f}",
                    "suc50":   f"{np.mean(recent_successes):.0%}",
                    "eps":     f"{agent.epsilon:.3f}",
                    "lr":      f"{current_lr:.2e}",
                    "best_sr": f"{best_success:.0%}",
                }, refresh=True)

        if ep % train_cfg["save_every"] == 0:
            agent.save(os.path.join(
                train_cfg["checkpoint_dir"], f"{name}_ep{ep}.pt"
            ))
            _save_last(ep)

    # Save the resumable checkpoint at phase end so an interruption between
    # phases resumes from the next phase.
    _save_last(n_episodes)

    pbar.close()
    env.close()
    return global_episode, best_success, best_reward


# ---------------------------------------------------------------------------
# Parallel evaluation during training (optional, config-driven)
# ---------------------------------------------------------------------------

def _spawn_parallel_evaluator(config, config_path, train_cfg, done_file):
    """Launch ``eval_checkpoints.py --watch`` in a separate process.

    The evaluator watches the active phase's checkpoint pattern and evaluates
    each new checkpoint as the training loop writes it, then ranks them when we
    create ``done_file``. Returns a handle dict (or ``None`` if not started).
    """
    active = [(i, p) for i, p in enumerate(config["curriculum"])
              if p.get("n_episodes", 0) > 0]
    if not active:
        print("[train] eval_during_training is on but no active phase "
              "(n_episodes>0); evaluator not started.")
        return None
    if len(active) > 1:
        print(f"[train] note: multiple active phases; the evaluator watches only "
              f"the last one ('{active[-1][1]['name']}').")

    phase_idx, phase_meta = active[-1]
    pattern  = f"{phase_meta['name']}_ep*.pt"
    episodes = int(train_cfg.get("eval_during_training_episodes", 100))
    device   = str(train_cfg.get("eval_during_training_device", "cpu"))
    out_dir  = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    # Clear any stale sentinel so the evaluator does not stop immediately.
    if os.path.exists(done_file):
        os.remove(done_file)

    log_path = os.path.join(out_dir, "parallel_eval.log")
    log_fh   = open(log_path, "w")
    cmd = [
        sys.executable, "eval_checkpoints.py", "--watch",
        "--config",    config_path,
        "--ckpt_dir",  train_cfg["checkpoint_dir"],
        "--pattern",   pattern,
        "--episodes",  str(episodes),
        "--phase",     str(phase_idx),
        "--device",    device,
        "--out_dir",   out_dir,
        "--done_file", done_file,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"[train] could not launch parallel evaluator: {e}")
        log_fh.close()
        return None

    print(f"[train] parallel evaluator started (pid {proc.pid}): watching "
          f"'{pattern}', {episodes} ep/ckpt on {device}.")
    print(f"[train] evaluator output -> {log_path}")
    return {"proc": proc, "log": log_fh, "log_path": log_path, "out_dir": out_dir}


def _finalize_parallel_evaluator(handle, done_file):
    """Signal training completion, wait for the evaluator, and echo the ranking."""
    if handle is None:
        return
    try:
        with open(done_file, "w") as f:
            f.write("done\n")
    except Exception as e:
        print(f"[train] could not write done-file: {e}")

    print("\n[train] training finished - waiting for the parallel evaluator to "
          "rank remaining checkpoints...")
    try:
        handle["proc"].wait()
    finally:
        handle["log"].close()

    print(f"[train] parallel evaluation complete. Full log: {handle['log_path']}")
    rankings = sorted(glob.glob(os.path.join(handle["out_dir"], "ranking_*.csv")))
    if rankings:
        print(f"[train] final ranking -> {rankings[-1]}")
        try:
            with open(rankings[-1]) as f:
                print(f.read())
        except Exception:
            pass


def train(config_path: str = "configs/default.yaml", resume_path: str = None,
          init_weights_path: str = None):
    """Run the curriculum training loop.

    Args:
        config_path: Path to the YAML config.
        resume_path: If given, RESUME an interrupted run from this checkpoint —
            restores weights, optimizer, LR schedule, epsilon, and the curriculum
            position, then continues. Mutually exclusive with ``init_weights_path``.
        init_weights_path: If given, WARM-START a fresh run: load only the network
            weights from this checkpoint; optimizer, schedule, epsilon, and the
            curriculum all start from scratch.
    """
    if resume_path is not None and init_weights_path is not None:
        raise ValueError("Pass only one of resume_path / init_weights_path.")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["env"]["seed"]
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_cfg = config["training"]
    global_dr = config["domain_randomization"]
    os.makedirs(train_cfg["log_dir"],        exist_ok=True)
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=train_cfg["log_dir"])

    # Optional parallel evaluator: ranks checkpoints as they are written.
    eval_handle = None
    done_file   = os.path.join(train_cfg["checkpoint_dir"], ".training_done")
    if train_cfg.get("eval_during_training", False):
        eval_handle = _spawn_parallel_evaluator(config, config_path, train_cfg, done_file)

    # Single source of truth for both grid size and normalisation denominators
    grid_size  = config["env"]["grid_size"]
    ctx_norm   = build_ctx_norm(config)

    max_agents     = max(p["n_agents"] for p in config["curriculum"])
    n_actions      = 4
    total_episodes = count_total_episodes(config)

    print(f"Total training episodes across all phases: {total_episodes:,}")
    print(f"CTX normalisation denominators (from DR max): {ctx_norm}")
    print(f"Fixed eval seeds: {_EVAL_SEEDS}")

    agent = DQNAgent(
        grid_size      = grid_size,
        n_actions      = n_actions,
        config         = config,
        device         = str(device),
        total_episodes = total_episodes,
    )

    agent.print_architecture()

    # ------------------------------------------------------------------
    # Checkpoint loading mode
    #   init_weights : warm-start a fresh run from given network weights
    #   resume       : continue an interrupted run (full state + position)
    # ------------------------------------------------------------------
    resume_state = None
    if init_weights_path is not None:
        if not os.path.isfile(init_weights_path):
            raise FileNotFoundError(f"init-weights checkpoint not found: {init_weights_path}")
        agent.load_weights_only(init_weights_path)
        print(f"Warm-start: loaded network weights from {init_weights_path} "
              f"(optimizer / LR schedule / epsilon / curriculum start fresh)")
    elif resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = agent.load(resume_path)
        if resume_state is None:
            print(f"Resumed agent state from {resume_path} (no training-position "
                  f"metadata found; curriculum restarts from the beginning)")
        else:
            print(f"Resumed from {resume_path}: phase {resume_state['phase_idx']}, "
                  f"ep_in_phase {resume_state['ep_in_phase']}, "
                  f"global_episode {resume_state['global_episode']}")

    if resume_state is not None:
        global_episode = resume_state.get("global_episode", 0)
        best_success   = resume_state.get("best_success", 0.0)
        best_reward    = resume_state.get("best_reward", float("-inf"))
        resume_phase   = resume_state.get("phase_idx", None)
        resume_ep      = resume_state.get("ep_in_phase", 0)
    else:
        global_episode = 0
        best_success   = 0.0
        best_reward    = float("-inf")
        resume_phase   = None
        resume_ep      = 0

    n_phases = len(config["curriculum"])
    print(f"Curriculum: {n_phases} phases  |  update_every={train_cfg.get('update_every', 4)}")
    print(
        f"Domain Randomisation: vision={global_dr['vision_radius']}  "
        f"comm={global_dr['comm_range']}  "
        f"agents={global_dr['n_agents']}  "
        f"grid=fixed {grid_size}x{grid_size}\n"
    )

    try:
        for phase_idx, phase_meta in enumerate(config["curriculum"], 1):
            n_ep = phase_meta["n_episodes"]
            if n_ep == 0:
                tqdm.write(f"[{phase_idx}/{n_phases}] {phase_meta['name']}  SKIP (n_episodes=0)")
                continue

            # --- Resume bookkeeping: skip completed phases, restart the active one
            if resume_phase is not None and phase_idx < resume_phase:
                tqdm.write(f"[{phase_idx}/{n_phases}] {phase_meta['name']}  SKIP (completed before resume)")
                continue
            if resume_phase is not None and phase_idx == resume_phase:
                start_ep        = resume_ep + 1
                apply_eps_reset = False   # keep epsilon restored from the checkpoint
                if start_ep > n_ep:
                    tqdm.write(f"[{phase_idx}/{n_phases}] {phase_meta['name']}  SKIP (already finished at resume)")
                    continue
            else:
                start_ep        = 1
                apply_eps_reset = True

            dr_cfg = get_dr_cfg(config, phase_meta)
            print(
                f"[{phase_idx}/{n_phases}] {phase_meta['name']}"
                f"  agents={phase_meta['n_agents']}"
                f"  grid=fixed {grid_size}x{grid_size}"
                f"  episodes={n_ep}  start_ep={start_ep}"
                f"  DR vision={dr_cfg['vision_radius']}"
            )
            phase_cfg = make_phase_config(config, phase_meta, max_agents)
            global_episode, best_success, best_reward = run_phase(
                phase_cfg, phase_meta, agent, writer,
                global_episode, best_success, best_reward,
                train_cfg, dr_cfg, device,
                ctx_norm=ctx_norm,
                grid_size=grid_size,
                phase_idx=phase_idx,
                start_ep=start_ep,
                apply_eps_reset=apply_eps_reset,
            )
    finally:
        # Always signal the evaluator (even on Ctrl-C / exception) so it ranks
        # what it has and exits instead of waiting on the sentinel forever.
        _finalize_parallel_evaluator(eval_handle, done_file)

    writer.close()
    print(f"\nDone. Best success rate: {best_success:.1%}  Best eval reward: {best_reward:.3f}")
    return agent


if __name__ == "__main__":
    train()
