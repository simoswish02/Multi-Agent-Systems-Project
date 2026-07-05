"""
main.py — Entry point for the Multi-Drone Search RL project.

Modes:
  train : run the DQN training loop with curriculum
  eval  : load a checkpoint and run evaluation episodes with GUI
  play  : interactive human-controlled episode (keyboard)

Usage:
  python main.py --mode train
  python main.py --mode train --resume checkpoints/best.pt
  python main.py --mode eval --checkpoint checkpoints/best.pt
  python main.py --mode eval --checkpoint checkpoints/best.pt --phase 0
  python main.py --mode play
"""

import argparse
import yaml
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Drone Search RL")
    parser.add_argument("--mode",       choices=["train", "eval", "play", "simulate"], default="train")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint for eval/play mode")
    parser.add_argument("--resume",     default=None,
                        help="Path to a .pt checkpoint to RESUME an interrupted training run: "
                             "restores weights, optimizer, LR schedule, epsilon and curriculum "
                             "position, then continues. Use checkpoints/last.pt.")
    parser.add_argument("--init-weights", dest="init_weights", default=None,
                        help="Path to a .pt checkpoint to WARM-START a fresh run: loads only the "
                             "network weights (instead of random init); optimizer, schedule, "
                             "epsilon and curriculum all start from scratch. Mutually exclusive "
                             "with --resume.")
    parser.add_argument("--phase",      type=int, default=-1,
                        help="Curriculum phase index for eval/play (default: last phase)")
    # Context overrides for eval/play (optional; defaults to config env values)
    parser.add_argument("--vision-radius", type=int, default=None,
                        help="Override vision_radius for eval/play")
    parser.add_argument("--comm-range",    type=int, default=None,
                        help="Override comm_range for eval/play")
    parser.add_argument("--n-agents",      type=int, default=None,
                        help="Override n_agents for eval/play")
    parser.add_argument("--fault-prob",    type=float, default=None,
                        help="Per-step drone fault probability for eval/play/simulate "
                             "(default: env.fault_prob from config)")
    parser.add_argument("--auto-nav",      action="store_true",
                        help="Eval/simulate only: once a drone knows the target, it "
                             "follows the BFS shortest path on its known map instead "
                             "of the network policy")
    return parser.parse_args()


def build_eval_config(config: dict, phase_idx: int = -1) -> dict:
    import copy
    cfg   = copy.deepcopy(config)
    phase = cfg["curriculum"][phase_idx]
    e     = cfg["env"]
    # grid_size stays unchanged: it is the single fixed global constant
    e["n_agents"]         = phase["n_agents"]
    e["obstacle_density"] = phase.get("obstacle_density", e["obstacle_density"])
    e["max_steps"]        = phase.get("max_steps",        e["max_steps"])
    return cfg


def build_agent(config: dict, device: str = "cpu"):
    """Instantiate DQNAgent from config.  grid_size is the only size parameter."""
    from agents.dqn_agent import DQNAgent
    return DQNAgent(
        grid_size = config["env"]["grid_size"],
        n_actions = 4,
        config    = config,
        device    = device,
    )


def _resolve_ctx_params(config: dict, env_cfg: dict, args) -> dict:
    """Return context parameter dict, applying CLI overrides when provided."""
    return {
        "vision_radius": args.vision_radius if args.vision_radius is not None
                         else env_cfg["vision_radius"],
        "comm_range":    args.comm_range    if args.comm_range    is not None
                         else env_cfg["comm_range"],
        "n_agents":      args.n_agents      if args.n_agents      is not None
                         else env_cfg["n_agents"],
    }


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(config: dict, checkpoint_path: str, args, phase_idx: int = -1):
    import torch
    from env.grid_env import DroneSearchEnv
    from env.utils import auto_nav_action
    from training.train import ctx_to_numpy, build_ctx_norm

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cfg  = build_eval_config(config, phase_idx)
    env       = DroneSearchEnv(eval_cfg, render_mode="human")
    agent     = build_agent(config, device=device)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0   # pure greedy during eval display
    print(f"Loaded checkpoint: {checkpoint_path}  |  device: {device}")

    grid_size  = config["env"]["grid_size"]
    ctx_norm   = build_ctx_norm(config)
    ctx_params = _resolve_ctx_params(config, eval_cfg["env"], args)
    fault_prob = args.fault_prob if args.fault_prob is not None \
        else eval_cfg["env"].get("fault_prob", 0.0)
    # Apply the context to the real env so the FOV / comm range / team size the
    # network is told about match what it actually observes.
    env.set_domain_params(**ctx_params, fault_prob=fault_prob)
    print(
        f"Context: vision_radius={ctx_params['vision_radius']}  "
        f"comm_range={ctx_params['comm_range']}  "
        f"n_agents={ctx_params['n_agents']}  "
        f"grid=fixed {grid_size}x{grid_size}  "
        f"fault_prob={fault_prob}  auto_nav={args.auto_nav}"
    )

    n_episodes = 10
    for ep in range(n_episodes):
        obs, info = env.reset()
        env.debug_nav_paths = {}   # drone idx -> planned path (GUI overlay)
        total_r   = 0.0
        done      = False

        while not done:
            env.render()
            for i in range(env.n_agents):
                if done:
                    break

                if not env.agent_alive[i]:
                    # Broken drone: no-op turn, the clock still ticks.
                    env.debug_nav_paths.pop(i, None)
                    obs_i, r, terminated, truncated, info = env.step_agent(i, 0)
                    obs[i] = obs_i
                    done   = terminated or truncated
                    continue

                action = None
                if args.auto_nav:
                    # Deterministic homing once this drone knows the target.
                    action, path = auto_nav_action(env, i)
                    if action is None:
                        env.debug_nav_paths.pop(i, None)
                    else:
                        env.debug_nav_paths[i] = path
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
                obs[i]  = obs_i
                total_r += r
                done    = terminated or truncated

        status   = "FOUND" if info["found"] else "TIMEOUT"
        n_broken = env.n_agents - int(env.agent_alive.sum())
        print(f"Ep {ep+1:2d}: {status} | steps={info['step']:3d} | "
              f"reward={total_r:.3f} | broken={n_broken}/{env.n_agents}")
    env.close()


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------

def run_play(config: dict, args, phase_idx: int = -1):
    """Keyboard-controlled drone 0 (arrow keys), others act randomly."""
    import random
    import torch
    import pygame
    from env.grid_env import DroneSearchEnv
    from training.train import ctx_to_numpy, build_ctx_norm

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    play_cfg = build_eval_config(config, phase_idx)
    env      = DroneSearchEnv(play_cfg, render_mode="human")

    grid_size  = config["env"]["grid_size"]
    ctx_norm   = build_ctx_norm(config)
    ctx_params = _resolve_ctx_params(config, play_cfg["env"], args)
    fault_prob = args.fault_prob if args.fault_prob is not None \
        else play_cfg["env"].get("fault_prob", 0.0)
    env.set_domain_params(**ctx_params, fault_prob=fault_prob)

    obs, info = env.reset()
    env.render()

    KEY_ACTION = {
        pygame.K_UP:    0,
        pygame.K_DOWN:  1,
        pygame.K_LEFT:  2,
        pygame.K_RIGHT: 3,
    }

    running = True
    done    = False
    total_r = 0.0

    while running:
        action0 = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action0 = KEY_ACTION.get(event.key)

        if action0 is None:
            continue

        if done:
            obs, info = env.reset()
            done    = False
            total_r = 0.0
            env.render()
            continue

        # Drone 0: keyboard-controlled
        if not env.agent_alive[0]:
            print("[broken] Drone 0 is down — its turns are no-ops now.")
        mask0 = env._get_action_mask(0)
        if env.agent_alive[0] and not mask0[action0]:
            print("[blocked] That direction is a wall or OOB.")
            continue

        obs_0, r0, terminated, truncated, info = env.step_agent(0, action0)
        obs[0]  = obs_0
        total_r += r0
        done     = terminated or truncated

        # Drones 1..n: random sequential moves
        if not done:
            for i in range(1, env.n_agents):
                mask_i  = env._get_action_mask(i)
                valid    = np.where(mask_i)[0]
                action_i = int(random.choice(valid))
                obs_i, ri, terminated, truncated, info = env.step_agent(i, action_i)
                obs[i]  = obs_i
                total_r += ri
                done     = terminated or truncated
                if done:
                    break

        env.render()

        if done:
            status = "FOUND! " if info["found"] else "Timeout"
            print(f"{status} — steps={info['step']}  reward={total_r:.3f}")
            print("Press an arrow key to restart.")

    env.close()


# ---------------------------------------------------------------------------
# Simulate (config screen -> N episodes -> repeat)
# ---------------------------------------------------------------------------

def run_simulate(config: dict, args):
    """Interactive setup screen, then run the chosen number of episodes with the
    live GUI; when they finish, reopen the setup screen. Loops until the setup
    window is closed. Spawn is forced to the top-left corner."""
    import copy
    import torch
    from env.grid_env import DroneSearchEnv
    from env.utils import auto_nav_action
    from training.train import ctx_to_numpy, build_ctx_norm
    from gui.config_screen import ConfigScreen

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = config["env"]["grid_size"]
    ctx_norm  = build_ctx_norm(config)

    while True:
        sel = ConfigScreen(config).run()
        if sel is None:
            break   # setup window closed -> leave simulate mode

        fault_prob = sel.get(
            "fault_prob",
            args.fault_prob if args.fault_prob is not None
            else config["env"].get("fault_prob", 0.0),
        )
        auto_nav = sel.get("auto_nav", args.auto_nav)

        print(
            f"Simulate: agents={sel['n_agents']}  vision={sel['vision_radius']}  "
            f"comm={sel['comm_range']}  max_steps={sel['max_steps']}  "
            f"episodes={sel['episodes']}  fault_prob={fault_prob}  "
            f"auto_nav={auto_nav}  weights={sel['weights']}"
        )

        cfg = copy.deepcopy(config)
        cfg["env"]["n_agents"]     = sel["n_agents"]
        cfg["env"]["max_steps"]    = sel["max_steps"]
        cfg["env"]["spawn_corner"] = 0          # force top-left

        ctx_params = {
            "vision_radius": sel["vision_radius"],
            "comm_range":    sel["comm_range"],
            "n_agents":      sel["n_agents"],
        }

        env = None
        try:
            env = DroneSearchEnv(cfg, render_mode="human")
            env.set_domain_params(**ctx_params, fault_prob=fault_prob)
            agent = build_agent(config, device=device)
            agent.load(sel["weights"])
            agent.epsilon = 0.0

            for ep in range(sel["episodes"]):
                obs, info = env.reset()
                env.debug_nav_paths = {}   # drone idx -> path (GUI overlay)
                done, total_r = False, 0.0
                while not done:
                    env.render()
                    if getattr(env.renderer, "_closed", False):
                        break
                    for i in range(env.n_agents):
                        if done:
                            break

                        if not env.agent_alive[i]:
                            # Broken drone: no-op turn, the clock still ticks.
                            env.debug_nav_paths.pop(i, None)
                            obs_i, r, terminated, truncated, info = \
                                env.step_agent(i, 0)
                            obs[i] = obs_i
                            done   = terminated or truncated
                            continue

                        action = None
                        if auto_nav:
                            action, path = auto_nav_action(env, i)
                            if action is None:
                                env.debug_nav_paths.pop(i, None)
                            else:
                                env.debug_nav_paths[i] = path
                        if action is None:
                            ctx_np = ctx_to_numpy(
                                ctx_params, ctx_norm, i, env.n_alive_belief(i)
                            )
                            ctx_t  = torch.tensor(
                                ctx_np, dtype=torch.float32,
                                device=torch.device(device)
                            ).unsqueeze(0)
                            mask   = env._get_action_mask(i)
                            action = agent.select_action(
                                obs[i]["global"], obs[i]["local"], ctx_t,
                                greedy=True, action_mask=mask,
                            )
                        obs_i, r, terminated, truncated, info = env.step_agent(i, action)
                        obs[i]   = obs_i
                        total_r += r
                        done     = terminated or truncated
                if getattr(env.renderer, "_closed", False):
                    break
                status   = "FOUND" if info["found"] else "TIMEOUT"
                n_broken = env.n_agents - int(env.agent_alive.sum())
                print(f"  Ep {ep + 1:2d}/{sel['episodes']}: {status}  "
                      f"steps={info['step']:3d}  reward={total_r:.2f}  "
                      f"broken={n_broken}/{env.n_agents}")
        except Exception as e:   # keep the setup loop alive on a bad run
            print(f"[simulate] run aborted: {e}")
        finally:
            if env is not None:
                env.close()

    print("Simulate mode closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        if args.resume is not None and args.init_weights is not None:
            raise SystemExit("Use only one of --resume / --init-weights, not both.")
        from training.train import train
        train(
            args.config,
            resume_path=args.resume,
            init_weights_path=args.init_weights,
        )

    elif args.mode == "eval":
        assert args.checkpoint, "Provide --checkpoint for eval mode"
        run_eval(config, args.checkpoint, args, phase_idx=args.phase)

    elif args.mode == "play":
        run_play(config, args, phase_idx=args.phase)

    elif args.mode == "simulate":
        run_simulate(config, args)
