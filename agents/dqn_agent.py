"""
dqn_agent.py — DQN agent: policy, action selection, training step, persistence.

Network definitions live in ``agents/networks.py``, the replay buffer in
``agents/replay_buffer.py``, and the LR schedule in ``agents/scheduler_utils.py``.
This module wires them together behind the :class:`~agents.base_agent.BaseAgent`
interface consumed by ``training/train.py``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from agents.base_agent import BaseAgent
from agents.replay_buffer import ReplayBuffer
from agents.scheduler_utils import EpisodeLRScheduler
from agents.networks import (
    CnnQNetwork,
    LocalCNN,
    GLOBAL_CHANNELS,
    LOCAL_CHANNELS,
    LOCAL_PATCH_SIZE,
    CTX_DIM,
)


class DQNAgent(BaseAgent):
    """Dueling Double DQN with parameter sharing across all drones.

    A single :class:`~agents.networks.CnnQNetwork` drives every drone. Action
    selection batches all agents through one forward pass; learning uses Double
    DQN targets, a Huber loss, gradient clipping, and a hard target sync every
    ``target_update_freq`` gradient steps.

    Args:
        grid_size: Side length of the (square) map.
        n_actions: Size of the discrete action space.
        config: Full parsed config dict (reads the ``agent`` section).
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        total_episodes: Total episodes across all phases; needed to size the LR
            schedule. ``None`` (eval/play) disables the scheduler.

    Raises:
        ValueError: On incoherent hyperparameters (non-positive sizes,
            ``batch_size > buffer_size``, or ``gamma`` outside ``[0, 1]``).
    """

    def __init__(
        self,
        grid_size:       int,
        n_actions:       int,
        config:          dict,
        device:          str = "cpu",
        total_episodes:  int = None,
    ):
        agent_cfg = config["agent"]
        lr_cfg    = agent_cfg.get("lr_scheduler", {})

        # --- Input validation: fail fast on incoherent hyperparameters ------
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        if agent_cfg["batch_size"] > agent_cfg["buffer_size"]:
            raise ValueError(
                f"batch_size ({agent_cfg['batch_size']}) cannot exceed "
                f"buffer_size ({agent_cfg['buffer_size']})"
            )
        if not 0.0 <= agent_cfg["gamma"] <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {agent_cfg['gamma']}")
        # --------------------------------------------------------------------

        self.grid_size          = grid_size
        self.n_actions          = n_actions
        self.gamma              = agent_cfg["gamma"]
        self.n_step             = agent_cfg.get("n_step", 1)
        self.batch_size         = agent_cfg["batch_size"]
        self.target_update_freq = agent_cfg["target_update_freq"]
        self.grad_clip          = agent_cfg.get("grad_clip", 1.0)
        self.device             = torch.device(device)

        self.epsilon       = agent_cfg["epsilon_start"]
        self.epsilon_end   = agent_cfg["epsilon_end"]
        self.epsilon_decay = agent_cfg["epsilon_decay"]

        net_kwargs = dict(
            grid_size = grid_size,
            n_actions = n_actions,
            fc_hidden = agent_cfg.get("fc_hidden", 1024),
        )
        self.q_net      = CnnQNetwork(**net_kwargs).to(self.device)
        self.target_net = CnnQNetwork(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self._base_lr = agent_cfg["lr"]
        self._max_lr  = lr_cfg.get("max_lr", agent_cfg["lr"])

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr           = self._base_lr,
            weight_decay = agent_cfg.get("weight_decay", 0.0),
        )

        # Per-episode warmup -> cosine schedule (only when sized for training).
        self.lr_scheduler = None
        if (
            lr_cfg.get("use_episode_scheduler", False)
            and total_episodes is not None
            and total_episodes > 0
        ):
            self.lr_scheduler = EpisodeLRScheduler(
                self.optimizer,
                base_lr        = self._base_lr,
                max_lr         = self._max_lr,
                min_lr         = lr_cfg.get("min_lr", 1e-6),
                total_episodes = total_episodes,
                warmup_pct     = lr_cfg.get("warmup_pct", 0.05),
            )

        self.buffer       = ReplayBuffer(agent_cfg["buffer_size"])
        self.train_steps  = 0
        self.loss_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the online network."""
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def print_architecture(self):
        """Print a human-readable summary of the network and LR schedule."""
        total    = self.count_parameters()
        comb_dim = self.q_net._combined_dim
        gs       = self.q_net.grid_size
        g_out    = self.q_net.global_cnn.out_dim
        l_out    = self.q_net.local_cnn.out_dim
        P        = LOCAL_PATCH_SIZE
        flat_dim = LocalCNN.PROJ_CH * P * P
        print("\n" + "=" * 72)
        print("  CnnQNetwork  —  GlobalMapCNN (ConvNeXt+attn) + LocalCNN (ConvNeXt) + ctx")
        print("  Dueling Double DQN")
        print("=" * 72)
        for name, module in self.q_net.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:<28s}  {params:>10,} params")
        print("-" * 72)
        print(f"  {'TOTAL':<28s}  {total:>10,} params")
        h1       = self.q_net.fc1[0].out_features      # fc_hidden (from config)
        print(f"  global channels  :  {GLOBAL_CHANNELS}  [Visited, Obstacle, Recency, Target, Own_Position, Broken]")
        print(f"  local  channels  :  {LOCAL_CHANNELS}  [Visited, Obstacle, Recency, Target, Other_Pos]")
        print(f"  local patch size :  {P}x{P}  (covers max vision_radius=6: 2*6+1=13)")
        print(f"  grid size        :  {gs}x{gs}  (from config — no hardcoded constant)")
        print(f"  context dim      :  {CTX_DIM}  [vision_radius, comm_range, n_agents, agent_id, n_alive_belief]")
        print(f"  own position     :  Own_Position global channel (argmax -> local-patch crop)")
        print(f"  GlobalMapCNN     :  ConvNeXt stages (32->16->8) + FiLM(ctx)")
        print(f"                      -> MHSA over 8x8 tokens (+CLS,+ctx) -> {g_out}-dim")
        print(f"  LocalCNN         :  ConvNeXt blocks (dw 7x7, d=1,2,1) -> GRN -> FiLM(global)")
        print(f"                      -> 1x1conv({LocalCNN.BLOCK_CH}->{LocalCNN.PROJ_CH}) -> flatten({flat_dim}) -> Linear->{l_out}")
        print(f"                      no stride-2: full {P}x{P} resolution preserved")
        print(f"  local  out       :  {l_out}-dim")
        print(f"  combined dim     :  {comb_dim}  (global {g_out} + local {l_out} + ctx {CTX_DIM})")
        print(f"  FC head          :  {comb_dim}->{h1}->{h1 // 2}->{h1 // 4}->{{V,A}}  (LayerNorm after each linear)")
        if self.lr_scheduler is not None:
            warmup_name = type(self.lr_scheduler.warmup).__name__
            cosine_name = type(self.lr_scheduler.cosine).__name__
            warmup_eps  = self.lr_scheduler.warmup_eps
        else:
            warmup_name = cosine_name = "None"
            warmup_eps  = 0
        print(f"  lr scheduler     :  warmup={warmup_name} -> {cosine_name}")
        print(f"  base_lr / max_lr :  {self._base_lr:.2e} / {self._max_lr:.2e}")
        print(f"  warmup episodes  :  {warmup_eps}")
        print(f"  grad clip        :  {self.grad_clip}")
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Action selection — single agent
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs_global:  np.ndarray,
        obs_local:   np.ndarray,
        ctx:         torch.Tensor,
        greedy:      bool       = False,
        action_mask: np.ndarray = None,
    ) -> int:
        """Return one action index for a single agent (ε-greedy unless ``greedy``).

        The drone position is carried inside ``obs_global`` (Own_Position
        channel); the network recovers the crop coordinates from it, so no
        position argument is needed.
        """
        valid = np.where(action_mask)[0] if action_mask is not None else np.arange(self.n_actions)
        if not greedy and random.random() < self.epsilon:
            return int(random.choice(valid))

        og_t  = torch.tensor(obs_global, dtype=torch.float32, device=self.device).unsqueeze(0)
        ol_t  = torch.tensor(obs_local,  dtype=torch.float32, device=self.device).unsqueeze(0)

        # BatchNorm must use running stats (not noisy per-batch stats) during
        # inference, and must not update its running averages. Toggle eval()
        # around the forward, then restore the previous mode.
        was_training = self.q_net.training
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(og_t, ol_t, ctx)[0].cpu().numpy()
        self.q_net.train(was_training)
        if action_mask is not None:
            q[~action_mask] = -np.inf
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # Action selection — batched
    # ------------------------------------------------------------------

    def select_actions_batch(
        self,
        obs_list:  list,
        ctx_nps:   list,
        masks:     list,
    ) -> list:
        """Select an action for every agent in a single shared forward pass.

        Each agent's position is carried inside its ``obs["global"]``
        (Own_Position channel), so no separate positions list is needed.
        """
        n = len(obs_list)
        explore = [random.random() < self.epsilon for _ in range(n)]
        greedy_idx = [i for i, e in enumerate(explore) if not e]

        # If no agent is acting greedily (all exploring), there is nothing to run
        # through the network. Return random valid actions immediately — this also
        # guards against building a zero-size batch (np.stack on an empty list
        # raises, and a 0-length batch is meaningless to the network).
        if not greedy_idx:
            actions = []
            for i in range(n):
                mask  = masks[i]
                valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
                actions.append(int(random.choice(valid)))
            return actions

        og_batch  = torch.tensor(
            np.stack([obs_list[i]["global"] for i in greedy_idx], axis=0),
            dtype=torch.float32, device=self.device,
        )
        ol_batch  = torch.tensor(
            np.stack([obs_list[i]["local"] for i in greedy_idx], axis=0),
            dtype=torch.float32, device=self.device,
        )
        ctx_batch = torch.tensor(
            np.stack([ctx_nps[i] for i in greedy_idx], axis=0),
            dtype=torch.float32, device=self.device,
        )

        # Use BatchNorm running stats for inference (see select_action).
        was_training = self.q_net.training
        self.q_net.eval()
        with torch.no_grad():
            q_all = self.q_net(og_batch, ol_batch, ctx_batch).cpu().numpy()
        self.q_net.train(was_training)

        greedy_actions = {}
        for k, i in enumerate(greedy_idx):
            q = q_all[k].copy()
            if masks[i] is not None:
                q[~masks[i]] = -np.inf
            greedy_actions[i] = int(np.argmax(q))

        actions = []
        for i in range(n):
            if explore[i]:
                mask  = masks[i]
                valid = np.where(mask)[0] if mask is not None else np.arange(self.n_actions)
                actions.append(int(random.choice(valid)))
            else:
                actions.append(greedy_actions[i])
        return actions

    # ------------------------------------------------------------------
    # Replay buffer interface
    # ------------------------------------------------------------------

    def push_transition(
        self,
        obs_global, obs_local, ctx_np,
        action, n_step_return,
        next_obs_global, next_obs_local, next_ctx_np,
        bootstrap_disc,
    ):
        """Store one (n-step) transition in the replay buffer.

        Args:
            n_step_return: Discounted n-step return ``sum_k gamma^k r_{t+k}``.
            bootstrap_disc: ``gamma^n`` for bootstrapping the next-state value,
                or ``0.0`` if the window reached a terminal state. Built by
                :class:`agents.nstep.NStepBuffer`.
        """
        self.buffer.push(
            obs_global, obs_local, ctx_np,
            action, n_step_return,
            next_obs_global, next_obs_local, next_ctx_np,
            bootstrap_disc,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self):
        """Run one Double-DQN gradient step. Returns the loss, or ``None``."""
        if len(self.buffer) < self.batch_size:
            return None

        og, ol, ct, ac, rw, nog, nol, nct, disc = self.buffer.sample(self.batch_size)

        og_t   = torch.from_numpy(og).to(self.device)
        ol_t   = torch.from_numpy(ol).to(self.device)
        ct_t   = torch.from_numpy(ct).to(self.device)
        ac_t   = torch.from_numpy(ac).to(self.device)
        rw_t   = torch.from_numpy(rw).to(self.device)
        nog_t  = torch.from_numpy(nog).to(self.device)
        nol_t  = torch.from_numpy(nol).to(self.device)
        nct_t  = torch.from_numpy(nct).to(self.device)
        disc_t = torch.from_numpy(disc).to(self.device)

        q_vals = self.q_net(og_t, ol_t, ct_t).gather(1, ac_t.unsqueeze(1)).squeeze(1)

        # n-step Double-DQN target. `rw` is the (already discounted) n-step
        # return and `disc` is gamma^n (or 0 when the window hit a terminal),
        # so gamma and the done-mask are folded in — for n_step=1 this reduces
        # to r + gamma * Q(s') * (1 - terminated).
        with torch.no_grad():
            best_actions = self.q_net(nog_t, nol_t, nct_t).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(nog_t, nol_t, nct_t).gather(1, best_actions).squeeze(1)
            targets      = rw_t + disc_t * next_q

        loss = F.smooth_l1_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        val = loss.item()
        self.loss_history.append(val)
        return val

    # ------------------------------------------------------------------
    # Scheduler / epsilon
    # ------------------------------------------------------------------

    def step_episode_scheduler(self):
        """Advance the per-episode LR schedule (no-op if disabled)."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def current_lr(self) -> float:
        """Return the current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def decay_epsilon(self):
        """Multiplicatively decay ε once per episode, floored at ``epsilon_end``."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, training_state: dict = None):
        """Serialize agent state to ``path``.

        Args:
            path: Destination ``.pt`` file.
            training_state: Optional dict of training-loop progress
                (``global_episode``, ``phase_idx``, ``ep_in_phase``,
                ``best_success``, ``best_reward``). Written under the
                ``training_state`` key so a later ``--resume`` can continue the
                curriculum from where it stopped. Omit it for plain weight
                checkpoints (eval/best).

        Raises:
            OSError: If the file cannot be written.
        """
        ckpt = {
            "q_net":         self.q_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "warmup_sched":  self.lr_scheduler.warmup.state_dict() if self.lr_scheduler else None,
            "cosine_sched":  self.lr_scheduler.cosine.state_dict() if self.lr_scheduler else None,
            "sched_episode": self.lr_scheduler.episode if self.lr_scheduler else 0,
            "epsilon":       self.epsilon,
            "train_steps":   self.train_steps,
        }
        if training_state is not None:
            ckpt["training_state"] = training_state
        try:
            torch.save(ckpt, path)
        except OSError as e:
            raise OSError(f"Failed to save checkpoint to '{path}': {e}") from e

    def _read_checkpoint(self, path: str) -> dict:
        """Load a raw checkpoint dict from ``path`` with clear errors."""
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint not found: '{path}'") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{path}': {e}") from e

    def load(self, path: str) -> dict:
        """Full restore for **resuming** an interrupted training run.

        Restores network weights, optimizer state, LR schedule, ε, and the
        gradient-step counter, so training continues seamlessly.

        Args:
            path: Source ``.pt`` checkpoint.

        Returns:
            The stored ``training_state`` dict if present (so the caller can
            resume the curriculum position), else ``None``.
        """
        ckpt = self._read_checkpoint(path)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict({
                "warmup":  ckpt.get("warmup_sched"),
                "cosine":  ckpt.get("cosine_sched"),
                "episode": ckpt.get("sched_episode", 0),
            })
        self.epsilon     = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        return ckpt.get("training_state")

    def load_weights_only(self, path: str):
        """Warm-start: load **only** the network weights, leave everything fresh.

        Use this to start a brand-new training run from pretrained weights
        instead of random initialisation. The optimizer, LR schedule, ε, replay
        buffer, and curriculum position are all left at their fresh/initial
        values.

        Args:
            path: Source ``.pt`` checkpoint to copy ``q_net`` weights from.
        """
        ckpt = self._read_checkpoint(path)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
