"""
replay_buffer.py — Experience replay for the DQN agent.

Isolated from the agent so it can be unit-tested independently and swapped for a
prioritized variant later.
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer of (n-step) DQN transitions.

    Each transition is ``(obs_global, obs_local, ctx, action, n_step_return,
    next_obs_global, next_obs_local, next_ctx, bootstrap_disc)``, where
    ``n_step_return`` is the discounted return over the window and
    ``bootstrap_disc`` is ``gamma^n`` (or ``0`` at a terminal). For one-step DQN
    these reduce to the reward and ``gamma``/``0``. Backed by a
    ``deque(maxlen=capacity)`` so the oldest transition is evicted on overflow.

    Args:
        capacity: Maximum number of transitions retained.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs_global, obs_local, ctx,
        action, reward,
        next_obs_global, next_obs_local, next_ctx,
        done,
    ):
        """Append one transition."""
        self.buffer.append((
            obs_global, obs_local, ctx,
            action, reward,
            next_obs_global, next_obs_local, next_ctx,
            done,
        ))

    def sample(self, batch_size: int):
        """Draw ``batch_size`` transitions i.i.d. as stacked NumPy arrays.

        Args:
            batch_size: Number of transitions to sample (must be ``<= len(self)``).

        Returns:
            Tuple of arrays ``(og, ol, ct, ac, rw, nog, nol, nct, dn)`` with
            dtypes float32 except ``ac`` (int64) and ``dn`` (float32).
        """
        batch = random.sample(self.buffer, batch_size)
        og, ol, ct, ac, rw, nog, nol, nct, dn = zip(*batch)
        return (
            np.array(og,  dtype=np.float32),
            np.array(ol,  dtype=np.float32),
            np.array(ct,  dtype=np.float32),
            np.array(ac,  dtype=np.int64),
            np.array(rw,  dtype=np.float32),
            np.array(nog, dtype=np.float32),
            np.array(nol, dtype=np.float32),
            np.array(nct, dtype=np.float32),
            np.array(dn,  dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
