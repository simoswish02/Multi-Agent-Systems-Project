"""
base_agent.py — Abstract agent interface.

Defines the contract the training loop (training/train.py) relies on, so that
alternative algorithms (e.g. Dueling DQN, Rainbow, a policy-gradient method)
can be swapped in without modifying the loop. Any concrete agent must implement
action selection, a learning step, epsilon/schedule bookkeeping, and
checkpoint persistence.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Interface contract for agents consumed by the training loop.

    The loop calls, per agent-step: ``select_actions_batch`` (or
    ``select_action``), ``push_transition``, and periodically ``update``.
    Per episode it calls ``step_episode_scheduler`` and ``decay_epsilon``.
    Checkpointing uses ``save`` / ``load``.

    Note:
        ``select_actions_batch`` exists because the environment runs N drones
        through a single shared network; batching their forward passes is the
        hot path. ``select_action`` is the single-agent convenience used by
        eval/play.
    """

    @abstractmethod
    def select_action(self, obs_global, obs_local, ctx, agent_pos,
                      greedy: bool = False, action_mask=None) -> int:
        """Return a single action index for one agent."""

    @abstractmethod
    def select_actions_batch(self, obs_list, ctx_nps, positions, masks) -> list:
        """Return one action index per agent in a single forward pass."""

    @abstractmethod
    def push_transition(self, obs_global, obs_local, ctx_np, action, reward,
                        next_obs_global, next_obs_local, next_ctx_np, done):
        """Store one transition in the replay buffer."""

    @abstractmethod
    def update(self):
        """Run one gradient update. Returns the loss value or ``None``."""

    @abstractmethod
    def step_episode_scheduler(self):
        """Advance the LR schedule by one episode."""

    @abstractmethod
    def decay_epsilon(self):
        """Decay the exploration rate by one episode."""

    @abstractmethod
    def save(self, path: str):
        """Serialize agent state to ``path``."""

    @abstractmethod
    def load(self, path: str):
        """Restore agent state from ``path``."""
