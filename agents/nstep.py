"""
nstep.py — Per-agent n-step return accumulation for the sequential env.

In this project drones act in round-robin (one ``step_agent`` call each per
round), so an agent's "next n steps" are n rounds away. This buffer keeps a
per-agent sliding window of single-step records and emits n-step transitions
suitable for the replay buffer.

Emitted transition layout (matches ``ReplayBuffer.push`` / ``DQNAgent.update``):
    (obs_global, obs_local, ctx, action, n_step_return,
     next_obs_global, next_obs_local, next_ctx, bootstrap_discount)

where ``n_step_return = sum_{k=0}^{m-1} gamma^k r_{t+k}`` (truncated at a terminal
state) and ``bootstrap_discount = gamma^m`` if the window did not hit a terminal,
else ``0.0``. The TD target in the agent is therefore simply
``R + bootstrap_discount * Q(next_state)`` — gamma and the done-mask are folded
into the stored values, so 1-step (``n_step=1``) reduces exactly to the standard
target ``r + gamma * Q(s') * (1 - terminated)``.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class _Step:
    """One single-step record held in the sliding window."""
    og: object
    ol: object
    ctx: object
    action: int
    reward: float
    nog: object
    nol: object
    nctx: object
    terminated: bool


class NStepBuffer:
    """Accumulates n-step returns independently for each agent.

    Args:
        n_agents: Number of drones this episode.
        n_step: Number of steps to accumulate (``1`` = standard one-step TD).
        gamma: Discount factor.
    """

    def __init__(self, n_agents: int, n_step: int, gamma: float):
        self.n_step = max(1, int(n_step))
        self.gamma  = gamma
        self._dq    = [deque() for _ in range(n_agents)]

    def push(self, agent_i, og, ol, ctx, action, reward, nog, nol, nctx, terminated):
        """Record one step for ``agent_i``; return any emitted n-step transitions.

        Returns:
            A list with the single emitted transition once the window is full,
            else an empty list.
        """
        self._dq[agent_i].append(
            _Step(og, ol, ctx, action, reward, nog, nol, nctx, bool(terminated))
        )
        if len(self._dq[agent_i]) >= self.n_step:
            return [self._emit(agent_i)]
        return []

    def flush(self, agent_i):
        """Emit all remaining (shorter-than-n) transitions for ``agent_i``.

        Call at episode end so the tail of each trajectory is not lost.
        """
        out = []
        while self._dq[agent_i]:
            out.append(self._emit(agent_i))
        return out

    def apply_team_terminal(self, finder: int, bonus: float):
        """Share a terminal reward across the team at episode success.

        Adds ``bonus`` to the most recent pending step of every agent except the
        ``finder`` (who already received the env terminal reward), and marks the
        most recent step of every agent terminal so its window stops
        bootstrapping. Requires ``n_step >= 2`` so the terminal-round steps are
        still in the windows (with ``n_step == 1`` every step is emitted
        immediately and nothing remains to credit).

        Args:
            finder: Index of the agent that reached the target.
            bonus: Reward to add to each non-finder agent's last pending step.
        """
        for i, dq in enumerate(self._dq):
            if not dq:
                continue
            dq[-1].terminated = True
            if i != finder:
                dq[-1].reward += bonus

    def _emit(self, agent_i):
        dq = self._dq[agent_i]
        R = 0.0
        g = 1.0
        steps = 0
        bootstrap = True
        last = dq[0]
        for s in dq:
            R += g * s.reward
            g *= self.gamma
            steps += 1
            last = s
            if s.terminated:
                bootstrap = False
                break
        first = dq[0]
        disc = (self.gamma ** steps) if bootstrap else 0.0
        dq.popleft()
        return (
            first.og, first.ol, first.ctx, first.action, R,
            last.nog, last.nol, last.nctx, disc,
        )
