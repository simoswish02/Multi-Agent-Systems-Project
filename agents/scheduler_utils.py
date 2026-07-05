"""
scheduler_utils.py — Per-episode learning-rate schedule.

Encapsulates the two-stage schedule used during training: a linear warmup from
``base_lr`` to ``max_lr`` over the first ``warmup_pct`` of episodes, followed by
cosine annealing down to ``min_lr``. The schedule is advanced once per *episode*
(not per gradient step).
"""

import torch


class EpisodeLRScheduler:
    """Warmup-then-cosine LR schedule stepped once per episode.

    Args:
        optimizer: The optimizer whose single param-group LR is driven.
        base_lr: LR at episode 0, before any warmup step (the optimizer's
            starting LR).
        max_lr: Peak LR reached at the end of warmup.
        min_lr: Floor LR at the end of cosine annealing.
        total_episodes: Total episodes across all phases; sizes both stages.
        warmup_pct: Fraction of ``total_episodes`` used for the linear warmup.

    Note:
        ``SequentialLR`` is deliberately avoided: it pre-steps its sub-schedulers
        during construction, which skips the first LR value. Instead a
        :class:`~torch.optim.lr_scheduler.LambdaLR` warmup and a
        :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` are switched
        manually. The warmup factor is ``max_lr / base_lr`` (LambdaLR has no
        ``end_factor <= 1`` constraint, unlike ``LinearLR``). The cosine stage
        starts cleanly from ``max_lr`` because PyTorch's chainable ``get_lr``
        advances relative to the *current* param-group LR, and the LR is set to
        ``max_lr`` at the transition.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        min_lr: float,
        total_episodes: int,
        warmup_pct: float = 0.05,
    ):
        self.optimizer = optimizer
        self.base_lr   = base_lr
        self.max_lr    = max_lr
        self.warmup_eps = max(1, int(total_episodes * warmup_pct))
        cosine_episodes = total_episodes - self.warmup_eps
        ratio = max_lr / base_lr

        def warmup_lambda(ep):
            # ep=0 -> 1.0 (base_lr); ep=warmup_eps -> ratio (max_lr)
            return 1.0 + (ratio - 1.0) * ep / self.warmup_eps

        self.warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cosine_episodes), eta_min=min_lr,
        )
        # Ensure training starts exactly at base_lr.
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        self.episode = 0   # number of step() calls so far

    def step(self):
        """Advance the schedule by one episode."""
        if self.episode < self.warmup_eps:
            self.warmup.step()
        else:
            if self.episode == self.warmup_eps:
                # Transition: set LR to exactly max_lr so cosine starts there.
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.max_lr
            self.cosine.step()
        self.episode += 1

    def current_lr(self) -> float:
        """Return the current optimizer LR."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        """Serialize warmup, cosine, and episode-counter state."""
        return {
            "warmup":  self.warmup.state_dict(),
            "cosine":  self.cosine.state_dict(),
            "episode": self.episode,
        }

    def load_state_dict(self, sd: dict):
        """Restore from a :meth:`state_dict` payload (tolerant of legacy keys)."""
        if sd.get("warmup") is not None:
            self.warmup.load_state_dict(sd["warmup"])
        if sd.get("cosine") is not None:
            self.cosine.load_state_dict(sd["cosine"])
        self.episode = sd.get("episode", 0)
