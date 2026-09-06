"""Regenerate the two training-side figures of the report.

`testing/analysis.ipynb` owns every figure that is derived from the evaluation
CSVs in `testing_results/`. The two figures below come from a different source
and so live here:

  fig_training_curves.png   per-episode telemetry of the final run, read from
                            the TensorBoard event file (report Section 5).
  fig_schedules.png         the epsilon and learning-rate schedules, recomputed
                            analytically from `configs/default.yaml`
                            (report Appendix C).

Only `fig_schedules` is reproducible from the repository alone: the training
telemetry needs the run's TensorBoard event file, which is far too large to
track. The final run was executed as a chain of SLURM jobs and only the first
job's event file survives, covering episodes 1-30,169; that is the tranche the
report shows, and every logged quantity has flattened long before it ends.

Usage
-----
    python testing/make_training_figures.py                # both figures
    python testing/make_training_figures.py --only schedules

Writes to `testing_results/plots/` and mirrors into `report/figures/`, matching
the notebook's convention.

Note on units: TensorBoard's `episode_length` is `info["step"]`, which counts
*agent-turns* (the budget is `max_steps * n_agents` and a wreck's no-op turn
counts too). Panel (c) is therefore labelled in agent-turns, not rounds.
"""

import argparse
import glob
import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

# --- paths -----------------------------------------------------------------

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR  = os.path.join(ROOT, "runs", "mas_50k_dr_faults")
PLOT_DIR = os.path.join(ROOT, "testing_results", "plots")
FIG_DIR  = os.path.join(ROOT, "report", "figures")
CONFIG   = os.path.join(ROOT, "configs", "default.yaml")

# --- style (kept in sync with testing/analysis.ipynb) -----------------------

mpl.rcParams.update({
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.35,
    "lines.linewidth":  1.7,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})

C_MAIN = "#1f4e79"
C_EVAL = "#c0392b"
C_ALT  = "#2e8b57"

MA_WINDOW = 500          # episodes, the report's stated smoothing window


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, name)
    fig.savefig(out)
    shutil.copy2(out, os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("  saved", os.path.relpath(out, ROOT).replace(os.sep, "/"),
          "(mirrored into report/figures/)")


# ---------------------------------------------------------------------------
# (1) training telemetry
# ---------------------------------------------------------------------------

def _load_scalars():
    """Return {tag: (steps, values)} from the largest event file of the run."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    files = glob.glob(os.path.join(RUN_DIR, "events.out.tfevents.*"))
    if not files:
        raise SystemExit(
            f"No TensorBoard event file under {os.path.relpath(RUN_DIR, ROOT)}.\n"
            "The training telemetry figure needs the run's log, which is not "
            "tracked in the repository (see this file's docstring). Run\n"
            "  python testing/make_training_figures.py --only schedules\n"
            "to regenerate the reproducible figure alone."
        )
    # Several jobs of the chain may have written here; take the substantial one.
    path = max(files, key=os.path.getsize)
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        rows = ea.Scalars(tag)
        out[tag.split("/")[-1]] = (
            np.array([r.step for r in rows], dtype=float),
            np.array([r.value for r in rows], dtype=float),
        )
    return out


def _rolling(x, w, fn):
    """Rolling statistic over a window of w samples, aligned to the window end."""
    if len(x) < w:
        return np.array([]), np.array([])
    view = np.lib.stride_tricks.sliding_window_view(x, w)
    return np.arange(w, len(x) + 1, dtype=float), fn(view, axis=1)


def _panel(ax, steps, values, ylabel, title, band=True, color=C_MAIN):
    idx, mean = _rolling(values, MA_WINDOW, np.mean)
    ax.plot(steps[MA_WINDOW - 1:], mean, color=color, label=f"mean ({MA_WINDOW} ep.)")
    if band:
        _, q1 = _rolling(values, MA_WINDOW, lambda v, axis: np.percentile(v, 25, axis=axis))
        _, q3 = _rolling(values, MA_WINDOW, lambda v, axis: np.percentile(v, 75, axis=axis))
        ax.fill_between(steps[MA_WINDOW - 1:], q1, q3, color=color, alpha=0.22,
                        lw=0, label="interquartile range")
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_training_curves(eps_floor_ep):
    s = _load_scalars()
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.3))

    st, rew = s["episode_reward"]
    _panel(axs[0], st, rew, "return", "(a) episode return")
    axs[0].axhline(0, ls=":", lw=0.9, color="0.35")

    st, found = s["found_target"]
    idx, mean = _rolling(found * 100, MA_WINDOW, np.mean)
    axs[1].plot(st[MA_WINDOW - 1:], mean, color=C_ALT,
                label=f"training ({MA_WINDOW} ep.)")
    if "eval_success_rate" in s:
        est, ev = s["eval_success_rate"]
        axs[1].plot(est, ev * 100, "--o", ms=4, color=C_EVAL,
                    label=r"near-greedy eval ($\varepsilon = 0.05$, 20 ep.)")
    axs[1].set_ylim(0, 100)
    axs[1].set_xlabel("episode")
    axs[1].set_ylabel("success rate (%)")
    axs[1].set_title("(b) target found")

    st, length = s["episode_length"]
    _panel(axs[2], st, length, "agent-turns", "(c) episode length")

    for i, ax in enumerate(axs):
        ax.axvline(eps_floor_ep, ls="--", lw=1.3, color=C_EVAL)
        if i == 0:
            ax.annotate(r"$\varepsilon$ floor", xy=(eps_floor_ep, 0.06),
                        xycoords=("data", "axes fraction"),
                        xytext=(7, 0), textcoords="offset points",
                        color=C_EVAL, fontsize=11, va="bottom")
        ax.legend(loc="lower right" if i != 1 else "lower center", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig_training_curves.png")


# ---------------------------------------------------------------------------
# (2) schedules
# ---------------------------------------------------------------------------

def _schedules(cfg, n_episodes):
    """Recompute the two per-episode schedules exactly as training applies them.

    Mirrors `DQNAgent.decay_epsilon` and `agents/scheduler_utils.py`
    (linear warmup over `warmup_pct` of the run, then cosine annealing).
    """
    a  = cfg["agent"]
    ls = a["lr_scheduler"]

    ep = np.arange(n_episodes + 1, dtype=float)
    eps = np.maximum(a["epsilon_start"] * a["epsilon_decay"] ** ep, a["epsilon_end"])
    floor_ep = int(np.argmax(eps <= a["epsilon_end"]))

    warm = int(round(ls["warmup_pct"] * n_episodes))
    lr = np.empty_like(ep)
    lr[:warm] = a["lr"] + (ls["max_lr"] - a["lr"]) * (ep[:warm] / max(warm, 1))
    prog = (ep[warm:] - warm) / max(n_episodes - warm, 1)
    lr[warm:] = ls["min_lr"] + 0.5 * (ls["max_lr"] - ls["min_lr"]) * (
        1 + np.cos(np.pi * np.clip(prog, 0, 1))
    )
    return ep, eps, lr, floor_ep, warm


def make_schedules(cfg, n_episodes):
    ep, eps, lr, floor_ep, warm = _schedules(cfg, n_episodes)
    a = cfg["agent"]

    fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.6))

    axs[0].plot(ep, eps, color=C_MAIN, lw=2.4)
    axs[0].axvline(floor_ep, ls="--", lw=1.3, color=C_EVAL)
    axs[0].axhline(a["epsilon_end"], ls=":", lw=1.0, color="0.4")
    axs[0].annotate(rf"floor $\varepsilon = {a['epsilon_end']}$ at ep. {floor_ep:,}",
                    xy=(floor_ep, a["epsilon_end"]), xytext=(0.22, 0.30),
                    textcoords="axes fraction", color=C_EVAL, fontsize=11,
                    arrowprops=dict(arrowstyle="-", color=C_EVAL, lw=0.9))
    axs[0].set_xlabel("episode")
    axs[0].set_ylabel(r"exploration rate $\varepsilon$")
    axs[0].set_title(r"(a) $\varepsilon$-greedy schedule")

    axs[1].plot(ep, lr * 1e4, color=C_ALT, lw=2.4)
    axs[1].axvline(warm, ls="--", lw=1.3, color="0.45")
    axs[1].annotate(f"warmup ends\n(ep. {warm:,})", xy=(warm, lr[warm] * 1e4),
                    xytext=(0.20, 0.72), textcoords="axes fraction",
                    color="0.35", fontsize=11,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    axs[1].set_xlabel("episode")
    axs[1].set_ylabel(r"learning rate ($\times 10^{-4}$)")
    axs[1].set_title("(b) warmup + cosine LR schedule")

    fig.tight_layout()
    _save(fig, "fig_schedules.png")
    return floor_ep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["curves", "schedules"],
                   help="Regenerate a single figure (default: both).")
    p.add_argument("--config", default=CONFIG)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n_episodes = sum(ph.get("n_episodes", 0) for ph in cfg["curriculum"])

    floor_ep = None
    if args.only != "curves":
        floor_ep = make_schedules(cfg, n_episodes)
    if args.only != "schedules":
        if floor_ep is None:
            floor_ep = _schedules(cfg, n_episodes)[3]
        make_training_curves(floor_ep)


if __name__ == "__main__":
    main()
