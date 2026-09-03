"""Casting: pick the episodes the film is shot on.

Scenes 0, 2, 3 and 4 are all the *same* mission seen at different moments, so
the film has one continuous through-line. That mission has to behave: exactly
one drone should fail, somewhere in the middle, and the target should not be
found before the network scene has had its say.

Rather than staging any of that, we search seeds until a real episode does it
on its own, then record the seed. Nothing is faked -- ``reset(seed)``
reproduces both the layout and the fault sequence (``env/grid_env.py:280``).

    python -m demo.seeds --scout          # search and write demo/seeds.json
    python -m demo.seeds --show           # report the cast currently recorded
"""

import json
import os

from demo import config as C
from demo.capture import SimSpec, Sim, make_policy


# The hero mission: the report's baseline configuration, with faults on.
HERO = dict(n_agents=4, vision=3, comm=5, density=0.20,
            fault_prob=0.003, max_steps=200, spawn_corner=0)

# Scouting stops here. The median episode ends around round 50, so a mission
# still running at this point is already in the long tail the film needs --
# and there is no reason to pay for the rest of it just to find that out.
SCOUT_ROUNDS = 115


# What "behaves" means, in rounds. The median episode in this configuration
# lasts about 50 rounds, which is far too brisk to narrate; the film needs one
# from the long tail, and one that ends in success.
# Scenes 2 and 3 take their cue from whenever the first fault *actually*
# happens, so nothing here pins it to a particular round -- it only has to
# land late enough that the team has spread out first, and early enough that
# scene 3 is not the end of the mission.
WANT = dict(
    n_broken=(1, 2),          # at least one wreck, not a massacre
    break_round=(25, 90),
    alive_at=SCOUT_ROUNDS,    # still running here, so scene 4 has a live mission
    min_coverage=0.30,        # the map visibly opens up
)


def evaluate(policy, seed, spec_kw=None, max_rounds=SCOUT_ROUNDS):
    """Run one episode headlessly (no rendering) and describe what happened.

    Stops at `max_rounds`; `running` says whether the episode was still going.
    """
    kw = dict(HERO)
    kw.update(spec_kw or {})
    sim = Sim(SimSpec(seed=seed, **kw), policy, render=False).reset()

    breaks, find_round, rounds = [], None, 0
    while not sim.done and rounds < max_rounds:
        info = sim.step_round()
        rounds += 1
        for i in info.broke:
            breaks.append((i, rounds))
        if info.found and find_round is None:
            find_round = rounds

    env = sim.env
    known = float((env.agent_visited.max(axis=0) > 0).sum())
    traversable = float((env.grid == 0).sum())
    sim.close()
    return {
        "seed": seed,
        "breaks": breaks,
        "n_broken": len(breaks),
        "find_round": find_round,
        "rounds": rounds,
        "running": not sim.done,
        "coverage": known / max(1.0, traversable),
    }


def matches(r):
    """Does this episode satisfy every casting requirement?"""
    lo, hi = WANT["n_broken"]
    if not (lo <= r["n_broken"] <= hi):
        return False
    lo, hi = WANT["break_round"]
    if not (lo <= r["breaks"][0][1] <= hi):
        return False
    if not r["running"]:
        return False
    return r["coverage"] >= WANT["min_coverage"]


def _score(r):
    """Rank candidates: prefer a break near the middle of its window, a
    generous run-time, and a well-explored map."""
    s = -abs(r["breaks"][0][1] - 55) * 0.05 if r["breaks"] else -8.0
    s += r["coverage"] * 2.0
    s -= (r["n_broken"] - 1) * 0.3        # one wreck reads more clearly
    return s


def scout(policy, seeds=range(0, 400), verbose=True):
    """Best seed satisfying WANT across the whole range."""
    found = []
    for s in seeds:
        r = evaluate(policy, s)
        if matches(r):
            found.append(r)
            if verbose:
                print("  seed %3d  break=r%-3d still running at %d  cov=%.2f"
                      "  <== cast" % (s, r["breaks"][0][1], r["rounds"],
                                      r["coverage"]))
        elif verbose and s % 20 == 0:
            print("  seed %3d  broken=%d len=%-3d running=%-5s cov=%.2f"
                  % (s, r["n_broken"], r["rounds"], r["running"],
                     r["coverage"]))
    if not found:
        raise SystemExit(
            "No seed in %r satisfies WANT=%r. Widen the range or relax the "
            "criteria." % (seeds, WANT))
    best = max(found, key=_score)
    if verbose:
        print("  %d candidates; cast seed %d" % (len(found), best["seed"]))
    return best


# ---------------------------------------------------------------------------
# The recorded cast
# ---------------------------------------------------------------------------

def load():
    if os.path.exists(C.SEEDS_JSON):
        with open(C.SEEDS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(data):
    with open(C.SEEDS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return C.SEEDS_JSON


def hero_spec():
    """SimSpec for the hero mission, using the recorded seed."""
    cast = load()
    return SimSpec(seed=int(cast.get("hero", {}).get("seed", 0)), **HERO)


def hero_facts():
    """What the recorded hero mission does -- scenes cue off these."""
    return load().get("hero", {})


def hero_sim(policy, upto_round=0, cell=None):
    """A hero-mission Sim fast-forwarded to `upto_round`.

    One frame is rendered per skipped round: the sidebar's charts and log are
    accumulated by the render path (``_tick_data``), so skipping it entirely
    would leave scene 3 and 4 with an empty, and untruthful, history panel.
    """
    sim = Sim(hero_spec(), policy, cell=cell).reset()
    sim.frame(1.0)                      # log the episode-start banner
    for _ in range(max(0, int(upto_round))):
        if sim.done:
            break
        sim.step_round()
        sim.frame(1.0)
    return sim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", action="store_true", help="search for the hero seed")
    ap.add_argument("--show", action="store_true", help="report the recorded cast")
    ap.add_argument("--range", type=int, nargs=2, default=[0, 400])
    args = ap.parse_args()

    if args.show:
        print(json.dumps(load(), indent=2))
        return

    if args.scout:
        print("loading policy...")
        pol = make_policy()
        print("scouting hero mission seeds %d..%d" % tuple(args.range))
        r = scout(pol, range(args.range[0], args.range[1]))
        cast = load()
        cast["hero"] = r
        print("wrote", save(cast))
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
