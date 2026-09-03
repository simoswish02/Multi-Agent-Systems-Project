"""Checks that the network scene tells the truth.

Scene 4 claims to show what the policy actually sees and decides. That claim is
only worth making if it is enforced, so every quantity the scene draws is
checked here against the code that produces it, and the build fails rather than
ship a convincing picture of something false.

Run standalone:  python -m demo.tensorviz.verify
"""

import numpy as np
import torch

from agents.networks import (GLOBAL_CHANNELS, LOCAL_CHANNELS, CTX_DIM,
                             OWN_POSITION_CHANNEL, LOCAL_PATCH_SIZE)
from env.grid_env import N_ACTIONS
from demo.tensorviz import planes as P


class VerificationError(AssertionError):
    pass


def _check(ok, msg):
    if not ok:
        raise VerificationError(msg)


def channel_tables():
    """The scene's channel lists match the real channel counts."""
    _check(len(P.GLOBAL_CHANNELS) == GLOBAL_CHANNELS,
           "scene lists %d global channels, network expects %d"
           % (len(P.GLOBAL_CHANNELS), GLOBAL_CHANNELS))
    _check(len(P.LOCAL_CHANNELS) == LOCAL_CHANNELS,
           "scene lists %d local channels, network expects %d"
           % (len(P.LOCAL_CHANNELS), LOCAL_CHANNELS))
    _check(len(P.CTX_LABELS) == CTX_DIM,
           "scene lists %d context entries, CTX_DIM is %d"
           % (len(P.CTX_LABELS), CTX_DIM))
    _check(P.GLOBAL_CHANNELS[OWN_POSITION_CHANNEL][0] == "Own_Position",
           "channel %d is not the one the scene labels Own_Position"
           % OWN_POSITION_CHANNEL)


def reshape(obs_flat, n_channels, grid):
    """The reshape the network does: flat -> (C, H, W), channel-major."""
    return np.asarray(obs_flat, dtype=np.float32).reshape(n_channels, grid, grid)


def observation(env, obs_i, i, g_planes, l_planes):
    """The planes drawn are the observation, and the observation is the belief."""
    grid = env.grid_size
    _check(g_planes.shape == (GLOBAL_CHANNELS, grid, grid),
           "global planes are %r" % (g_planes.shape,))
    _check(l_planes.shape == (LOCAL_CHANNELS, grid, grid),
           "local planes are %r" % (l_planes.shape,))

    # ...they are literally the flat observation, reshaped
    _check(np.array_equal(g_planes.ravel(), obs_i["global"]),
           "global planes are not obs['global'] reshaped")
    _check(np.array_equal(l_planes.ravel(), obs_i["local"]),
           "local planes are not obs['local'] reshaped")

    # ...and each one is the per-drone belief the environment keeps
    _check(np.array_equal(g_planes[0], env.agent_visited[i]),
           "channel 0 is not this drone's Visited map")
    _check(np.array_equal(g_planes[1], env.agent_obstacle[i]),
           "channel 1 is not this drone's Obstacle map")
    _check(np.array_equal(g_planes[3], env.agent_target[i]),
           "channel 3 is not this drone's Target map")
    _check(np.array_equal(g_planes[5], env.agent_broken[i]),
           "channel 5 is not this drone's Broken map")

    # ...including the single 1.0 the network recovers the position from
    own = g_planes[OWN_POSITION_CHANNEL]
    _check(own.sum() == 1.0, "Own_Position holds %g, not a single 1.0"
           % own.sum())
    rc = np.unravel_index(int(own.argmax()), own.shape)
    _check(tuple(rc) == tuple(env.agent_pos[i]),
           "Own_Position argmax %r is not the drone's cell %r"
           % (tuple(rc), tuple(env.agent_pos[i])))
    return rc


def patch(net, l_planes, pos, drawn):
    """The 13x13 crop drawn is the one the network's own cropper produces."""
    _check(drawn.shape == (LOCAL_CHANNELS, LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE),
           "patch is %r, expected (%d, %d, %d)"
           % (drawn.shape, LOCAL_CHANNELS, LOCAL_PATCH_SIZE, LOCAL_PATCH_SIZE))
    lm = torch.tensor(l_planes, dtype=torch.float32).unsqueeze(0)
    ap = torch.tensor([[pos[0], pos[1]]], dtype=torch.long)
    with torch.no_grad():
        ref = net._extract_local_patch(lm, ap)[0].numpy()
    _check(np.array_equal(ref, drawn),
           "the drawn patch differs from CnnQNetwork._extract_local_patch")


def context(ctx_norm, ctx_params, i, n_alive, drawn):
    """The five chips are ctx_to_numpy's output, not a retelling of it."""
    from training.train import ctx_to_numpy
    ref = ctx_to_numpy(ctx_params, ctx_norm, i, n_alive)
    _check(np.allclose(ref, drawn, atol=1e-6),
           "context chips %r differ from ctx_to_numpy %r" % (drawn, ref))


def decision(q, mask, action):
    """The highlighted bar is the move select_action would actually return."""
    _check(len(q) == N_ACTIONS,
           "%d Q-values drawn, the action space has %d" % (len(q), N_ACTIONS))
    qq = np.array(q, dtype=np.float64).copy()
    if mask is not None:
        qq[~np.asarray(mask, dtype=bool)] = -np.inf
    _check(int(np.argmax(qq)) == int(action),
           "scene highlights action %d, the policy takes %d"
           % (action, int(np.argmax(qq))))


# ---------------------------------------------------------------------------

def main():
    """Run every check against a live episode."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from demo.capture import SimSpec, Sim, make_policy
    from demo import seeds as S

    channel_tables()
    print("channel tables      ok")

    pol = make_policy()
    sim = Sim(S.hero_spec(), pol, render=False).reset()
    for _ in range(25):
        sim.step_round()

    i = 1
    grid = sim.env.grid_size
    g = reshape(sim.obs[i]["global"], GLOBAL_CHANNELS, grid)
    l = reshape(sim.obs[i]["local"], LOCAL_CHANNELS, grid)

    pos = observation(sim.env, sim.obs[i], i, g, l)
    print("observation planes  ok  (drone at %r)" % (pos,))

    net = pol.agent.q_net
    lm = torch.tensor(l, dtype=torch.float32).unsqueeze(0)
    ap = torch.tensor([[pos[0], pos[1]]], dtype=torch.long)
    with torch.no_grad():
        drawn = net._extract_local_patch(lm, ap)[0].numpy()
    patch(net, l, pos, drawn)
    print("13x13 crop          ok")

    q, ctx = pol.q_values(sim.env, sim.obs[i], i, sim.ctx_params)
    context(pol.ctx_norm, sim.ctx_params, i, sim.env.n_alive_belief(i), ctx)
    print("context vector      ok  %s" % np.round(ctx, 3))

    mask = sim.env._get_action_mask(i)
    action = pol.act(sim.env, sim.obs[i], i, sim.ctx_params)
    decision(q, mask, action)
    print("Q-values / argmax   ok  %s -> action %d"
          % (np.round(q, 3), action))
    print("\nall checks passed")


if __name__ == "__main__":
    main()
