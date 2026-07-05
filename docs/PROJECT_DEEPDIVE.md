> **Historical document** — inherited from the RL-course project *before* the
> Multi-Agent-Systems fault extension (random faults, Broken channel, CTX_DIM=5,
> obstacle-density/fault DR, eval-only BFS auto-nav). Numbers such as channel
> counts or context dims may be stale here; the current references are
> [ARCHITECTURE.md](ARCHITECTURE.md), [ENV.md](ENV.md), [TRAINING.md](TRAINING.md).

# Project Deep-Dive — Multi-Drone Cooperative Search

This document is a complete, self-contained technical description of the project.
It is written to be read end to end, like a report: it explains every class, what
its methods do, how the components interact, how data and control flow through the
system during training and testing, and which algorithms are used and why. It is
meant to support the oral discussion of the work, so each design choice is
motivated rather than merely stated. It complements the shorter, paper-style
report in `report/report.tex` and the existing references in
`docs/ARCHITECTURE.md`, `docs/TRAINING.md` and `docs/ENV.md`.

---

## 1. Overview and problem statement

The task is cooperative search. A team of `N` drones must locate a single hidden
target on a fixed `32×32` grid, and it must do so in the shortest possible time.
The target is always placed on a cell that is reachable from the drones' starting
position (the environment guarantees this through a breadth-first reachability
check), so every instance has a solution and failures are due to the policy rather
than to impossible maps.

The problem is both *cooperative* and *partially observable*. Each drone perceives
only a local neighbourhood, whose size is the drone's *vision radius*, and it can
share what it has discovered only with teammates that are currently within a
limited *communication range*. No agent ever sees the full state of the world, and
there is no central controller that acts on behalf of the team. The drones must
therefore coordinate implicitly, through the maps they exchange and through a
shared objective.

The central design decision of the project is to train a **single shared policy**
rather than one controller per drone or one model per configuration. A single
Q-network is used to drive every drone (this is known as *parameter sharing*), and
the same network is expected to **generalise** across different team sizes, vision
radii and communication ranges. Generalisation is obtained through **per-episode
domain randomization**: at the beginning of every episode the environment samples a
new triplet `(vision, comm, n_agents)`, and these values are also handed to the
network as a **context vector**. Because the network is told the regime it is
operating in, the same weights can behave appropriately whether they are
controlling one drone with a narrow field of view or four drones with wide vision
and long-range communication.

The learning algorithm is a **Dueling Double DQN** with `n`-step returns. It is
paired with a context-conditioned convolutional and attention-based policy network,
and with a cooperative terminal reward that credits the whole team when any drone
finds the target.

---

## 2. Project structure

The repository is organised by responsibility, so that the environment, the
learning components and the orchestration code are kept separate and can be read
independently.

```
Reinforcment-Learning-Project-2526/
├── env/
│   ├── grid_env.py        # DroneSearchEnv: the Gymnasium environment
│   └── utils.py           # generate_grid, bfs_reachable
├── agents/
│   ├── networks.py        # CnnQNetwork, GlobalMapCNN, LocalCNN, ConvNeXt/FiLM blocks
│   ├── dqn_agent.py       # DQNAgent: Dueling Double DQN, action selection, update
│   ├── nstep.py           # NStepBuffer: n-step windows and cooperative terminal
│   ├── replay_buffer.py   # ReplayBuffer (uniform sampling)
│   ├── scheduler_utils.py # EpisodeLRScheduler (warmup then cosine annealing)
│   └── base_agent.py      # abstract interface
├── training/
│   └── train.py           # the training loop, domain randomization, curriculum, evaluation
├── gui/renderer.py        # pygame visualisation
├── configs/default.yaml   # the single configuration file
├── testing/
│   ├── analysis.ipynb     # analysis notebook
│   └── evaluate_policy.py # the generalisation harness (500 held-out seeds)
├── checkpoints/           # saved weights (Phase0/, Phase3/, with best.pt and last.pt)
├── eval_results/          # per-checkpoint evaluation logs produced during training
├── testing_results/       # the generalisation study: csv/ and plots/
├── main.py                # entry point: train / eval / play / simulate
├── eval_checkpoints.py    # batch and --watch evaluation of checkpoints
└── docs/                  # ARCHITECTURE, TRAINING, ENV, and this document
```

The remaining sections follow this structure: first the environment, then the
network, then the agent and the learning rule, then the training loop, and finally
the execution flows, the configuration, the results and the design rationale.

---

## 3. The environment — `env/grid_env.py` (`DroneSearchEnv`)

### 3.1 Gymnasium compliance

The environment is implemented as a subclass of `gymnasium.Env` (the file imports
`gymnasium as gym` and declares `class DroneSearchEnv(gym.Env)`). It defines all
the attributes and methods that a standard Gymnasium environment is expected to
expose, so that it can be used exactly like any classic environment.

The `observation_space` is a `spaces.Dict` containing two `Box` spaces, called
`global` and `local`, both holding flattened float32 arrays normalised to the
`[0,1]` range. The global array has `5 · 32 · 32 = 5120` entries and the local
array has `5 · 32 · 32 = 5120` entries. The `action_space` is a `spaces.Discrete(4)`
covering the four cardinal moves (up, down, left and right). The class also
declares `metadata = {"render_modes": ["human", "rgb_array"]}`.

The `reset` method follows the modern Gymnasium signature,
`reset(self, seed=None, options=None)`, and returns a `(observation, info)` tuple.
The `step` method follows the standard signature `step(self, action)` and returns
the five-tuple `(observation, reward, terminated, truncated, info)`. As a result,
the canonical smoke-test required by the assignment runs without any modification:

```python
env = DroneSearchEnv(config)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(reward)
```

There is, however, an important subtlety to understand, because the environment is
genuinely **multi-agent and sequential** rather than single-agent. During training
the drones are not stepped all at once; instead each drone is advanced
individually through a dedicated method, `step_agent(agent_idx, action)`, which
moves one specific drone, computes its reward, updates its map and fuses
information with nearby teammates. The public `step(action)` method is a thin
wrapper around this mechanism: it keeps an internal round-robin pointer
(`self._current_agent`), applies the incoming action to the next drone in turn,
advances the pointer, and automatically resets the episode if it has already
ended. This wrapper exists so that the environment satisfies the canonical
single-agent contract and can be probed with the smoke-test or with quick scripted
experiments, while the training loop continues to use the more explicit
`step_agent` interface that it needs for true multi-agent control. Consistently
with the multi-agent nature of the environment, `reset()` returns a **list** of
per-drone observation dictionaries (one entry per drone), which the training loop
indexes by agent.

### 3.2 The main methods and how they fit together

The constructor, `__init__(config, render_mode=None)`, reads the `env` section of
the configuration. It fixes the grid size to `32`, which is the single immutable
constant of the whole problem, and it stores the reward coefficients, the spawn
behaviour and the observation and action spaces. Vision, communication and team
size are deliberately *not* fixed here, because they are randomised per episode.

The method `set_domain_params(vision_radius, comm_range, n_agents)` updates exactly
those three quantities. It must be called **before every `reset()`**, and this
ordering matters: only if the environment is reconfigured first will the next
episode actually be generated with the intended field of view, communication range
and number of drones, so that what the network observes is coherent with the
context vector it is given. Because `reset()` re-allocates the per-agent map arrays
from the current team size, changing the number of drones through this method is
safe.

The method `reset(seed, options)` builds a fresh episode. It generates a new grid
and target through `generate_grid` in `env/utils.py`, which keeps sampling until
the target is reachable from the spawn location (verified with `bfs_reachable`). It
places all drones at one randomly chosen corner, zeroes the per-agent discovery
maps, performs an initial map update and communication round so that the drones
already "see" their starting surroundings, and finally returns the list of
observations together with an info dictionary.

The method `step_agent(agent_idx, action)` performs a single drone's move and is
where almost all of the environment dynamics live: it validates the move, applies
the reward terms, updates the acting drone's map, runs the communication fusion and
finally returns that drone's Gymnasium five-tuple. The public `step(action)`, as
described above, simply dispatches to `step_agent` in round-robin order. A small
set of internal helpers supports these methods: `_get_obs(i)` assembles a drone's
observation, `_get_action_mask(i)` computes its set of legal moves, `_update_map(i)`
refreshes its discovery maps, `_communicate()` performs the inter-drone fusion, and
`_sync_known_mask()` keeps the global "known cells" bookkeeping consistent.

### 3.3 The observation space in detail

Each drone receives two flattened map stacks, both with values in `[0,1]`. The
*global* stack has five channels, in the order Visited, Obstacle,
Trajectory (a recency or visitation-frequency map), Target and Own_Position. The
*local* stack also has five channels: the first four are the same, while the fifth
is an Other_Position channel that marks the positions of teammates currently seen
within the drone's vision. The Visited, Obstacle and Target channels are binary
maps that reflect what the drone, together with its current communication group,
has discovered so far, while the Trajectory channel is a continuous recency map
normalised by a fixed cap.

A deliberate and important choice is how a drone's **own position** is encoded: it
is the `Own_Position` channel of the *global* stack — a single `1.0` at the drone's
own cell. The network reads that channel back (via argmax) to crop a local patch
centred on the drone, so position is available both as a spatial signal to the
convolution and as the crop coordinate. (Earlier versions instead carried the
position out of band, inside the context vector; it now lives in the observation.)

The context vector itself now has three entries,
`[vision, comm, n_agents]`, and is assembled in
`training/train.py` by the helper `ctx_to_numpy`. Each entry is normalised: vision,
communication and team size are divided by their domain-randomization maxima
(`6`, `12` and `4` respectively). Because none of the three changes during an
episode, the context is constant per episode. Normalisation keeps the quantities on
comparable scales and means that, if the domain-randomization ranges are ever
changed in the configuration, the
context is rescaled everywhere automatically.

### 3.4 The reward function

The reward returned by each call to `step_agent` is the sum of several terms, each
of which encodes a piece of the desired behaviour.

| Term | Default value | When it applies |
|------|---------------|-----------------|
| step penalty | `-0.3` | on every move, to create time pressure |
| wall penalty | `-0.02` | when the chosen move is invalid (into a wall or off the grid) |
| revisit penalty | `-0.05 · k` | when entering a cell already visited `k` times |
| collision penalty | `-0.5 · m` | when `m` other drones occupy the same cell |
| exploration bonus | `+0.02 · Δ` | for the `Δ` newly discovered cells |
| target reward | `+200` | when this drone reaches the target |

The step penalty makes every wasted move costly and therefore rewards short
solutions. The exploration bonus pulls the drones towards unseen territory, which
is what makes search progress at all. The revisit penalty grows with the number of
prior visits to a cell, so that the cumulative cost of repeatedly returning to the
same place grows roughly quadratically; this strongly discourages the oscillation
loops that value-based agents tend to fall into. The collision penalty discourages
drones from piling onto the same cell, and the wall penalty gently discourages
useless moves into obstacles. The large terminal reward dominates everything else
and defines the actual objective.

When the configuration flag `shared_target_reward` is set to `true`, the terminal
`+200` is not given only to the drone that reaches the target but is **shared with
the whole team** through the `n`-step mechanism described later. This turns the
problem into a genuinely cooperative one, because every drone is rewarded for a
discovery made by any teammate. Sharing the terminal reward requires an `n`-step
window of at least two, since the bonus is injected into each drone's pending
transitions.

### 3.5 Communication and map fusion

After every move, the environment runs `_communicate()`, which models range-limited
information sharing. It first groups the drones using a union–find pass: any two
drones whose distance is within the communication range are placed in the same
group, and grouping is *transitive*, so a chain of nearby drones forms a single
connected component even if its endpoints are far apart. For each resulting group,
the members fuse their Visited, Obstacle and Target maps by taking the
element-wise maximum, which means that anything known to one member becomes known
to all of them. In this way the team effectively shares a partial map without any
central authority, and knowledge propagates outward through chains of neighbours.
The distance is Manhattan by default.

### 3.6 Action masking, termination and truncation

The helper `_get_action_mask(i)` returns a boolean array of length four indicating
which of the four moves are legal: a move is legal if it stays inside the grid and
does not enter an obstacle. As a safety fallback, if a drone happens to be boxed in
so that no move is legal, all four actions are re-enabled, so the drone is never
left without a valid choice.

The environment distinguishes carefully between termination and truncation. An
episode is **terminated** when a drone reaches the target; this is a true terminal
event, and the value of the next state must not be bootstrapped. An episode is
**truncated** when the step budget, equal to `max_steps · n_agents`, is exhausted;
this is merely a time-out, and the value of the next state *should* still be
bootstrapped, because the episode did not actually end in a terminal state. The two
are kept distinct throughout the code: transitions store only `float(terminated)`,
never `terminated or truncated`. The episode loop still stops on either condition,
but the learning signal treats them differently. This is one of the subtle
RL-correctness points of the project.

---

## 4. The network — `agents/networks.py`

The network consumes the global map, the agent-centred local patch and the context
vector, and produces one Q-value per action. A few layout constants are shared in
spirit with the environment (and deliberately duplicated rather than imported):
the global stack has four channels, the local stack has five, the local patch is
`13×13` (which is `2·6+1`, large enough to contain the maximum vision radius of
six), and the context dimension is five. A guiding constraint of the whole
architecture is that every normalisation layer is **batch-size independent** (it
uses Layer Normalisation or Global Response Normalisation rather than Batch
Normalisation), because the network must run on a single sample during action
selection, where batch statistics would be meaningless.

### 4.1 The building blocks

Several reusable modules are defined first. `LayerNorm2d` applies layer
normalisation across the channel axis of a channels-first tensor.
`GRN` (Global Response Normalization) is the ConvNeXt-V2 normalisation that
rescales each channel by a global L2 statistic. `DropPath` implements stochastic
depth, randomly dropping residual branches during training to regularise the
network. The `ConvNeXtBlock` is the core convolutional unit: it applies a depthwise
`7×7` convolution, then layer normalisation, then a pointwise expansion by a factor
of four with a GELU non-linearity and GRN, then a pointwise contraction back to the
original width, scaled by a learnable LayerScale and added to the input through a
residual connection (optionally with a dilation to widen the receptive field).
Finally, `FiLMLayer` implements Feature-wise Linear Modulation: given a conditioning
vector it produces a per-channel scale `γ` and shift `β` and returns `γ · x + β`.
It is initialised to the identity transform, and it is the mechanism through which
the context and the global features are allowed to *modulate* other feature maps
rather than being merely concatenated.

### 4.2 The global encoder, `GlobalMapCNN`

The global encoder maps the `(B, 5, 32, 32)` global stack to a `1024`-dimensional
feature vector. It begins with a `3×3` convolutional stem that lifts the five input
channels to ninety-six, followed by layer normalisation. It then applies three
ConvNeXt stages whose widths grow as `96 → 192 → 384` and whose depths are three,
four and six blocks; between stages the spatial resolution is halved, so the map
shrinks from `32×32` to `16×16` to `8×8`. After each stage the feature map is
modulated by the context vector through a FiLM layer, so the operating regime
influences the representation at several depths, and the stochastic-depth rate
increases linearly across the blocks up to `0.1`.

At the `8×8` resolution the encoder switches from convolution to attention. The
sixty-four spatial cells are treated as a sequence of sixty-four tokens, to which a
learned positional embedding is added; a learned classification (CLS) token and a
dedicated context token (produced by a small linear projection of the three-element
context vector) are prepended. This sequence is processed by a three-layer,
eight-head Transformer encoder with a feed-forward width of `1536`, GELU
activations and pre-normalisation. From the output, the CLS embedding and the mean
of the spatial tokens are concatenated and projected to the final `1024`-dimensional
global feature. The reason for adding self-attention on top of the convolutional
trunk is that searching is fundamentally about *relations between regions* — where
the explored area is, where the frontier between explored and unexplored cells
lies — and attention can model those long-range relations directly, whereas global
pooling would average them away.

### 4.3 The local encoder, `LocalCNN`

The local encoder refines fine-grained information around the drone. It receives a
`13×13` patch with five channels, cropped and centred on the drone, together with
the global feature. After a stem that lifts five channels to ninety-six, it applies
three stride-one ConvNeXt blocks (one of them dilated, so that its receptive field
reaches the edges of the patch) that preserve the `13×13` resolution, followed by a
GRN normalisation. The result is then modulated by a FiLM layer **conditioned on the
global feature**, which lets the broad, map-level understanding produced by the
global encoder shape the interpretation of the local neighbourhood. A `1×1`
convolution reduces the width to forty-eight channels, the result is flattened and
projected to a `384`-dimensional local feature with layer normalisation and GELU.

### 4.4 The dueling head, `CnnQNetwork`

The top-level network ties everything together. Its `forward` method receives the
flattened global and local observations and the context vector; the agent position
is no longer a separate argument but is recovered internally from the `Own_Position`
global channel via argmax. It first reshapes the flat inputs back into maps and runs
the global encoder to obtain the `1024`-dimensional global feature. It then extracts
the `13×13` local patch centred on the agent — this is done with a vectorised
"pad-then-gather" operation rather than a Python loop, so it is efficient even for a
whole team at once — and runs the local encoder to obtain the `384`-dimensional
local feature. The global feature, the local feature and the three context values
are concatenated into a `1411`-dimensional vector, which is normalised and passed
through a three-layer fully connected trunk of widths `1536 → 768 → 384`, each layer
followed by layer normalisation and GELU.

Finally, the network applies a **dueling** decomposition. From the shared trunk it
computes a scalar state value `V` and a vector of advantages `A`, one per action,
and combines them as `Q = V + A − mean(A)`. Subtracting the mean advantage makes the
decomposition identifiable and stabilises learning when several actions have similar
value, which is exactly the situation in a search task where many moves are roughly
equally good.

---

## 5. The agent and the learning rule — `agents/dqn_agent.py`

The agent is implemented in `DQNAgent`, supported by `NStepBuffer` (`nstep.py`),
`ReplayBuffer` (`replay_buffer.py`) and `EpisodeLRScheduler` (`scheduler_utils.py`).

### 5.1 Dueling Double DQN with parameter sharing

The agent holds two copies of the network: an online network, `self.q_net`, which is
trained and used to select actions, and a target network, `self.target_net`, which
is held fixed and synchronised only periodically. Both are instances of
`CnnQNetwork`. Crucially, the *same* online network is used to drive every drone:
each drone is queried with its own observation, context and position, but the
weights are shared. This parameter sharing is what makes the policy independent of
the team size and what allows transitions collected from all drones, under all team
sizes, to train a single set of weights.

### 5.2 Action selection

Two selection paths exist. The single-agent path, `select_action`, implements
`ε`-greedy behaviour: with probability `ε` it samples a random *legal* action, and
otherwise it runs the network in evaluation mode, masks the illegal actions by
setting their Q-values to negative infinity, and takes the argmax. The batched path,
`select_actions_batch`, is an optimisation used during training: instead of querying
the network once per drone, it identifies the subset of drones that will act
greedily, runs them through the network in a single forward pass, assigns random
legal actions to the exploring drones, and interleaves the two. This amortises the
relatively expensive convolutional inference over the whole team in one call.

### 5.3 `n`-step returns and cooperative credit

The `NStepBuffer` accumulates single-step experience for each drone in a sliding
window. When a drone's window reaches the configured length `n`, the buffer emits a
single `n`-step transition whose reward is the discounted sum
`R = Σ_{k=0}^{n-1} γ^k r_{t+k}` and whose bootstrap discount is `γ^n` — unless the
window ends in a terminal state, in which case the bootstrap discount is zero and no
future value is added. At the end of an episode, `flush()` emits the remaining
partial windows. The project uses `n = 3` and a discount factor `γ = 0.97`. Using
`n`-step returns extends the effective planning horizon and propagates the sparse
terminal reward backwards more quickly than single-step updates would.

The cooperative terminal reward is implemented here as well, in
`apply_team_terminal(finder, target_reward)`. When a drone finds the target, this
method injects the shared `+200` into every drone's currently pending window, so
that the discovery is credited to the whole team rather than to the finder alone.

### 5.4 The update step

The method `update` performs one gradient step. It first draws a minibatch from the
replay buffer, consisting of the current observations and context, the actions, the
`n`-step returns, the next observations and context, and the per-sample bootstrap
discounts. It then computes the **Double DQN** target. The key idea of Double DQN is
to separate the *selection* of the next action from its *evaluation*: the online
network chooses the greedy next action, `a* = argmax_a q_net(next)`, while the
frozen target network supplies the value of that action,
`Q_target(next)[a*]`. The temporal-difference target is therefore
`target = R + disc · Q_target(next)[a*]`, where `R` is the `n`-step return and
`disc` is `γ^n` for a truncated/continuing transition or zero for a terminal one.
This decoupling is what curbs the systematic over-estimation that ordinary DQN
suffers from, because the same network is no longer used both to pick and to praise
the next action.

The loss is the Huber (smooth-L1) loss between the predicted Q-value of the taken
action and this target; the Huber loss is preferred over the squared error because
it is less sensitive to the occasional large temporal-difference errors. The
gradient is clipped to a maximum norm of one before the optimiser step, which keeps
training stable. The target network is refreshed by a **hard copy** of the online
weights every `target_update_freq = 500` gradient steps, rather than by a slow
soft interpolation. The replay buffer is a uniform buffer backed by a deque with a
capacity of two hundred thousand transitions; it is intentionally *not* saved inside
checkpoints, because it can be regenerated and would otherwise dominate the file
size.

### 5.5 The learning rule in formulas, and why it is off-policy

It is worth restating the algorithm compactly and formally. The method is a
**Dueling Double DQN with `n`-step returns**, trained off-policy from a replay
buffer with a periodically synchronised target network, using a Huber loss and the
Adam optimiser. The parameters are shared across all drones, so a single vector
`θ` (with its frozen copy `θ⁻` for the target network) is learned.

The network produces action values through the **dueling** decomposition, which
splits a state value from per-action advantages and re-centres the advantages so
the decomposition is identifiable:

$$
Q_\theta(s,a) \;=\; V_\theta(s) \;+\; A_\theta(s,a) \;-\; \frac{1}{|\mathcal{A}|}\sum_{a'} A_\theta(s,a').
$$

For a stored `n`-step transition that starts in state `s_t` with action `a_t`, the
target is built in the **Double DQN** way: the *online* network selects the greedy
next action, and the *target* network evaluates it. Writing `d=1` for a terminal
transition (the target was reached) and `d=0` otherwise (including time-out
truncation, which therefore still bootstraps), the target is

$$
a^{\*} \;=\; \arg\max_{a} Q_\theta\!\left(s_{t+n}, a\right),
\qquad
y_t \;=\; \sum_{k=0}^{n-1} \gamma^{k}\, r_{t+k} \;+\; \gamma^{n}\,(1-d)\, Q_{\theta^{-}}\!\left(s_{t+n}, a^{\*}\right).
$$

Learning minimises the **Huber (smooth-L1) loss** of the temporal-difference error
`δ = y_t − Q_θ(s_t,a_t)` over a minibatch `B` sampled uniformly from the replay
buffer `D`:

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{B\sim D}\big[\,\ell_\kappa(\delta)\,\big],
\qquad
\ell_\kappa(\delta) =
\begin{cases}
\tfrac{1}{2}\delta^{2}, & |\delta| \le \kappa,\\[4pt]
\kappa\big(|\delta| - \tfrac{1}{2}\kappa\big), & |\delta| > \kappa,
\end{cases}
$$

with threshold `κ = 1`. The weights are then updated by clipping the gradient to a
maximum norm of one and taking an Adam step with the scheduled learning rate
`η_t`:

$$
g = \nabla_\theta \mathcal{L}(\theta),
\qquad
g \leftarrow g \cdot \min\!\Big(1, \tfrac{1}{\lVert g \rVert_2}\Big),
\qquad
\theta \leftarrow \theta - \eta_t \cdot \widehat{\text{Adam}}(g).
$$

Finally, the target network is refreshed by a hard copy every
`target_update_freq = 500` gradient steps:

$$
\theta^{-} \leftarrow \theta.
$$

The algorithm is **off-policy**, meaning that the policy used to *generate* the
data (the behaviour policy) differs from the policy that is being *evaluated and
improved* (the target policy). Two facts make this concrete. First, the
transitions used in each update are not produced by the current network: they are
sampled from a replay buffer and were collected by *earlier* versions of the
policy, under a larger exploration rate `ε`, so the behaviour distribution is both
exploratory and stale. Second, the bootstrap term evaluates the **greedy** action
`a^{\*} = \arg\max_a Q_\theta(s_{t+n},a)` rather than the action that was actually
taken when the data were collected; in effect we learn the value of the greedy
(near-optimal) policy from data generated by a different, exploratory one. This
decoupling is precisely what allows experience replay to be used at all: an
on-policy method such as SARSA or REINFORCE could not freely reuse old transitions
without importance-sampling corrections, whereas the off-policy Q-learning target
can.

### 5.6 Exploration and learning-rate schedules

Exploration follows a multiplicative `ε`-greedy schedule. The exploration rate
starts at `1.0`, is multiplied by `0.999` once per episode, and is floored at
`0.05`, so the agent explores aggressively at first and then settles into mostly
greedy behaviour. The learning rate follows a separate per-episode schedule
implemented by `EpisodeLRScheduler`: it ramps up linearly from `1e-4` to `5e-4`
over the first ten per cent of the episodes (a warmup phase that avoids unstable
early updates) and then decays with cosine annealing down to `1e-6` over the
remaining episodes.

### 5.7 Checkpoints

The agent can persist and restore its state in two different ways. The full
`save`/`load` path serialises the online network, the optimiser state, the
scheduler state, the current exploration rate and the training step counter, and
optionally a description of the curriculum position; this is what allows an
interrupted run to be resumed from `last.pt` and continue exactly where it left off.
The lighter `load_weights_only` path copies only the online network's weights and is
used to warm-start a brand-new run from a previously trained model. A `best.pt`
checkpoint is written whenever a new best evaluation is achieved, periodic
checkpoints are written as `{phase}_ep{N}.pt`, and `last.pt` always carries the
metadata needed for resumption.

---

## 6. The training loop — `training/train.py`

The training loop ties the environment and the agent together and applies domain
randomization and the curriculum. A few helper functions prepare the context.
`build_ctx_norm(config)` derives the normalisation denominators
(`{vision: 6, comm: 12, n_agents: 4}`) from the domain-randomization maxima, so that
the context is always scaled consistently with the ranges actually used.
`sample_ctx_params(dr_cfg, rng=None)` draws a fresh `(vision, comm, n_agents)`
triplet uniformly from those ranges, optionally from a seeded generator for
reproducible evaluation. `ctx_to_numpy(ctx_params, ctx_norm)` turns a triplet into
the normalised three-element context vector consumed by the network (position is no
longer part of it — it rides in the `Own_Position` observation channel).

Within a phase, each episode proceeds as follows. First a domain-randomization
triplet is sampled. The environment is then reconfigured with
`set_domain_params(**triplet)` **and only afterwards** reset, which guarantees that
the field of view, communication range and team size that actually shape the
episode match the context the network will receive. The episode then unfolds in
rounds: in each round the loop gathers the observations, contexts and action masks
of all drones, selects their actions in a single batched forward pass, and then
steps the drones **sequentially** with `step_agent`, so that each drone's reward,
map update and communication fusion are applied before the next drone moves. Each
resulting transition is pushed into the `n`-step buffer, and a gradient update is
performed every `update_every = 4` steps. If a drone finds the target and the
cooperative reward is enabled, `apply_team_terminal` shares the bonus across the
team. At the end of the episode the loop flushes the remaining `n`-step windows,
decays the exploration rate and advances the learning-rate scheduler.

Runs are organised as a **curriculum**, which is an ordered list of phases declared
in the configuration. A phase whose episode count is zero is simply skipped, so a
run is staged by editing episode counts in the YAML file rather than by changing
code. Each phase may override the number of drones, the obstacle density, the step
budget, an optional exploration reset and per-phase domain-randomization ranges, and
its name is used to tag the TensorBoard logs and the checkpoint filenames.

Evaluation is interleaved with training. Every hundred episodes the agent is
evaluated for twenty episodes with a nearly greedy policy (`ε = 0.05`) on a **fixed
set of seeds**, so that successive checkpoints are compared on identical maps and the
comparison is meaningful. A checkpoint is declared a new best when it improves the
success rate, with ties broken by reward. In addition, when evaluation-during-training
is enabled, a separate `eval_checkpoints.py --watch` process evaluates each periodic
checkpoint on a larger set of episodes as soon as it is written, and records the
outcome in `eval_results/.../watch_summary.csv`.

---

## 7. Execution flows during training and testing

The training flow can be summarised as follows. The configuration is loaded and the
random seeds are set; optionally a parallel evaluator is spawned. For each phase
with a non-zero episode count, a phase-specific environment is built and the episode
loop above is run, periodically evaluating the agent and saving checkpoints.

```
load config → set seeds → (optional) spawn parallel evaluator
for each phase with n_episodes > 0:
    build the phase environment
    for each episode:
        sample domain randomization → set_domain_params → reset
        repeat until the episode ends:
            batched action selection → step_agent for each drone →
            communication fusion → push to n-step buffer → periodic update
        flush n-step windows → decay ε → step the LR scheduler →
        (periodically) evaluate and checkpoint
```

The testing flow is used to produce the generalisation study in
`testing_results/`, through `testing/evaluate_policy.py`. It loads the final
checkpoint, fixes a set of five hundred held-out seeds (numbered 2000 to 2499), and
sweeps **one factor at a time** around a reference configuration of vision three,
communication five, three drones, obstacle density `0.2` and a step budget of two
hundred. For each setting it runs greedy episodes and logs a rich set of metrics —
success, reward, coverage, time-to-find, success-weighted path length, collisions
and revisits — into the `csv/` directory, from which the figures in `plots/` are
produced. Besides training and this study, `main.py` offers further entry points:
`--mode eval` runs a fixed greedy evaluation of a checkpoint, `--mode play` lets a
human drive the first drone with the arrow keys, and `--mode simulate` opens a
graphical configuration screen before running chosen episodes.

---

## 8. Configuration reference — `configs/default.yaml`

The table below collects the values that matter most for understanding and
reproducing the experiments.

| Group | Parameter | Value |
|-------|-----------|-------|
| environment | grid size | **32 (fixed)** |
| environment | obstacle density | 0.20 |
| environment | communication metric | Manhattan |
| environment | reward terms | step −0.3, wall −0.02, revisit −0.05, collision −0.5, exploration +0.02, target +200 |
| environment | shared target reward | true |
| domain randomization | vision / comm / team size | `[1, 6]` / `[2, 12]` / `[1, 4]` |
| agent | discount `γ` / `n`-step | 0.97 / 3 |
| agent | base / max / min learning rate / warmup | 1e-4 / 5e-4 / 1e-6 / 10% of episodes |
| agent | `ε` start / end / decay | 1.0 / 0.05 / 0.999 |
| agent | batch size / replay capacity | 16 / 200000 |
| agent | target sync / grad clip / weight decay | 500 steps / 1.0 / 1e-6 |
| training | update / evaluate / save interval | every 4 / 100 / 200 |
| training | evaluation episodes / `ε` | 20 / 0.05 |
| curriculum | reported phase | `phase2_ConvNeXT_10k_dr`: 10000 episodes, 4 drones, density 0.2, step budget 200 |

---

## 9. Results

The training dynamics, measured at the reference operating point and shown in
`fig_epoch`, are encouraging: over ten thousand episodes the success rate rises from
roughly sixty per cent to roughly eighty-eight per cent, the mean episode return
improves from about minus one hundred to slightly positive, the map coverage grows
from about forty-six to fifty-three per cent, and the success-weighted path length
improves from `0.32` to `0.54`. Measured under the harder full domain-randomized
evaluation (`eval_results/Phase2/watch_summary.csv`), the success rate over the same
run rises from `29.5%` to `77.5%`, while the mean number of steps to completion falls
from about four hundred to roughly two hundred and fifteen.

The generalisation study, run on five hundred held-out seeds, gives a more detailed
picture. At the reference configuration the policy succeeds about `87.6%` of the
time. Varying the **team size** reveals graceful and even improving transfer:
success climbs from `69.2%` with a single drone to `88.4%` with four and `89.2%`
with six, even though the policy was never trained with more than four drones; the
price of larger teams is a steep growth in collisions (from zero to about three
hundred and twenty per episode). Varying the **vision radius** shows that performance
peaks around `88.8%` at a radius of five and degrades only mildly when extrapolated
(`83.6%` at radius eight), with coverage rising monotonically as the sensor widens.
Varying the **communication range** has only a small effect: removing communication
entirely lowers success to `83.6%`, while additional range saturates around `88%`,
which indicates that the shared maps make the policy robust to communication loss.
The clear exception is **obstacle density**, which is by far the hardest axis:
success falls from `98.4%` on an empty grid to `87.6%` at the training density of
`0.2`, then to `52%` at `0.3` and only `25.2%` at `0.4`, and this is the single axis
on which out-of-distribution performance collapses. Finally, increasing the **step
budget** trades computation for success, rising from `47%` with fifty steps to
`87.6%` with two hundred and `89%` with three hundred.

---

## 10. Invariants and audited correctness points

A number of invariants explain why the code is written the way it is, and violating
them silently breaks training. The grid size of thirty-two is the only fixed
constant; vision, communication and team size are randomised. A drone's position is
encoded as the `Own_Position` channel of the global observation, and the network
recovers the local-patch crop coordinates from it via argmax, rather than carrying
position out of band in the context vector. The channel-layout constants are
intentionally duplicated between the network and the environment rather than shared
through an import, and the two copies must be kept in sync. The context normalisation is derived from the domain-randomization maxima,
so editing the ranges rescales the context everywhere. The call to
`set_domain_params` must always precede `reset`, otherwise the context channels
would describe a regime different from the one actually simulated. Execution is
sequential, one drone at a time through `step_agent`, and the team size for an
episode is read back from `env.n_agents`. Truncation is not termination: a time-out
must still bootstrap the next-state value. Finally, the cooperative terminal reward
requires an `n`-step window of at least two.

---

## 11. Anticipated questions for the discussion

This section collects the questions most likely to come up, with concise answers.

**Why Double DQN rather than plain DQN?** Because using the same network to both
select and evaluate the next action causes a systematic over-estimation of
Q-values; Double DQN lets the online network choose the action and the target
network score it, which removes most of that bias.

**Why a dueling architecture?** Because separating the state value from the
per-action advantages stabilises the value estimate, especially in a search task
where many actions from a given state are roughly equally good.

**Why parameter sharing?** Because it makes the policy independent of the number of
drones, it multiplies the amount of data each gradient step learns from, and it is
the standard and most scalable approach to cooperative multi-agent reinforcement
learning.

**Why is the agent's position not part of the observation?** So that the
convolutional input is position-invariant and a single encoder can be reused for any
location. The position is instead delivered through the context vector and used to
crop the local patch and to condition the network through FiLM and the context token.

**Why ConvNeXt together with self-attention?** ConvNeXt is a strong modern
convolutional backbone whose normalisation layers are batch-size independent, which
is essential for single-sample action selection; the self-attention bottleneck on
top of it captures the long-range relationships between explored and unexplored
regions that are central to efficient search and that pooling would discard.

**What is the role of FiLM and the context vector?** A single network has to behave
differently under different vision, communication and team-size regimes; FiLM lets
the regime *modulate* the features smoothly, instead of forcing the network to learn
disjoint behaviours.

**Why `n`-step returns and a shared terminal reward?** The `n`-step returns extend
the effective horizon and speed up the propagation of the sparse terminal signal,
while sharing the terminal reward turns the task into a genuinely cooperative one by
crediting every drone when any teammate succeeds.

**How does communication actually help?** The union–find map fusion pools the
discoveries of all drones within range, so the team shares a partial map without a
central controller, and information spreads transitively through chains of
neighbours.

**Why does the policy generalise, and even improve, beyond four drones?** Because
parameter sharing and map fusion compose naturally: more drones mean more parallel
coverage and more fused knowledge, and the shared weights have already been exposed
to teams of one to four during training.

**What is the main weakness of the approach?** Obstacle density. The policy is
trained at a density of `0.2` and its performance collapses on the denser, more
maze-like layouts it never saw. The natural remedies are a curriculum that exposes
the agent to higher densities and stronger spatial-memory mechanisms, together with
collision-aware coordination for large teams.

**Is the environment really Gym-compliant?** Yes. It subclasses `gymnasium.Env`,
exposes the standard `reset` and `step` signatures and the `action_space` and
`observation_space` attributes, and runs the canonical smoke-test unchanged.
Internally, training uses the per-drone `step_agent` method to realise true
sequential multi-agent control.
