"""
networks.py — Neural-network definitions for the Multi-Drone Search DQN.

Contains every ``nn.Module`` used by the agent (ConvNeXt encoders, the global
self-attention bottleneck, the FiLM conditioning block, and the dueling
Q-network) plus the channel/patch/context constants they depend on. The agent
logic (action selection, replay, training step) lives in ``agents/dqn_agent.py``;
the replay buffer in ``agents/replay_buffer.py``; the LR schedule in
``agents/scheduler_utils.py``.

Architecture (branch ``convnext_attn_net``):
  - ``GlobalMapCNN`` — a ConvNeXt backbone over the 32x32 global map that keeps
    spatial resolution down to 8x8, then a multi-head self-attention bottleneck
    over the 64 spatial tokens (+ a CLS and a context token). This restores the
    global spatial-relational reasoning that a flatten/pool would destroy
    (BoTNet/CoAtNet-style conv+attention hybrid).
  - ``LocalCNN`` — a stride-1 ConvNeXt encoder over the 13x13 patch; the 7x7
    depthwise kernels cover almost the whole field, replacing the old ASPP.
  - ``CnnQNetwork`` — Dueling Double DQN head fusing both encoders + raw context.

All normalisation is batch-size-independent (LayerNorm / GRN), identical in
train and eval — consistent with the project's earlier GroupNorm rationale.

Channel-layout constants are duplicated (intentionally, with no shared import) in
``env/grid_env.py`` — see docs/ARCHITECTURE.md "Invariant 3". Keep the two copies in sync.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
# Global map  (B, GLOBAL_CHANNELS, H, W): coarse spatial memory
#   0  Visited      – cells explored by any drone
#   1  Obstacle     – known obstacles
#   2  Trajectory   – normalised visit counts (cap at _TRAJ_CAP)
#   3  Target       – last known target position (memory after leaving FOV)
#   4  Own_Position – this drone's own cell (single 1.0). The network reads it
#                     back (argmax) to crop the local patch, so position is a
#                     spatial signal here rather than a context scalar.
GLOBAL_CHANNELS = 5

# Index of the Own_Position channel within the global map (kept last).
OWN_POSITION_CHANNEL = 4

# Local patch  (B, LOCAL_CHANNELS, P, P): fine-grained detail around drone
#   0  Visited
#   1  Obstacle
#   2  Trajectory
#   3  Target
#   4  Other_Position – positions of other drones within vision
LOCAL_CHANNELS = 5

# Local patch side length.  Must be odd; half-radius = PATCH // 2.
# Sized to cover the maximum vision_radius (6) -> field = 2*6+1 = 13.
LOCAL_PATCH_SIZE = 13

# Context vector: [vision_radius, comm_range, n_agents]. Agent position is NOT
# here — it travels as the Own_Position global channel (see above).
CTX_DIM = 3


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis of a channels-first ``(B, C, H, W)`` tensor.

    ConvNeXt normalises over channels (per spatial location) rather than over a
    batch. Like the project's earlier GroupNorm choice it carries no running
    statistics, so train and eval behave identically and batch-size-1 inference
    is safe.

    Args:
        channels: Number of channels to normalise.
        eps: Numerical-stability epsilon.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias   = nn.Parameter(torch.zeros(channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2, Woo et al. 2023).

    Channel-wise feature recalibration on a channels-last ``(B, H, W, C)`` tensor:
    normalises each channel's global L2 response across all others, then applies a
    learnable scale/shift. Replaces the Squeeze-and-Excitation gate; deterministic
    and batch-size-independent.

    Args:
        channels: Number of feature channels.
        eps: Numerical-stability epsilon.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)        # (B,1,1,C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)      # divisive norm
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    """Stochastic depth (Huang et al. 2016): drop the residual branch per sample.

    Active only in training (a regulariser, like dropout); identity in eval.

    Args:
        drop_prob: Probability of zeroing the residual branch for a sample.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # one mask value per sample, broadcast over the remaining dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt(-V2) block, stride-1 (spatial resolution preserved).

    depthwise 7x7 → LayerNorm(channels) → pointwise 1x1 expand x4 → GELU → GRN →
    pointwise 1x1 contract → LayerScale → residual (+ stochastic depth). An
    optional dilation widens the depthwise receptive field without striding.

    Args:
        dim: Number of channels (kept constant across the block).
        dilation: Atrous rate for the depthwise 7x7 conv (padding scales with it).
        drop_path: Stochastic-depth probability for the residual branch.
        layer_scale_init: Initial value of the per-channel LayerScale gamma.
    """

    def __init__(
        self,
        dim: int,
        dilation: int = 1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim,
            kernel_size=7, stride=1,
            padding=3 * dilation, dilation=dilation,
            groups=dim,
        )
        self.norm    = nn.LayerNorm(dim, eps=1e-6)   # channels-last
        self.pw1     = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.grn     = GRN(4 * dim)
        self.pw2     = nn.Linear(4 * dim, dim)
        self.gamma   = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)             # (B, H, W, C) channels-last
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)             # back to (B, C, H, W)
        return shortcut + self.drop_path(x)


class Downsample(nn.Module):
    """ConvNeXt downsampling layer: LayerNorm2d → 2x2 conv stride 2 (H,W halved).

    Args:
        cin: Input channels.
        cout: Output channels.
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm = LayerNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (Perez et al. 2018).

    Applies a context-conditioned affine transform channel-wise:
    ``out = gamma(ctx) * x + beta(ctx)``. Initialised as identity
    (``gamma = 1``, ``beta = 0``).

    Args:
        ctx_dim: Dimensionality of the conditioning vector.
        channels: Number of feature channels to modulate.
    """

    def __init__(self, ctx_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(ctx_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias[:channels], 1.0)   # gamma -> 1
        nn.init.zeros_(self.proj.bias[channels:])            # beta  -> 0

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Modulate ``x`` ``(B, C, H, W)`` with ``ctx`` ``(B, ctx_dim)``."""
        params = self.proj(ctx)
        C      = x.shape[1]
        gamma  = params[:, :C].unsqueeze(-1).unsqueeze(-1)
        beta   = params[:, C:].unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Global map encoder
# ---------------------------------------------------------------------------

class GlobalMapCNN(nn.Module):
    """ConvNeXt + self-attention encoder for the fixed-size global map (32x32).

    A ConvNeXt backbone keeps spatial resolution down to 8x8 (three stages, each
    FiLM-conditioned on the context vector), after which a multi-head
    self-attention bottleneck mixes the 64 spatial tokens together with a learned
    CLS token and a context token. Unlike a flatten/pool, this preserves global
    spatial-relational reasoning ("where are the unexplored frontiers relative to
    me?") — a BoTNet/CoAtNet-style conv+attention hybrid. The output fuses the
    CLS token with the mean spatial token and projects to ``OUT_DIM``.

    Args:
        ctx_dim: Dimensionality of the context vector used for FiLM / context token.
    """

    STEM_CH      = 96
    STAGE_CH     = (96, 192, 384)   # channels @ 32x32, 16x16, 8x8
    STAGE_DEPTH  = (3, 4, 6)        # ConvNeXt blocks per stage
    ATTN_DIM     = 384              # == STAGE_CH[-1]
    ATTN_HEADS   = 8
    ATTN_LAYERS  = 3
    N_TOKENS     = 8 * 8            # spatial tokens after downsampling to 8x8
    OUT_DIM      = 1024
    DROP_PATH    = 0.1

    def __init__(self, ctx_dim: int = CTX_DIM):
        super().__init__()

        # Stochastic-depth rates, linearly increasing across all blocks.
        total_blocks = sum(self.STAGE_DEPTH)
        dp_rates = [self.DROP_PATH * i / max(total_blocks - 1, 1)
                    for i in range(total_blocks)]

        self.stem = nn.Sequential(
            nn.Conv2d(GLOBAL_CHANNELS, self.STEM_CH,
                      kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.STEM_CH),
        )

        self.stages       = nn.ModuleList()   # ConvNeXt blocks per stage
        self.stage_films  = nn.ModuleList()   # FiLM(ctx) at the end of each stage
        self.downsamples  = nn.ModuleList()   # between stages (Identity after last)

        blk = 0
        ch_in = self.STEM_CH
        for s, (ch, depth) in enumerate(zip(self.STAGE_CH, self.STAGE_DEPTH)):
            # A stage runs at constant channel width; channel changes happen in
            # the preceding downsample (or the stem for the first stage).
            if ch_in != ch:
                raise ValueError(f"Stage {s}: expected ch_in={ch}, got {ch_in}")
            blocks = nn.Sequential(*[
                ConvNeXtBlock(ch, drop_path=dp_rates[blk + i]) for i in range(depth)
            ])
            blk += depth
            self.stages.append(blocks)
            self.stage_films.append(FiLMLayer(ctx_dim, ch))
            if s < len(self.STAGE_CH) - 1:
                self.downsamples.append(Downsample(ch, self.STAGE_CH[s + 1]))
                ch_in = self.STAGE_CH[s + 1]
            else:
                self.downsamples.append(nn.Identity())

        # Self-attention bottleneck over the 8x8 = 64 spatial tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.ATTN_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N_TOKENS, self.ATTN_DIM))
        self.ctx_token = nn.Linear(ctx_dim, self.ATTN_DIM)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.ATTN_DIM,
            nhead=self.ATTN_HEADS,
            dim_feedforward=4 * self.ATTN_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(
            encoder_layer, num_layers=self.ATTN_LAYERS, enable_nested_tensor=False,
        )
        self.attn_norm = nn.LayerNorm(self.ATTN_DIM)

        self.proj = nn.Sequential(
            nn.Linear(2 * self.ATTN_DIM, self.OUT_DIM),
            nn.LayerNorm(self.OUT_DIM),
            nn.GELU(),
        )

        self.out_dim = self.OUT_DIM

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` ``(B, GLOBAL_CHANNELS, 32, 32)`` to ``(B, OUT_DIM)``."""
        B = x.shape[0]
        out = self.stem(x)
        for blocks, film, down in zip(self.stages, self.stage_films, self.downsamples):
            out = film(blocks(out), ctx)     # (B, C, H, W)
            out = down(out)
        # out: (B, ATTN_DIM, 8, 8)

        tokens = out.flatten(2).transpose(1, 2)          # (B, 64, ATTN_DIM)
        tokens = tokens + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, ATTN_DIM)
        ctx_tok = self.ctx_token(ctx).unsqueeze(1)       # (B, 1, ATTN_DIM)
        seq = torch.cat([cls, ctx_tok, tokens], dim=1)   # (B, 66, ATTN_DIM)

        seq = self.attn_norm(self.attn(seq))
        cls_out  = seq[:, 0]                              # (B, ATTN_DIM)
        spatial  = seq[:, 2:].mean(dim=1)                # (B, ATTN_DIM)
        return self.proj(torch.cat([cls_out, spatial], dim=1))   # (B, OUT_DIM)


# ---------------------------------------------------------------------------
# Local patch encoder
# ---------------------------------------------------------------------------

class LocalCNN(nn.Module):
    """Spatial-preserving ConvNeXt encoder for the local patch around the agent.

    Uses ONLY stride-1 operations so that, on a grid where every cell matters, the
    precise spatial signal survives to the embedding: stem → 3 ConvNeXt blocks
    (7x7 depthwise kernels, one dilated, cover almost the whole 13x13 field, so no
    ASPP is needed) → GRN → FiLM (cross-conditioned on the global features) → 1x1
    channel bottleneck → flatten → Linear to 384-dim. Keeping all ``P^2`` positions
    (vs. global pooling) preserves cell-level distinctions.

    Args:
        global_feat_dim: Dimensionality of the :class:`GlobalMapCNN` output, used
            as the conditioning dimension for the cross-branch FiLM layer.

    Note:
        ``P = LOCAL_PATCH_SIZE`` (13); the projection flattens ``48 * 13 * 13 =
        8112`` features to 384.
    """

    STEM_CH  = 96
    BLOCK_CH = 96
    PROJ_CH  = 48    # 1x1 conv output channels before flatten
    OUT_DIM  = 384

    def __init__(self, global_feat_dim: int):
        super().__init__()

        P = LOCAL_PATCH_SIZE   # 13

        self.stem = nn.Sequential(
            nn.Conv2d(LOCAL_CHANNELS, self.STEM_CH,
                      kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.STEM_CH),
        )

        # 7x7 depthwise kernels on a 13x13 patch already see most of the field;
        # the dilated middle block extends the effective range to the edges.
        self.blocks = nn.Sequential(
            ConvNeXtBlock(self.BLOCK_CH, dilation=1),
            ConvNeXtBlock(self.BLOCK_CH, dilation=2),
            ConvNeXtBlock(self.BLOCK_CH, dilation=1),
        )

        self.grn  = GRN(self.BLOCK_CH)                       # channels-last
        self.film = FiLMLayer(global_feat_dim, self.BLOCK_CH)

        flat_dim = self.PROJ_CH * P * P   # 48 * 13 * 13 = 8112
        self.proj_1x1 = nn.Sequential(
            nn.Conv2d(self.BLOCK_CH, self.PROJ_CH, kernel_size=1),
            LayerNorm2d(self.PROJ_CH),
            nn.GELU(),
        )
        self.proj_fc = nn.Sequential(
            nn.Linear(flat_dim, self.OUT_DIM),
            nn.LayerNorm(self.OUT_DIM),
            nn.GELU(),
        )

        self.out_dim = self.OUT_DIM

    def forward(self, x: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """Encode patch ``x`` ``(B, LOCAL_CHANNELS, P, P)`` to ``(B, 384)``."""
        out = self.stem(x)               # (B, 96, P, P)
        out = self.blocks(out)           # (B, 96, P, P) — all stride=1, P unchanged
        out = out.permute(0, 2, 3, 1)    # channels-last for GRN
        out = self.grn(out)
        out = out.permute(0, 3, 1, 2)    # back to channels-first
        out = self.film(out, global_feat)            # FiLM(global)
        out = self.proj_1x1(out)         # (B, 48, P, P)
        out = out.flatten(1)             # (B, 48 * P * P = 8112)
        out = self.proj_fc(out)          # (B, 384)
        return out


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class CnnQNetwork(nn.Module):
    """Dueling Double DQN Q-network.

    Fuses a :class:`GlobalMapCNN` (1024-dim ConvNeXt + self-attention, FiLM on
    context), a :class:`LocalCNN` (384-dim ConvNeXt, FiLM cross-conditioned on the
    global features), and the raw context vector, then a 3-layer FC head splits
    into value/advantage streams: ``Q = V + A - mean(A)``.

    Args:
        grid_size: Side length of the (square) map; used to reshape the flat
            observations back into spatial maps.
        n_actions: Size of the discrete action space.
        fc_hidden: Width of the first FC head layer (then ``//2``, ``//4``).
    """

    def __init__(
        self,
        grid_size: int,
        n_actions: int,
        fc_hidden: int = 1024,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.n_actions = n_actions

        self.global_cnn = GlobalMapCNN(ctx_dim=CTX_DIM)
        self.local_cnn  = LocalCNN(global_feat_dim=self.global_cnn.out_dim)

        combined           = self.global_cnn.out_dim + self.local_cnn.out_dim + CTX_DIM
        self._combined_dim = combined
        self.combined_norm = nn.LayerNorm(combined)

        self.fc1 = nn.Sequential(
            nn.Linear(combined,      fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_hidden,     fc_hidden // 2),
            nn.LayerNorm(fc_hidden // 2),
            nn.GELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(fc_hidden // 2, fc_hidden // 4),
            nn.LayerNorm(fc_hidden // 4),
            nn.GELU(),
        )

        self.value_stream     = nn.Linear(fc_hidden // 4, 1)
        self.advantage_stream = nn.Linear(fc_hidden // 4, n_actions)

    def _extract_local_patch(
        self,
        local_map: torch.Tensor,
        agent_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Crop a ``P x P`` patch centred on each agent's position.

        Args:
            local_map: ``(B, LOCAL_CHANNELS, H, W)`` full local maps.
            agent_pos: ``(B, 2)`` integer ``(row, col)`` in map coordinates.

        Returns:
            ``(B, LOCAL_CHANNELS, P, P)`` with ``P = LOCAL_PATCH_SIZE``.
        """
        B, C   = local_map.shape[0], local_map.shape[1]
        ps     = LOCAL_PATCH_SIZE
        half   = ps // 2
        padded = F.pad(local_map, (half, half, half, half))
        device = local_map.device

        # Vectorized crop (no Python per-sample loop): build, for every sample,
        # the ``ps`` padded-row and padded-col indices of its window, then gather
        # the whole (B, C, ps, ps) block in one advanced-indexing call. Because
        # the map is padded by ``half`` on each side, agent (row, col) maps to the
        # window's top-left corner in padded coordinates, so the window indices
        # are simply ``pos + arange(ps)``. Broadcasting the four index tensors
        # ``(B,1,1,1) x (1,C,1,1) x (B,1,ps,1) x (B,1,1,ps)`` yields (B, C, ps, ps),
        # identical to cropping each sample with ``[:, r:r+ps, c:c+ps]``.
        win      = torch.arange(ps, device=device)
        row_idx  = agent_pos[:, 0].long().unsqueeze(1) + win.unsqueeze(0)   # (B, ps)
        col_idx  = agent_pos[:, 1].long().unsqueeze(1) + win.unsqueeze(0)   # (B, ps)
        b_idx    = torch.arange(B, device=device).view(B, 1, 1, 1)
        c_idx    = torch.arange(C, device=device).view(1, C, 1, 1)
        r_idx    = row_idx.view(B, 1, ps, 1)
        cc_idx   = col_idx.view(B, 1, 1, ps)
        return padded[b_idx, c_idx, r_idx, cc_idx]                          # (B, C, ps, ps)

    def forward(
        self,
        x_global:  torch.Tensor,
        x_local:   torch.Tensor,
        ctx:       torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values.

        Args:
            x_global: ``(B, GLOBAL_CHANNELS * gs * gs)`` flat global maps. The
                Own_Position channel marks the drone's own cell; the crop
                coordinates are recovered from it (argmax), so no separate
                position argument is needed.
            x_local: ``(B, LOCAL_CHANNELS * gs * gs)`` flat local maps.
            ctx: ``(B, CTX_DIM)`` normalised context vector
                ``[vision_radius, comm_range, n_agents]``.

        Returns:
            ``(B, n_actions)`` Q-values.
        """
        B  = x_global.shape[0]
        gs = self.grid_size

        global_map = x_global.view(B, GLOBAL_CHANNELS, gs, gs)
        local_map  = x_local.view(B, LOCAL_CHANNELS,   gs, gs)

        global_feat = self.global_cnn(global_map, ctx)                   # (B, OUT_DIM)

        # Recover the integer (row, col) from the Own_Position channel (a single
        # 1.0 per sample) to crop the drone-centred local patch.
        flat      = global_map[:, OWN_POSITION_CHANNEL].reshape(B, -1).argmax(dim=1)
        agent_pos = torch.stack([flat // gs, flat % gs], dim=1)          # (B, 2)

        local_patch = self._extract_local_patch(local_map, agent_pos)    # (B, 5, P, P)
        local_feat  = self.local_cnn(local_patch, global_feat)           # (B, 384)

        combined = self.combined_norm(
            torch.cat([global_feat, local_feat, ctx], dim=1)             # (B, 1024+384+3)
        )

        h = self.fc3(self.fc2(self.fc1(combined)))   # (B, fc_hidden // 4)
        V = self.value_stream(h)                     # (B, 1)
        A = self.advantage_stream(h)                 # (B, n_actions)
        return V + A - A.mean(dim=1, keepdim=True)
