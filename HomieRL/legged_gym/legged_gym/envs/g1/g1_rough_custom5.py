"""G1 rough custom5 environment: custom2 terrain-curriculum gating + pillar-aware
pooled tracking criteria; rough_reward_gating disabled."""

from __future__ import annotations

import torch

from legged_gym.envs.g1.g1_rough_custom2 import G1RoughCustom2


def _segment_hits_pillars(
    base_xy: torch.Tensor,        # [N, 2] world
    end_xy: torch.Tensor,         # [N, 2] world
    pillar_meta: torch.Tensor,    # [..., P, 4] — last axis = (cx, cy, side, yaw)
    pillar_valid: torch.Tensor,   # [..., P] — bool
    sample_steps: int = 4,
) -> torch.Tensor:
    """Check (per row) whether the segment base→end intersects any valid pillar.

    ``pillar_meta`` and ``pillar_valid`` may carry any leading batch shape, but
    their shape along the second-to-last axis must be the max pillar count P.
    Returns a bool tensor of shape ``base_xy.shape[:-1]``.
    """
    if base_xy.shape[0] == 0:
        return torch.zeros(base_xy.shape[:-1], dtype=torch.bool, device=base_xy.device)

    # broadcast check: handles both the unit-test shape [1, 1, 4] and the
    # live shape [N, MAX_PILLARS, 4] (env class indexes into the padded grid).
    if pillar_meta.dim() == 3 and pillar_meta.shape[0] != base_xy.shape[0]:
        pillar_meta = pillar_meta.expand(base_xy.shape[0], *pillar_meta.shape[-2:])
        pillar_valid = pillar_valid.expand(base_xy.shape[0], pillar_valid.shape[-1])

    N = base_xy.shape[0]
    P = pillar_meta.shape[-2]

    # sample points along the segment (t = 0..1)
    ts = torch.linspace(0.0, 1.0, sample_steps, device=base_xy.device)  # [S]
    delta = end_xy - base_xy                                             # [N, 2]
    sample = base_xy[:, None, :] + ts[None, :, None] * delta[:, None, :] # [N, S, 2]

    cx = pillar_meta[..., 0]          # [N, P]
    cy = pillar_meta[..., 1]
    side = pillar_meta[..., 2]
    yaw = pillar_meta[..., 3]

    # pillar-local coords: rotate samples into each pillar's frame
    dx = sample[:, :, None, 0] - cx[:, None, :]   # [N, S, P]
    dy = sample[:, :, None, 1] - cy[:, None, :]
    cos_y = torch.cos(-yaw)[:, None, :]            # [N, 1, P]
    sin_y = torch.sin(-yaw)[:, None, :]
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    half = (side / 2.0)[:, None, :]
    inside = (lx.abs() <= half) & (ly.abs() <= half) & pillar_valid[:, None, :]  # [N, S, P]
    return inside.any(dim=(1, 2))


class G1RoughCustom5(G1RoughCustom2):
    """Custom5: custom4-style config + random pillars in the heightfield.

    Adds per-step "path-clear" tracking that gates the curriculum criteria so that
    envs whose commanded velocity points into a pillar don't drag the progression
    metrics down. Also disables rough_reward_gating (handled in the cfg).
    """

    # ---- tuning knobs (class-level; override in a subclass or cfg if needed) ----
    AHEAD_TIME: float = 0.5         # seconds of commanded velocity projected forward
    SAMPLE_STEPS: int = 4            # points sampled along the forward segment
    MAX_PILLARS: int = 5             # matches pillars.count_range[1] in the cfg

    # -----------------------------------------------------------------------------

    def _init_buffers(self) -> None:
        super()._init_buffers()

        # per-env accumulators
        self._clear_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._clear_tracking_x = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._clear_tracking_y = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        # pillar metadata tensors: [num_rows, num_cols, MAX_PILLARS, 4]
        num_rows = self.cfg.terrain.num_rows
        num_cols = self.cfg.terrain.num_cols

        # Guard against silent truncation: MAX_PILLARS must cover cfg.pillars.count_range[1].
        pillar_cfg = getattr(self.cfg.terrain, "pillars", None)
        if pillar_cfg is not None:
            cfg_max = int(getattr(pillar_cfg, "count_range", (0, 0))[1])
            assert cfg_max <= self.MAX_PILLARS, (
                f"cfg.terrain.pillars.count_range[1]={cfg_max} exceeds "
                f"G1RoughCustom5.MAX_PILLARS={self.MAX_PILLARS}; bump MAX_PILLARS or lower count_range."
            )

        self.pillar_meta = torch.zeros(
            (num_rows, num_cols, self.MAX_PILLARS, 4),
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.pillar_valid = torch.zeros(
            (num_rows, num_cols, self.MAX_PILLARS),
            dtype=torch.bool, device=self.device, requires_grad=False,
        )

        # populate from self.terrain.pillars (only when using trimesh/heightfield)
        if not hasattr(self, "terrain") or not hasattr(self.terrain, "pillars"):
            return
        for r in range(num_rows):
            for c in range(num_cols):
                for k, (cx, cy, side, yaw) in enumerate(self.terrain.pillars[r][c]):
                    if k >= self.MAX_PILLARS:
                        break
                    self.pillar_meta[r, c, k, 0] = cx
                    self.pillar_meta[r, c, k, 1] = cy
                    self.pillar_meta[r, c, k, 2] = side
                    self.pillar_meta[r, c, k, 3] = yaw
                    self.pillar_valid[r, c, k] = True

    def _path_clear_per_step(self) -> torch.Tensor:
        """True per-env when the 0.5 s forward-projection of the commanded
        velocity does not enter any pillar on the env's current subterrain."""
        # per-env padded pillar slice: [N, MAX_PILLARS, 4]
        p_meta = self.pillar_meta[self.terrain_levels, self.terrain_types]
        p_valid = self.pillar_valid[self.terrain_levels, self.terrain_types]

        # transform commanded velocity (base frame, xy) into world frame using yaw
        yaw = self.yaw[:, 0]
        c, s = torch.cos(yaw), torch.sin(yaw)
        cmd_b = self.commands[:, :2]
        cmd_w = torch.stack(
            [cmd_b[:, 0] * c - cmd_b[:, 1] * s,
             cmd_b[:, 0] * s + cmd_b[:, 1] * c],
            dim=1,
        )

        base_xy = self.root_states[:, :2]
        end_xy = base_xy + cmd_w * self.AHEAD_TIME

        blocked = _segment_hits_pillars(
            base_xy, end_xy, p_meta, p_valid, sample_steps=self.SAMPLE_STEPS
        )
        return ~blocked
