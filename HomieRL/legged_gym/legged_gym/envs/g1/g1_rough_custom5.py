"""G1 rough custom5 environment: custom2 terrain-curriculum gating + pillar-aware
pooled tracking criteria; rough_reward_gating disabled."""

from __future__ import annotations

import torch

from legged_gym.envs.g1.g1_rough_custom2 import G1RoughCustom2


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
