"""G1 rough custom5 environment: custom2 terrain-curriculum gating + pillar-aware
pooled tracking criteria; rough_reward_gating disabled."""

from __future__ import annotations

import numpy as np
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

    ``pillar_meta`` must be 3-D with shape ``[N, P, 4]`` or the broadcast-
    compatible ``[1, P, 4]``; ``pillar_valid`` must be ``[N, P]`` or ``[1, P]``.
    Returns a bool tensor of shape ``base_xy.shape[:-1]``.
    """
    if base_xy.shape[0] == 0:
        return torch.zeros(base_xy.shape[:-1], dtype=torch.bool, device=base_xy.device)

    # broadcast check: handles both the unit-test shape [1, 1, 4] and the
    # live shape [N, MAX_PILLARS, 4] (env class indexes into the padded grid).
    if pillar_meta.dim() == 3 and pillar_meta.shape[0] != base_xy.shape[0]:
        pillar_meta = pillar_meta.expand(base_xy.shape[0], *pillar_meta.shape[-2:])
        pillar_valid = pillar_valid.expand(base_xy.shape[0], pillar_valid.shape[-1])

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

        # transform commanded velocity (base frame, xy) into world frame using yaw.
        # euler_from_quaternion returns flat [N] each post_physics_step, so no index.
        yaw = self.yaw
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

    def compute_reward(self) -> None:
        super().compute_reward()
        # Accumulate per-step "path clear" + tracking reward so curriculum
        # criteria can pool over steps where the env wasn't blocked.
        # _clear_tracking_{x,y} hold the UNSCALED Gaussian (range 0..1 per
        # step); downstream curriculum code must not compare them directly
        # against episode_sums[...] which are multiplied by reward_scales.
        clear = self._path_clear_per_step().float()
        self._clear_time += clear
        sigma = self.cfg.rewards.tracking_sigma
        err_x = (self.commands[:, 0] - self.base_lin_vel[:, 0]).square()
        err_y = (self.commands[:, 1] - self.base_lin_vel[:, 1]).square()
        rew_x = torch.exp(-err_x / sigma)
        rew_y = torch.exp(-err_y / sigma)
        self._clear_tracking_x += rew_x * clear
        self._clear_tracking_y += rew_y * clear

    def reset_idx(self, env_ids) -> None:
        if len(env_ids) == 0:
            return

        # Cache pooled metrics BEFORE any reset so telemetry sees the episode
        # we're actually ending (not the post-zero state).
        total_t, eff_x, eff_y = self._pooled_effective_tracking(env_ids)

        # Inject eff_x into the parent's terrain-unlock check by rewriting the
        # slice the parent averages. Parent (g1_rough_custom2.py:38-48) reads
        #   episode_sums["tracking_x_vel"][env_ids].mean() / max_episode_length
        # and compares to 0.8 * reward_scales["tracking_x_vel"]. reward_scales
        # is dt-scaled at _prepare_reward_function (legged_robot.py:1087), so to
        # make the parent's check fire iff eff_x > 0.8 we inject
        #   sum = eff_x * reward_scales["tracking_x_vel"] * max_episode_length
        # ⇒ parent sees mean/max_ep = eff_x * reward_scales, compared to
        #   0.8 * reward_scales ⇒ unlock iff eff_x > 0.8.
        # Parent (custom2.reset_idx) calls update_action/command_curriculum FIRST
        # (both overridden here → they read _pooled_effective_tracking directly,
        # so the injected value is irrelevant to them), THEN reads
        # episode_sums["tracking_x_vel"][env_ids].mean() for the terrain-unlock
        # check — that's the consumer of the injected value.
        orig_episode_x = self.episode_sums["tracking_x_vel"][env_ids].clone()
        if total_t.item() >= 1.0:
            self.episode_sums["tracking_x_vel"][env_ids] = (
                eff_x * self.reward_scales["tracking_x_vel"] * self.max_episode_length
            )
        else:
            self.episode_sums["tracking_x_vel"][env_ids] = 0.0

        try:
            super().reset_idx(env_ids)
        except Exception:
            # Restore only the tracking_x_vel slice this override mutated. If
            # super().reset_idx crashes mid-way, other env-state (dof_pos,
            # last_actions, commands, episode_sums[...] for other keys, extras)
            # may be in a half-reset state — isaac-gym generally cannot recover
            # from mid-reset failures, but at least we don't leave the injected
            # sentinel value visible to any downstream handler of the exception.
            self.episode_sums["tracking_x_vel"][env_ids] = orig_episode_x
            raise

        # parent already zeroed episode_sums for env_ids as part of its cleanup,
        # so no restoration needed. Now zero our private accumulators.
        self._clear_time[env_ids] = 0.0
        self._clear_tracking_x[env_ids] = 0.0
        self._clear_tracking_y[env_ids] = 0.0

        # pooled telemetry — append to extras["episode"] populated by the parent
        if "episode" in self.extras:
            max_ep = float(self.max_episode_length)
            n = float(len(env_ids))
            self.extras["episode"]["pool_clear_ratio"] = (
                total_t.item() / max(1.0, n * max_ep)
            )
            self.extras["episode"]["pool_tracking_x_effective"] = float(eff_x.item())
            self.extras["episode"]["pool_tracking_y_effective"] = float(eff_y.item())

    # ---------- pooled curriculum criteria ----------

    def _pooled_effective_tracking(self, env_ids) -> tuple:
        """Return (total_clear_time, eff_x, eff_y) pooled across env_ids.

        eff_x / eff_y are set to 0.0 if total_clear_time < 1 (nobody progressed).
        """
        # Callers that already have total_t can read .item() once and avoid re-syncing.
        total_t = self._clear_time[env_ids].sum()
        if total_t.item() < 1.0:
            zero = torch.zeros((), device=self.device)
            return total_t, zero, zero
        eff_x = self._clear_tracking_x[env_ids].sum() / total_t
        eff_y = self._clear_tracking_y[env_ids].sum() / total_t
        return total_t, eff_x, eff_y

    def update_action_curriculum(self, env_ids) -> None:
        total_t, eff_x, _ = self._pooled_effective_tracking(env_ids)
        if total_t.item() < 1.0:
            return
        # eff_x is the pooled UNSCALED Gaussian on clear steps (range 0..1), so
        # the threshold is "80% of peak tracking" — not 0.8 * reward_scales
        # which would be dt-scaled.
        if eff_x > 0.8:
            self.action_curriculum_ratio = min(self.action_curriculum_ratio + 0.05, 1.0)

    def update_command_curriculum(self, env_ids) -> None:
        total_t, eff_x, eff_y = self._pooled_effective_tracking(env_ids)
        if total_t.item() < 1.0:
            return
        if eff_x > 0.8 and eff_y > 0.8:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.2,
                -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.2,
                0., self.cfg.commands.max_curriculum)
