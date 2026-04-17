# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import numpy as np
from isaacgym import terrain_utils

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def _rasterize_pillars(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    count_range: tuple = (0, 5),
    height_range: tuple = (0.25, 1.25),
    side_range: tuple = (0.4, 1.5),
    top_noise: float = 0.05,
    platform_size: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    max_rejections: int = 100,
) -> list:
    """Rasterize N random yaw-rotated square pillars into an int16 heightfield.

    Mutates ``height_field_raw`` in place and returns the list of pillars placed
    as ``[(cx_m, cy_m, side_m, yaw_rad), ...]`` in local (subterrain) meter
    coordinates. Pillars never lie inside the ``platform_size`` x ``platform_size``
    spawn region at the center and never lower existing terrain cells.

    ``height_field_raw`` is assumed to be shaped (width_cells, length_cells), with
    coordinate (m) = cell_index * horizontal_scale.
    """
    if rng is None:
        rng = np.random.default_rng()

    W, L = height_field_raw.shape
    env_w_m = W * horizontal_scale
    env_l_m = L * horizontal_scale
    half_plat = platform_size / 2.0
    center_x = env_w_m / 2.0
    center_y = env_l_m / 2.0

    n = int(rng.integers(count_range[0], count_range[1] + 1))
    pillars: list = []
    if n == 0:
        return pillars

    xs = np.arange(W) * horizontal_scale
    ys = np.arange(L) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")  # [W, L] each

    for _ in range(n):
        side = float(rng.uniform(side_range[0], side_range[1]))
        height = float(rng.uniform(height_range[0], height_range[1]))
        yaw = float(rng.uniform(0.0, np.pi / 2.0))  # square has 90 deg symmetry
        half = side / 2.0

        # axis-aligned bounding box of a rotated square: half-diagonal
        aabb_half = half * (abs(np.cos(yaw)) + abs(np.sin(yaw)))

        # rejection-sample a center:
        #  * full square must fit inside the terrain
        #  * square's AABB must be outside the spawn platform + its half-AABB buffer
        placed = False
        for _reject in range(max_rejections):
            cx = float(rng.uniform(aabb_half, env_w_m - aabb_half))
            cy = float(rng.uniform(aabb_half, env_l_m - aabb_half))
            dx_to_center = abs(cx - center_x)
            dy_to_center = abs(cy - center_y)
            # keep the rotated square's AABB fully outside the spawn platform
            if (dx_to_center > half_plat + aabb_half) or (dy_to_center > half_plat + aabb_half):
                placed = True
                break
        if not placed:
            continue  # give up on this pillar; count_range is inclusive of underfill

        # transform cell centers into pillar-local frame (rotate by -yaw around (cx,cy))
        dx = gx - cx
        dy = gy - cy
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        lx = cos_y * dx - sin_y * dy
        ly = sin_y * dx + cos_y * dy
        inside = (np.abs(lx) <= half) & (np.abs(ly) <= half)

        noise_m = rng.uniform(-top_noise, top_noise, size=inside.shape)
        top_m = height + noise_m
        top_units = np.rint(top_m / vertical_scale).astype(np.int32)  # wide to avoid overflow

        # raise only cells inside the pillar; never lower
        current = height_field_raw.astype(np.int32)
        new_total = current + top_units
        chosen = np.where(inside, np.maximum(current, new_total), current)
        height_field_raw[:] = np.clip(chosen, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

        pillars.append((cx, cy, side, yaw))

    return pillars


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type

        if self.type in ["none", "plane"]:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[: i + 1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if cfg.curriculum:
            self.curriculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            row, col = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = self.make_terrain(np.random.uniform(0, 1), np.random.choice([0.5, 0.75, 0.9]))
            self.add_terrain_to_map(terrain, row, col)

    def curriculum(self):
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = row / max(1, self.cfg.num_rows - 1)
                choice = col / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, row, col)

    def selected_terrain(self):
        if self.cfg.terrain_kwargs is None:
            raise ValueError("terrain_kwargs must be set when terrain.selected=True")
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            row, col = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            getattr(terrain_utils, terrain_type)(terrain, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain, row, col)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2

        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-0.05,
                max_height=0.05,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.31,
                step_height=step_height,
                platform_size=3.0,
            )
        else:
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                1.0,
                2.0,
                20,
                platform_size=3.0,
            )
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        start_x = self.border + row * self.length_per_env_pixels
        end_x = self.border + (row + 1) * self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = self.border + (col + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (row + 0.5) * self.env_length
        env_origin_y = (col + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
