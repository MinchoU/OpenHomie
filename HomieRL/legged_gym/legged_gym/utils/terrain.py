# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from isaacgym import terrain_utils

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


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
