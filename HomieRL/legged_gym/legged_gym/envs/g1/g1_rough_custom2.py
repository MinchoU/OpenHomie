"""G1 rough custom2 environment: terrain curriculum gated on action curriculum completion."""

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class G1RoughCustom2(LeggedRobot):
    """LeggedRobot variant that delays terrain curriculum until action curriculum reaches 1.0.

    All agents spawn at terrain level 0. Terrain curriculum only begins after:
    1. action_curriculum_ratio >= 1.0 (action curriculum fully expanded)
    2. The action curriculum "level up" condition is still met
       (tracking_x_vel > 80% of reward scale)
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._terrain_curriculum_unlocked = False

    def reset_idx(self, env_ids):
        """Override to gate terrain curriculum on action curriculum completion."""
        if len(env_ids) == 0:
            return

        # Update command curriculum
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # Update action curriculum
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_action_curriculum(env_ids)

        # Gate terrain curriculum on action curriculum completion
        if self.cfg.terrain.curriculum and self.custom_origins:
            if self._terrain_curriculum_unlocked:
                self._update_terrain_curriculum(env_ids)
            elif self.cfg.env.action_curriculum and self.action_curriculum_ratio >= 1.0:
                # Action curriculum has reached max. Check if the "level up" condition is still met.
                tracking_x_mean = torch.mean(
                    self.episode_sums["tracking_x_vel"][env_ids]
                ) / self.max_episode_length
                if tracking_x_mean > 0.8 * self.reward_scales["tracking_x_vel"]:
                    self._terrain_curriculum_unlocked = True
                    print("[G1RoughCustom2] Terrain curriculum UNLOCKED "
                          f"(action_curriculum_ratio={self.action_curriculum_ratio:.2f}, "
                          f"tracking_x_vel={tracking_x_mean:.4f})")
                    self._update_terrain_curriculum(env_ids)

        self.refresh_actor_rigid_shape_props(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.random_upper_actions[env_ids] = 0.
        self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        from legged_gym.utils.math import euler_from_quaternion
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll.unsqueeze(-1)
        self.pitch[env_ids] = reset_pitch.unsqueeze(-1)
        self.yaw[env_ids] = reset_yaw.unsqueeze(-1)
        self.reset_buf[env_ids] = 1

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            from isaacgym.torch_utils import torch_rand_float
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_kd:
            from isaacgym.torch_utils import torch_rand_float
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_actuation_offset:
            from isaacgym.torch_utils import torch_rand_float
            self.actuation_offset[env_ids] = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (len(env_ids), self.num_dof), device=self.device
            ) * self.torque_limits.unsqueeze(0)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt
            )
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        self.extras["episode"]["terrain_curriculum_unlocked"] = float(self._terrain_curriculum_unlocked)
        if self.custom_origins and hasattr(self, "terrain_levels"):
            terrain_levels_float = self.terrain_levels.float()
            self.extras["episode"]["terrain_level_mean"] = torch.mean(terrain_levels_float)
            self.extras["episode"]["terrain_level_min"] = torch.min(terrain_levels_float)
            self.extras["episode"]["terrain_level_max"] = torch.max(terrain_levels_float)
            for level in range(self.max_terrain_level):
                self.extras["episode"][f"terrain_level_{level}_ratio"] = torch.mean(
                    (self.terrain_levels == level).float()
                )
        if getattr(self.cfg.rewards, "rough_reward_gating_enabled", False) and hasattr(self, "measured_heights"):
            rough_gate = self._get_rough_reward_gate()
            self.extras["episode"]["rough_reward_gate_ratio"] = torch.mean(rough_gate.float())
        if self.actor_use_height and hasattr(self, "measured_heights") and self.scandot_raw_count.item() > 0.0:
            self.extras["episode"]["scandot_raw_min"] = self.scandot_raw_current_min
            self.extras["episode"]["scandot_raw_max"] = self.scandot_raw_current_max
            self.extras["episode"]["scandot_raw_running_min"] = self.scandot_raw_min
            self.extras["episode"]["scandot_raw_running_max"] = self.scandot_raw_max
            self.extras["episode"]["scandot_raw_running_mean"] = self.scandot_raw_mean

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0

        if hasattr(self, "_viser_episode_start_steps"):
            self._viser_episode_start_steps[env_ids] = self.common_step_counter