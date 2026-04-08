import torch
from rsl_rl.algorithms.him_ppo import HIMPPO


class HIMPPO43dof(HIMPPO):
    """HIMPPO with flip functions for G1 43-DOF (27 original + 2 waist + 14 hand).

    DOF ordering:
      0-5:   left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
      6-11:  right leg
      12:    waist_yaw
      13:    waist_roll
      14:    waist_pitch
      15-21: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
      22-28: left hand (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2)
      29-35: right arm
      36-42: right hand (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2)

    One-step obs structure (108):
      [0:4]     commands (x_vel, y_vel, yaw, height)
      [4:7]     imu ang vel (roll, pitch, yaw)
      [7:10]    projected gravity (x, y, z)
      [10:53]   dof_pos (43)
      [53:96]   dof_vel (43)
      [96:108]  actions (12, lower body only)
    """

    NUM_DOFS = 43

    def _flip_dof_block(self, f, p, o):
        """Flip a 43-element DOF block (pos or vel) at offset o.
        f: output tensor, p: input tensor, o: column offset.
        """
        # Left leg (0-5) <-> Right leg (6-11)
        f[:, :, o+0]  =  p[:, :, o+6]
        f[:, :, o+1]  = -p[:, :, o+7]
        f[:, :, o+2]  = -p[:, :, o+8]
        f[:, :, o+3]  =  p[:, :, o+9]
        f[:, :, o+4]  =  p[:, :, o+10]
        f[:, :, o+5]  = -p[:, :, o+11]
        f[:, :, o+6]  =  p[:, :, o+0]
        f[:, :, o+7]  = -p[:, :, o+1]
        f[:, :, o+8]  = -p[:, :, o+2]
        f[:, :, o+9]  =  p[:, :, o+3]
        f[:, :, o+10] =  p[:, :, o+4]
        f[:, :, o+11] = -p[:, :, o+5]

        # Waist
        f[:, :, o+12] = -p[:, :, o+12]  # waist_yaw: negate
        f[:, :, o+13] = -p[:, :, o+13]  # waist_roll: negate
        f[:, :, o+14] =  p[:, :, o+14]  # waist_pitch: keep

        # Left arm (15-21) <-> Right arm (29-35)
        f[:, :, o+15] =  p[:, :, o+29]  # shoulder_pitch
        f[:, :, o+16] = -p[:, :, o+30]  # shoulder_roll
        f[:, :, o+17] = -p[:, :, o+31]  # shoulder_yaw
        f[:, :, o+18] =  p[:, :, o+32]  # elbow
        f[:, :, o+19] = -p[:, :, o+33]  # wrist_roll
        f[:, :, o+20] =  p[:, :, o+34]  # wrist_pitch
        f[:, :, o+21] = -p[:, :, o+35]  # wrist_yaw

        f[:, :, o+29] =  p[:, :, o+15]
        f[:, :, o+30] = -p[:, :, o+16]
        f[:, :, o+31] = -p[:, :, o+17]
        f[:, :, o+32] =  p[:, :, o+18]
        f[:, :, o+33] = -p[:, :, o+19]
        f[:, :, o+34] =  p[:, :, o+20]
        f[:, :, o+35] = -p[:, :, o+21]

        # Left hand (22-28) <-> Right hand (36-42)
        # order: index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2
        # thumb_0 keeps sign; all other finger joints negate under left/right mirroring.
        f[:, :, o+22] = -p[:, :, o+36]  # index_0
        f[:, :, o+23] = -p[:, :, o+37]  # index_1
        f[:, :, o+24] = -p[:, :, o+38]  # middle_0
        f[:, :, o+25] = -p[:, :, o+39]  # middle_1
        f[:, :, o+26] =  p[:, :, o+40]  # thumb_0
        f[:, :, o+27] = -p[:, :, o+41]  # thumb_1
        f[:, :, o+28] = -p[:, :, o+42]  # thumb_2

        f[:, :, o+36] = -p[:, :, o+22]  # index_0
        f[:, :, o+37] = -p[:, :, o+23]  # index_1
        f[:, :, o+38] = -p[:, :, o+24]  # middle_0
        f[:, :, o+39] = -p[:, :, o+25]  # middle_1
        f[:, :, o+40] =  p[:, :, o+26]  # thumb_0
        f[:, :, o+41] = -p[:, :, o+27]  # thumb_1
        f[:, :, o+42] = -p[:, :, o+28]  # thumb_2

    def _flip_commands_imu_gravity(self, f, p):
        """Flip commands, IMU ang vel, and projected gravity (indices 0-9)."""
        f[:, :, 0] =  p[:, :, 0]   # x vel cmd
        f[:, :, 1] = -p[:, :, 1]   # y vel cmd
        f[:, :, 2] = -p[:, :, 2]   # yaw cmd
        f[:, :, 3] =  p[:, :, 3]   # height cmd
        f[:, :, 4] = -p[:, :, 4]   # ang vel roll
        f[:, :, 5] =  p[:, :, 5]   # ang vel pitch
        f[:, :, 6] = -p[:, :, 6]   # ang vel yaw
        f[:, :, 7] =  p[:, :, 7]   # gravity x
        f[:, :, 8] = -p[:, :, 8]   # gravity y
        f[:, :, 9] =  p[:, :, 9]   # gravity z

    def _flip_lower_actions(self, f, p, o):
        """Flip lower-body actions (12) at offset o."""
        f[:, :, o+0]  =  p[:, :, o+6]
        f[:, :, o+1]  = -p[:, :, o+7]
        f[:, :, o+2]  = -p[:, :, o+8]
        f[:, :, o+3]  =  p[:, :, o+9]
        f[:, :, o+4]  =  p[:, :, o+10]
        f[:, :, o+5]  = -p[:, :, o+11]
        f[:, :, o+6]  =  p[:, :, o+0]
        f[:, :, o+7]  = -p[:, :, o+1]
        f[:, :, o+8]  = -p[:, :, o+2]
        f[:, :, o+9]  =  p[:, :, o+3]
        f[:, :, o+10] =  p[:, :, o+4]
        f[:, :, o+11] = -p[:, :, o+5]

    def flip_g1_actor_obs(self, obs):
        num_one_step = self.actor_critic.num_one_step_obs
        hist_len = self.actor_critic.actor_history_length
        prop_len = num_one_step * hist_len

        p = torch.clone(obs[:, :prop_len]).view(-1, hist_len, num_one_step)
        f = torch.zeros_like(p)

        self._flip_commands_imu_gravity(f, p)
        self._flip_dof_block(f, p, 10)          # dof_pos
        self._flip_dof_block(f, p, 10 + self.NUM_DOFS)     # dof_vel
        self._flip_lower_actions(f, p, 10 + 2 * self.NUM_DOFS)  # actions (12)

        flipped_obs = f.view(-1, prop_len)
        height_obs = obs[:, prop_len:]
        if height_obs.shape[1] > 0:
            height_obs = self.flip_g1_height_obs(height_obs)
            flipped_obs = torch.cat((flipped_obs, height_obs), dim=-1)
        return flipped_obs.detach()

    def flip_g1_critic_obs(self, critic_obs):
        num_one_step = self.actor_critic.num_one_step_critic_obs
        hist_len = self.actor_critic.critic_history_length
        prop_len = num_one_step * hist_len

        p = torch.clone(critic_obs[:, :prop_len]).view(-1, hist_len, num_one_step)
        f = torch.zeros_like(p)

        self._flip_commands_imu_gravity(f, p)
        self._flip_dof_block(f, p, 10)          # dof_pos
        self._flip_dof_block(f, p, 10 + self.NUM_DOFS)     # dof_vel
        self._flip_lower_actions(f, p, 10 + 2 * self.NUM_DOFS)  # actions (12)

        # base_lin_vel (3) appended after actions
        o = 10 + 2 * self.NUM_DOFS + 12
        f[:, :, o+0] =  p[:, :, o+0]  # base lin vel x
        f[:, :, o+1] = -p[:, :, o+1]  # base lin vel y
        f[:, :, o+2] =  p[:, :, o+2]  # base lin vel z

        flipped_obs = f.view(-1, prop_len)
        height_obs = critic_obs[:, prop_len:]
        if height_obs.shape[1] > 0:
            height_obs = self.flip_g1_height_obs(height_obs)
            flipped_obs = torch.cat((flipped_obs, height_obs), dim=-1)
        return flipped_obs.detach()

    def flip_g1_actions(self, actions):
        flipped = torch.zeros_like(actions)
        flipped[:,  0] =  actions[:, 6]   # left_hip_pitch  <- right_hip_pitch
        flipped[:,  1] = -actions[:, 7]   # left_hip_roll   <- -right_hip_roll
        flipped[:,  2] = -actions[:, 8]   # left_hip_yaw    <- -right_hip_yaw
        flipped[:,  3] =  actions[:, 9]   # left_knee       <- right_knee
        flipped[:,  4] =  actions[:, 10]  # left_ankle_pitch <- right_ankle_pitch
        flipped[:,  5] = -actions[:, 11]  # left_ankle_roll <- -right_ankle_roll
        flipped[:,  6] =  actions[:, 0]   # right_hip_pitch <- left_hip_pitch
        flipped[:,  7] = -actions[:, 1]   # right_hip_roll  <- -left_hip_roll
        flipped[:,  8] = -actions[:, 2]   # right_hip_yaw   <- -left_hip_yaw
        flipped[:,  9] =  actions[:, 3]   # right_knee      <- left_knee
        flipped[:, 10] =  actions[:, 4]   # right_ankle_pitch <- left_ankle_pitch
        flipped[:, 11] = -actions[:, 5]   # right_ankle_roll <- -left_ankle_roll
        return flipped.detach()


class HIMPPO43dofNoHandObs(HIMPPO43dof):
    """G1 43-DOF policy with 29-DOF proprioception (hands omitted from obs)."""

    NUM_DOFS = 29

    def _flip_dof_block(self, f, p, o):
        # Left leg (0-5) <-> Right leg (6-11)
        f[:, :, o+0]  =  p[:, :, o+6]
        f[:, :, o+1]  = -p[:, :, o+7]
        f[:, :, o+2]  = -p[:, :, o+8]
        f[:, :, o+3]  =  p[:, :, o+9]
        f[:, :, o+4]  =  p[:, :, o+10]
        f[:, :, o+5]  = -p[:, :, o+11]
        f[:, :, o+6]  =  p[:, :, o+0]
        f[:, :, o+7]  = -p[:, :, o+1]
        f[:, :, o+8]  = -p[:, :, o+2]
        f[:, :, o+9]  =  p[:, :, o+3]
        f[:, :, o+10] =  p[:, :, o+4]
        f[:, :, o+11] = -p[:, :, o+5]

        # Waist
        f[:, :, o+12] = -p[:, :, o+12]
        f[:, :, o+13] = -p[:, :, o+13]
        f[:, :, o+14] =  p[:, :, o+14]

        # Left arm (15-21) <-> Right arm (22-28)
        f[:, :, o+15] =  p[:, :, o+22]
        f[:, :, o+16] = -p[:, :, o+23]
        f[:, :, o+17] = -p[:, :, o+24]
        f[:, :, o+18] =  p[:, :, o+25]
        f[:, :, o+19] = -p[:, :, o+26]
        f[:, :, o+20] =  p[:, :, o+27]
        f[:, :, o+21] = -p[:, :, o+28]

        f[:, :, o+22] =  p[:, :, o+15]
        f[:, :, o+23] = -p[:, :, o+16]
        f[:, :, o+24] = -p[:, :, o+17]
        f[:, :, o+25] =  p[:, :, o+18]
        f[:, :, o+26] = -p[:, :, o+19]
        f[:, :, o+27] =  p[:, :, o+20]
        f[:, :, o+28] = -p[:, :, o+21]
