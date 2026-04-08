# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import onnxruntime as ort

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def _checkpoint_iteration(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    model_prefix = "model_"
    if not checkpoint_name.startswith(model_prefix) or not checkpoint_name.endswith(".pt"):
        raise ValueError(f"Cannot infer checkpoint iteration from: {checkpoint_path}")
    return checkpoint_name[len(model_prefix):-len(".pt")]


def export_policy(actor_critic, checkpoint_path, dummy_input):
    iteration = _checkpoint_iteration(checkpoint_path)
    path = os.path.join(os.path.dirname(checkpoint_path), 'exported')
    jit_name = f'policy_{iteration}.pt'
    export_policy_as_jit(actor_critic, path, jit_name)
    jit_path = os.path.join(path, jit_name)
    onnx_path = os.path.join(path, os.path.splitext(jit_name)[0] + '.onnx')
    jit_model = torch.jit.load(jit_path, map_location='cpu')
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("ONNX export requires the `onnx` package. Install it with `pip install onnx`.") from exc
    torch.onnx.export(
        jit_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print('Exported policy as jit script to: ', jit_path)
    print('Exported policy as onnx to: ', onnx_path)
    print('Exported policy directory: ', path)


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        return action
    return policy

def load_onnx_policy():
    model = ort.InferenceSession("")
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):
    if getattr(args, "viser", False):
        args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    
    # policy = load_onnx_policy() # Use this to load from exported onnx file
    
    if getattr(args, "export_policy", False):
        checkpoint_path = getattr(ppo_runner, "loaded_model_path", None)
        if checkpoint_path is None:
            raise RuntimeError("Cannot export policy without a loaded checkpoint path.")
        export_policy(ppo_runner.alg.actor_critic, checkpoint_path, obs[:1].detach().cpu())
    print(policy)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    for _ in range(10*int(env.max_episode_length)):
        env.action_curriculum_ratio = 1.0
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        env.commands[:, 4] = height
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.24)
