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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    # If load_run is an absolute path, skip listing `root` (which may not exist yet).
    if isinstance(load_run, str) and os.path.isabs(load_run):
        load_run_path = load_run
    else:
        try:
            runs = os.listdir(root)
            #TODO sort by date to handle change of month
            runs.sort()
            if 'exported' in runs: runs.remove('exported')
            last_run = os.path.join(root, runs[-1])
        except:
            raise ValueError("No runs in this directory: " + root)
        if load_run == -1 or load_run == "-1":
            load_run_path = last_run
        else:
            load_run_path = os.path.join(root, load_run)
    load_run = load_run_path

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def apply_reward_scale_overrides(env_cfg, reward_scale_overrides):
    if not reward_scale_overrides:
        return
    if isinstance(reward_scale_overrides, str):
        reward_scale_overrides = [reward_scale_overrides]

    for override in reward_scale_overrides:
        for override in override.split(","):
            override = override.strip()
            if not override:
                continue
            _apply_reward_scale_override(env_cfg, override)

def _apply_reward_scale_override(env_cfg, override):
    if "=" not in override:
        raise ValueError(f"Invalid --reward_scale override '{override}'. Expected key=value.")
    key, value = override.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid --reward_scale override '{override}'. Expected key=value.")
    if not hasattr(env_cfg.rewards.scales, key):
        raise ValueError(f"Unknown reward scale '{key}' in --reward_scale {override}.")
    try:
        value = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for --reward_scale {override}. Expected a float.") from exc

    old_value = getattr(env_cfg.rewards.scales, key)
    setattr(env_cfg.rewards.scales, key, value)
    print(f"Overriding reward scale: {key} {old_value} -> {value}")

def _parse_init_terrain_ratio(raw):
    """Parse --init_terrain_ratio into a list of non-negative floats summing to 1.

    Accepts ``"[0.5,0.3,0.2]"``, ``"0.5,0.3,0.2"``, or whitespace-separated forms.
    """
    s = raw.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    parts = [p.strip() for chunk in s.split(",") for p in chunk.split() if p.strip()]
    if not parts:
        raise ValueError(f"--init_terrain_ratio is empty: {raw!r}")
    try:
        ratio = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--init_terrain_ratio has non-float entry in {raw!r}: {exc}") from exc
    if any(r < 0 for r in ratio):
        raise ValueError(f"--init_terrain_ratio has negative entry: {ratio}")
    total = sum(ratio)
    if total <= 0:
        raise ValueError(f"--init_terrain_ratio sums to <= 0: {ratio}")
    return [r / total for r in ratio]


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if getattr(args, "viser", False):
            env_cfg.viser.enabled = True
        if getattr(args, "viser_port", None) is not None:
            env_cfg.viser.port = args.viser_port
        if getattr(args, "viser_env_idx", None) is not None:
            env_cfg.viser.env_idx = args.viser_env_idx
        if getattr(args, "viser_update_interval", None) is not None:
            env_cfg.viser.update_interval = args.viser_update_interval
        if getattr(args, "viser_no_meshes", False):
            env_cfg.viser.show_meshes = False
        apply_reward_scale_overrides(env_cfg, getattr(args, "reward_scale", None))
        if getattr(args, "debug", False):
            env_cfg.debug = True
        if getattr(args, "init_upper_ratio", None) is not None:
            env_cfg.domain_rand.init_upper_ratio = args.init_upper_ratio
        if getattr(args, "init_terrain_ratio", None) is not None:
            env_cfg.terrain.init_terrain_ratio = _parse_init_terrain_ratio(args.init_terrain_ratio)
            print(f"Using init_terrain_ratio: {[round(r, 4) for r in env_cfg.terrain.init_terrain_ratio]}")
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "aliengo", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--export_policy", "action": "store_true", "default": False, "help": "Export the loaded policy as JIT and ONNX during play."},
        {"name": "--viser", "action": "store_true", "default": False, "help": "Enable viser robot-state visualization."},
        {"name": "--viser_port", "type": int, "default": None, "help": "Port for the viser server."},
        {"name": "--viser_env_idx", "type": int, "default": None, "help": "Initial environment index to visualize in viser."},
        {"name": "--viser_update_interval", "type": int, "default": None, "help": "Number of env steps between viser updates."},
        {"name": "--viser_no_meshes", "action": "store_true", "default": False, "help": "Render the viser robot without visual meshes."},
        {"name": "--reward_scale", "action": "append", "default": None, "help": "Override a reward scale as key=value. Can be passed multiple times."},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Enable training debug checkpoints (breakpoints on NaN/exploding values)."},
        {"name": "--init_upper_ratio", "type": float, "default": None, "help": "Override init_upper_ratio (action curriculum starting point). Use 1.0 to skip curriculum on resume."},
        {"name": "--init_terrain_ratio", "type": str, "default": None, "help": "Initial terrain-level distribution as a comma-separated list of floats (brackets optional), e.g. [0.5,0.3,0.2]. Auto-normalized to sum to 1."},
        {"name": "--log_dir", "type": str, "default": None, "help": "Override the logs base directory (replaces LEGGED_GYM_ROOT_DIR). Logs go to <log_dir>/logs/<experiment_name>/<date_time>_<run_name>."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, filename=None):
    if filename is None:
        filename = 'policy.pt' if hasattr(actor_critic, 'estimator') else 'policy_1.pt'
    if hasattr(actor_critic, 'estimator'):
        # assumes LSTM: TODO add GRU
        if getattr(actor_critic, "actor_use_height", False):
            exporter = PolicyExporterHIMWithTerrain(actor_critic)
        else:
            exporter = PolicyExporterHIM(actor_critic)
        exporter.export(path, filename)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterHIM(torch.nn.Module):
    __constants__ = [
        "num_one_step_obs",
        "actor_proprioceptive_obs_length",
    ]

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.actor_proprioceptive_obs_length = actor_critic.actor_proprioceptive_obs_length

    def forward(self, obs_history):
        parts = self.estimator(obs_history[:, 0:self.actor_proprioceptive_obs_length])
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        latest_obs = obs_history[:, -self.num_one_step_obs:]
        actor_input = torch.cat((latest_obs, vel, z), dim=1)
        return self.actor(actor_input)

    def export(self, path, filename='policy.pt'):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterHIMWithTerrain(torch.nn.Module):
    __constants__ = [
        "num_one_step_obs",
        "actor_proprioceptive_obs_length",
        "num_height_points",
    ]

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.terrain_encoder = copy.deepcopy(actor_critic.terrain_encoder)
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.actor_proprioceptive_obs_length = actor_critic.actor_proprioceptive_obs_length
        self.num_height_points = actor_critic.num_height_points

    def forward(self, obs_history):
        parts = self.estimator(obs_history[:, 0:self.actor_proprioceptive_obs_length])
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        terrain_input = obs_history[:, -(self.num_height_points + self.num_one_step_obs):]
        terrain_latent = self.terrain_encoder(terrain_input)
        latest_obs = obs_history[:, -(self.num_height_points + self.num_one_step_obs):-self.num_height_points]
        actor_input = torch.cat((latest_obs, vel, z, terrain_latent), dim=1)
        return self.actor(actor_input)

    def export(self, path, filename='policy.pt'):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
