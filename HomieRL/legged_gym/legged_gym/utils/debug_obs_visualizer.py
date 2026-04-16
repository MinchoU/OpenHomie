"""Standalone visualization of saved critic observations via viser.

Usage:
  1. In pdb, save tensors:
     !torch.save({'bad': critic_obs_batch[52816].cpu(), 'good': critic_obs_batch[0].cpu()}, '/tmp/debug_obs.pt')

  2. In a separate terminal:
     python debug_obs_visualizer.py /tmp/debug_obs.pt
     # or with custom port:
     python debug_obs_visualizer.py /tmp/debug_obs.pt --port 8090

  3. Open http://localhost:8090 in browser.
"""

# Remove script directory from sys.path to prevent local math.py from
# shadowing the stdlib math module (which breaks numpy/datetime imports).
import sys as _sys, os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if _script_dir in _sys.path:
    _sys.path.remove(_script_dir)

import argparse
import time
import numpy as np
import torch
from functools import partial
from pathlib import Path

try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError:
    raise ImportError("viser/yourdfpy not installed. pip install viser yourdfpy")

# Compat shim for Python < 3.9
if not hasattr(Path, "is_relative_to"):
    def _is_relative_to(self, *other):
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False
    Path.is_relative_to = _is_relative_to

# -- G1 rough custom3 critic observation layout (critic_history=1) --
# [0:3]    cmd_scaled          (commands[:,:3] * commands_scale)
# [3:4]    height_cmd          (commands[:,4])
# [4:7]    imu_ang_vel         (* obs_scales.ang_vel=0.5)
# [7:10]   imu_proj_gravity    (unscaled, unit vector)
# [10:37]  dof_pos_rel         (* obs_scales.dof_pos=1.0)
# [37:64]  dof_vel             (* obs_scales.dof_vel=0.05)
# [64:76]  actions             (raw, 12 lower-body actions)
# [76:79]  base_lin_vel        (* obs_scales.lin_vel=2.0)
# [79:80]  rough_gate          (0 or 1)
# [80:267] scandots            (clip(robot_z-0.5-terrain_z, -1, 1) * 5.0)

SCANDOT_X = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                       0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
SCANDOT_Y = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
NX, NY = len(SCANDOT_X), len(SCANDOT_Y)  # 17, 11

SCALE_DOF_POS = 1.0
SCALE_DOF_VEL = 0.05
SCALE_ANG_VEL = 0.5
SCALE_LIN_VEL = 2.0
SCALE_HEIGHT = 5.0

ROBOT_Z = 0.75  # nominal init height

# Isaac Gym DOF order for g1.urdf (27 revolute joints, URDF order)
SIM_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

DEFAULT_DOF_POS = np.array([
    -0.1, 0., 0., 0.3, -0.2, 0.,    # left leg
    -0.1, 0., 0., 0.3, -0.2, 0.,    # right leg
    0.,                               # waist
    0., 0., 0., 0., 0., 0., 0.,      # left arm
    0., 0., 0., 0., 0., 0., 0.,      # right arm
])

URDF_PATH = '/home/cmw9903/gauss_gym/holosoma/third_party/OpenHomie/HomieRL/legged_gym/resources/robots/g1_description/g1.urdf'


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def decode_critic_obs(obs):
    """Decode a 267-dim critic observation into semantic components."""
    o = _to_numpy(obs).ravel()
    assert o.shape[0] == 267, "Expected 267 dims, got {}".format(o.shape[0])

    d = {}
    d['cmd_scaled'] = o[0:3]
    d['height_cmd'] = o[3]
    d['ang_vel_raw'] = o[4:7] / SCALE_ANG_VEL
    d['ang_vel_scaled'] = o[4:7]
    d['proj_gravity'] = o[7:10]
    d['dof_pos_rel'] = o[10:37] / SCALE_DOF_POS
    d['dof_pos'] = d['dof_pos_rel'] + DEFAULT_DOF_POS
    d['dof_vel_raw'] = o[37:64] / SCALE_DOF_VEL
    d['dof_vel_scaled'] = o[37:64]
    d['actions'] = o[64:76]
    d['base_lin_vel_raw'] = o[76:79] / SCALE_LIN_VEL
    d['base_lin_vel_scaled'] = o[76:79]
    d['rough_gate'] = o[79]
    d['scandots_scaled'] = o[80:267]
    d['scandots_raw'] = o[80:267] / SCALE_HEIGHT  # clip(robot_z-0.5-terrain_z, -1, 1)
    return d


def _proj_gravity_to_wxyz(pg):
    """Recover base orientation (wxyz quaternion) from projected gravity.

    proj_gravity = R^T @ [0,0,-1]  so  R @ pg = [0,0,-1].
    We find the minimal rotation R (no yaw component).
    """
    pg = np.asarray(pg, dtype=np.float64)
    norm = np.linalg.norm(pg)
    if norm < 1e-8:
        return np.array([1., 0., 0., 0.])
    pg = pg / norm
    target = np.array([0., 0., -1.])
    dot = np.dot(pg, target)
    if dot > 0.9999:
        return np.array([1., 0., 0., 0.])
    if dot < -0.9999:
        return np.array([0., 1., 0., 0.])
    axis = np.cross(pg, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1., 1.))
    w = np.cos(angle / 2.)
    xyz = axis * np.sin(angle / 2.)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def _build_terrain_mesh(scandots_raw):
    """Build a triangulated terrain mesh from 187 scandot values.

    scandots_raw = clip(robot_z - 0.5 - terrain_z, -1, 1)
    terrain_z = robot_z - 0.5 - scandots_raw
    """
    grid_x, grid_y = np.meshgrid(SCANDOT_X, SCANDOT_Y, indexing='ij')
    terrain_z = ROBOT_Z - 0.5 - scandots_raw.reshape(NX, NY)

    vertices = np.stack([grid_x, grid_y, terrain_z], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(NX - 1):
        for j in range(NY - 1):
            v00 = i * NY + j
            v01 = i * NY + (j + 1)
            v10 = (i + 1) * NY + j
            v11 = (i + 1) * NY + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.uint32)

    return vertices.astype(np.float32), faces


def _height_to_color(terrain_z):
    """Color terrain: green=normal ground, red=above robot (anomaly), blue=deep below."""
    z = terrain_z.ravel()
    colors = np.zeros((len(z), 3), dtype=np.float32)
    for i, h in enumerate(z):
        if h > ROBOT_Z - 0.3:
            # terrain near or above robot -> red (anomalous)
            t = min(1.0, max(0.0, (h - (ROBOT_Z - 0.3)) / 0.5))
            colors[i] = [200 + 55 * t, 50 * (1 - t), 50 * (1 - t)]
        elif h > -0.1:
            # normal ground level -> green
            t = min(1.0, max(0.0, (h + 0.1) / (ROBOT_Z - 0.3 + 0.1)))
            colors[i] = [50 * t, 150 + 50 * t, 50]
        else:
            # deep below -> blue
            colors[i] = [50, 50, 180]
    return colors


def _build_sim_to_viser(viser_joint_names):
    """Map Isaac Gym DOF indices to ViserUrdf joint indices."""
    sim_name_to_idx = {n: i for i, n in enumerate(SIM_JOINT_NAMES)}
    indices = []
    for name in viser_joint_names:
        if name in sim_name_to_idx:
            indices.append(sim_name_to_idx[name])
        else:
            indices.append(-1)
    return indices


def _print_decoded(d, label=""):
    prefix = "[{}] ".format(label) if label else ""
    print("")
    print("=" * 60)
    print("{}Decoded critic observation".format(prefix))
    print("=" * 60)
    print("  cmd_scaled:     {}".format(np.round(d['cmd_scaled'], 3)))
    print("  height_cmd:     {:.3f}".format(d['height_cmd']))
    print("  ang_vel (raw):  {} rad/s".format(np.round(d['ang_vel_raw'], 3)))
    print("  proj_gravity:   {}".format(np.round(d['proj_gravity'], 4)))
    print("  base_lin_vel:   {} m/s".format(np.round(d['base_lin_vel_raw'], 3)))
    print("  rough_gate:     {:.1f}".format(d['rough_gate']))
    print("  scandots:       min={:.3f} max={:.3f} mean={:.3f}".format(
        d['scandots_scaled'].min(), d['scandots_scaled'].max(), d['scandots_scaled'].mean()))
    print("  scandots (raw): min={:.3f} max={:.3f}".format(
        d['scandots_raw'].min(), d['scandots_raw'].max()))

    print("  Joint positions (deg):")
    for i, name in enumerate(SIM_JOINT_NAMES):
        deg = np.degrees(d['dof_pos'][i])
        rel_deg = np.degrees(d['dof_pos_rel'][i])
        if abs(rel_deg) > 5:
            flag = " <<<" if abs(rel_deg) > 30 else " <"
        else:
            flag = ""
        short = name.replace('_joint', '').replace('_', ' ')
        print("    [{:2d}] {:30s} = {:7.1f} deg  (d={:+.1f} deg){}".format(
            i, short, deg, rel_deg, flag))
    print("")


def debug_vis(obs_bad, obs_good=None, port=8090):
    """Visualize critic observation(s) in viser.

    Args:
        obs_bad:  267-dim tensor/array (the observation to inspect)
        obs_good: optional 267-dim tensor/array (reference normal observation)
        port:     viser server port

    Returns:
        viser.ViserServer (call .stop() to shut down)
    """
    d_bad = decode_critic_obs(obs_bad)
    _print_decoded(d_bad, "BAD")
    if obs_good is not None:
        d_good = decode_critic_obs(obs_good)
        _print_decoded(d_good, "GOOD")

    server = viser.ViserServer(port=port)

    # Load URDF
    filename_handler = partial(
        yourdfpy.filename_handler_relative_to_urdf_file,
        urdf_fname=URDF_PATH,
    )
    urdf = yourdfpy.URDF.load(
        URDF_PATH,
        load_meshes=True,
        build_scene_graph=True,
        filename_handler=filename_handler,
    )

    def add_scene(d, prefix, x_offset=0.0):
        # Robot root frame with orientation from proj_gravity
        quat_wxyz = _proj_gravity_to_wxyz(d['proj_gravity'])
        server.scene.add_frame(
            "{}/robot".format(prefix),
            show_axes=True,
            position=(x_offset, 0.0, ROBOT_Z),
            wxyz=tuple(quat_wxyz.tolist()),
        )
        robot = ViserUrdf(server, urdf_or_path=urdf, root_node_name="{}/robot".format(prefix))

        # Map sim DOFs to viser joint order
        if hasattr(robot, 'get_actuated_joint_names'):
            viser_names = list(robot.get_actuated_joint_names())
        else:
            viser_names = list(robot.get_actuated_joint_limits().keys())
        sim_to_viser = _build_sim_to_viser(viser_names)

        # Set joint angles
        cfg = np.zeros(len(viser_names), dtype=np.float64)
        for vi, si in enumerate(sim_to_viser):
            if 0 <= si < 27:
                cfg[vi] = d['dof_pos'][si]
        robot.update_cfg(cfg)

        # Terrain mesh
        verts, faces = _build_terrain_mesh(d['scandots_raw'])
        verts[:, 0] += x_offset
        colors = _height_to_color(verts[:, 2])
        server.scene.add_mesh_simple(
            "{}/terrain".format(prefix),
            vertices=verts,
            faces=faces,
            color=(150, 150, 150),
            opacity=0.85,
            side="double",
        )
        # Scandot point cloud (colored by height)
        server.scene.add_point_cloud(
            "{}/scandots".format(prefix),
            points=verts,
            colors=colors,
            point_size=0.04,
            point_shape="circle",
        )

        # Label
        server.scene.add_label(
            "{}/label".format(prefix),
            text=prefix.strip('/'),
            position=(x_offset, 0.0, ROBOT_Z + 0.8),
        )

        return robot

    add_scene(d_bad, "/bad", x_offset=0.0)
    if obs_good is not None:
        add_scene(d_good, "/good", x_offset=3.0)

    # Ground grid
    server.scene.add_grid("/grid", width=8.0, height=4.0, position=(1.5, 0.0, -0.01))

    # Camera setup for connecting clients
    def _set_camera(client):
        client.camera.position = (2.0, -3.0, 2.0)
        client.camera.look_at = (0.0, 0.0, 0.5)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    for client in server.get_clients().values():
        _set_camera(client)
    if hasattr(server, 'on_client_connect'):
        @server.on_client_connect
        def _(client):
            _set_camera(client)

    n_scenes = "2 (bad + good)" if obs_good is not None else "1"
    print("")
    print("[debug_vis] Server running at http://localhost:{}".format(port))
    print("[debug_vis] Showing {} observation(s)".format(n_scenes))
    print("[debug_vis] Press Ctrl+C to stop")
    print("")

    return server


def main():
    parser = argparse.ArgumentParser(description="Visualize saved critic observations")
    parser.add_argument("pt_file", help="Path to .pt file with 'bad' (and optionally 'good') keys")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    data = torch.load(args.pt_file, map_location='cpu')
    obs_bad = data['bad']
    obs_good = data.get('good', None)

    server = debug_vis(obs_bad, obs_good, port=args.port)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
