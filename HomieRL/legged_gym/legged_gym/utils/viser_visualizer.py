"""Lightweight viser robot-state visualization for HomieRL training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import threading
import time
from typing import ClassVar, Dict, List, Optional, Sequence

import numpy as np

try:
    import viser  # type: ignore[import-not-found]
    import yourdfpy  # type: ignore[import-untyped]
    from viser.extras import ViserUrdf  # type: ignore[import-not-found]
except ImportError:
    viser = None
    yourdfpy = None
    ViserUrdf = None


@dataclass
class ViserRuntimeConfig:
    enabled: bool = False
    port: int = 8080
    env_idx: int = 0
    update_interval: int = 1
    show_meshes: bool = True


@dataclass
class EpisodeFrame:
    root_state: np.ndarray
    dof_state: np.ndarray
    env_origin: np.ndarray
    commands: Optional[np.ndarray] = None
    head_position: Optional[np.ndarray] = None
    base_height: Optional[float] = None
    height_scan_points: Optional[np.ndarray] = None
    foot_ref_scandot_values: Optional[np.ndarray] = None


def _ensure_path_is_relative_to() -> None:
    if hasattr(Path, "is_relative_to"):
        return

    def is_relative_to(self: Path, *other: object) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    setattr(Path, "is_relative_to", is_relative_to)


class RobotStateViser:
    """Viewer that streams live state, then replays completed episodes."""

    _global_servers: ClassVar[Dict[int, object]] = {}

    def __init__(
        self,
        *,
        urdf_path: str,
        num_envs: int,
        dt: float,
        config: ViserRuntimeConfig,
        sim_joint_names: Optional[Sequence[str]] = None,
        terrain_mesh: object = None,
        terrain_extent: Optional[tuple] = None,
    ) -> None:
        if viser is None or yourdfpy is None or ViserUrdf is None:
            raise ImportError("viser visualization requested, but `viser` or `yourdfpy` is not installed.")

        if config.port in self._global_servers:
            print(f"[viser] Stopping existing server on port {config.port} before starting a new one.")
            self._global_servers.pop(config.port).stop()

        self.config = config
        self.dt = float(dt)
        self.num_envs = max(1, int(num_envs))
        _ensure_path_is_relative_to()

        self.server = viser.ViserServer(port=config.port)
        self._global_servers[config.port] = self.server
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=True)
        self.terrain_root = self.server.scene.add_frame("/terrain", show_axes=False)
        self._terrain_handle = None
        self._height_scan_handle = None
        self._velocity_command_handle = None
        self._yaw_command_handle = None
        self._height_command_handle = None
        self._foot_ref_scandot_min_text = None
        self._foot_ref_scandot_max_text = None
        self._foot_ref_scandot_mean_text = None
        self._terrain_vertices, self._terrain_faces = self._get_terrain_arrays(terrain_mesh)
        if self._terrain_vertices is None:
            self.server.scene.add_grid("/terrain/grid", width=20.0, height=20.0, position=(0.0, 0.0, 0.0))
        self._terrain_extent = terrain_extent
        self._terrain_origin_key = None

        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=urdf_path)
        urdf = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=True,
            build_scene_graph=True,
            filename_handler=filename_handler,
        )
        self.robot = ViserUrdf(self.server, urdf_or_path=urdf, root_node_name="/robot")
        self.robot.show_visual = bool(config.show_meshes)
        actuated_joint_limits = self.robot.get_actuated_joint_limits()
        if hasattr(self.robot, "get_actuated_joint_names"):
            self._joint_names = list(self.robot.get_actuated_joint_names())
        else:
            self._joint_names = list(actuated_joint_limits.keys())
        self._expected_dof = len(actuated_joint_limits)
        self._sim_to_viser_indices = self._build_sim_to_viser_indices(sim_joint_names)
        self.robot.update_cfg(np.zeros(self._expected_dof, dtype=np.float64))

        self._selected_env_idx = int(np.clip(config.env_idx, 0, self.num_envs - 1))
        self._tracked_env_idx = self._selected_env_idx
        self._warned_dof_mismatch = False
        self._logged_first_pose = False
        self._latest_root_position = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._prev_root_position = self._latest_root_position.copy()
        self._render_mode = "live"

        self._recording_episode: List[EpisodeFrame] = []
        self._ready_episode: Optional[List[EpisodeFrame]] = None
        self._last_completed_episode: Optional[List[EpisodeFrame]] = None
        self._playing_episode: Optional[List[EpisodeFrame]] = None
        self._playing_index = 0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._player_thread = threading.Thread(target=self._player_loop, daemon=True, name="HomieRLViserPlayer")

        self._build_gui()
        self._setup_camera_on_connect()
        self._player_thread.start()

        print(f"[viser] Started robot viewer at http://localhost:{config.port} (env_idx={self._selected_env_idx})")
        print(f"[viser] Loaded URDF: {urdf_path} (actuated_dof={self._expected_dof})")

    def _build_sim_to_viser_indices(self, sim_joint_names: Optional[Sequence[str]]) -> Optional[np.ndarray]:
        if sim_joint_names is None:
            return None

        sim_joint_to_idx = {name: idx for idx, name in enumerate(sim_joint_names)}
        missing = [name for name in self._joint_names if name not in sim_joint_to_idx]
        if missing:
            print(f"[viser] Joint name mismatch. Missing simulator joints for viser: {missing}")
            return None

        indices = np.array([sim_joint_to_idx[name] for name in self._joint_names], dtype=np.int64)
        if not np.array_equal(indices, np.arange(len(indices))):
            print("[viser] Reordering simulator DOF positions into URDF/viser joint order.")
        return indices

    def _get_terrain_arrays(self, terrain_mesh: object) -> tuple:
        if terrain_mesh is None:
            return None, None

        vertices = getattr(terrain_mesh, "vertices", None)
        faces = getattr(terrain_mesh, "faces", None)
        if faces is None:
            faces = getattr(terrain_mesh, "triangles", None)
        if vertices is None or faces is None:
            print("[viser] Skipping terrain mesh because it does not expose vertices/faces.")
            return None, None

        vertices_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.uint32)
        if vertices_np.size == 0 or faces_np.size == 0:
            print("[viser] Skipping empty terrain mesh.")
            return None, None
        cfg = getattr(terrain_mesh, "cfg", None)
        border_size = float(getattr(cfg, "border_size", 0.0)) if cfg is not None else 0.0
        vertices_np = vertices_np.copy()
        vertices_np[:, 0] -= border_size
        vertices_np[:, 1] -= border_size
        return vertices_np, faces_np

    def _update_terrain_mesh(self, env_origin: np.ndarray) -> None:
        if self._terrain_vertices is None or self._terrain_faces is None or self._terrain_extent is None:
            return

        origin = np.asarray(env_origin, dtype=np.float32)
        origin_key = tuple(np.round(origin, decimals=4).tolist())
        if origin_key == self._terrain_origin_key:
            return
        self._terrain_origin_key = origin_key

        terrain_length, terrain_width = self._terrain_extent
        padding = 0.5
        vertices = self._terrain_vertices
        local_xy = vertices[:, :2] - origin[:2]
        vertex_mask = (
            (np.abs(local_xy[:, 0]) <= terrain_length / 2.0 + padding)
            & (np.abs(local_xy[:, 1]) <= terrain_width / 2.0 + padding)
        )
        face_mask = vertex_mask[self._terrain_faces].all(axis=1)
        cropped_faces = self._terrain_faces[face_mask]
        if cropped_faces.size == 0:
            print(f"[viser] No terrain mesh faces found near env origin {np.round(origin, 4).tolist()}.")
            return

        used_vertices = np.unique(cropped_faces.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
        cropped_vertices = vertices[used_vertices].copy()
        cropped_vertices -= origin
        cropped_faces = remap[cropped_faces].astype(np.uint32, copy=False)

        if self._terrain_handle is not None:
            self._terrain_handle.remove()
        self._terrain_handle = self.server.scene.add_mesh_simple(
            "/terrain/mesh",
            vertices=cropped_vertices,
            faces=cropped_faces,
            color=(120, 120, 120),
            opacity=0.75,
            side="double",
        )
        print(
            "[viser] Loaded cropped terrain mesh: "
            f"origin={np.round(origin, 4).tolist()}, vertices={len(cropped_vertices)}, faces={len(cropped_faces)}"
        )

    def _build_gui(self) -> None:
        with self.server.gui.add_folder("Robot Viewer"):
            self.env_idx_slider = self.server.gui.add_slider(
                "Env idx",
                min=0,
                max=self.num_envs - 1,
                step=1,
                initial_value=self._selected_env_idx,
            )
            self.recenter_cb = self.server.gui.add_checkbox("Recenter env", initial_value=True)
            self.show_meshes_cb = self.server.gui.add_checkbox("Show meshes", initial_value=self.config.show_meshes)
            self.follow_robot_cb = self.server.gui.add_checkbox("Follow robot", initial_value=False)
            self.mode_text = self.server.gui.add_text("Mode", initial_value="live")

        @self.env_idx_slider.on_update
        def _(_event) -> None:
            with self._lock:
                self._selected_env_idx = int(self.env_idx_slider.value)
                self._tracked_env_idx = self._selected_env_idx
                self._recording_episode = []
                self._ready_episode = None
                self._last_completed_episode = None
                self._playing_episode = None
                self._playing_index = 0
                self._render_mode = "live"
                self.mode_text.value = self._render_mode

        @self.show_meshes_cb.on_update
        def _(_event) -> None:
            self.robot.show_visual = bool(self.show_meshes_cb.value)

    def _setup_camera_on_connect(self) -> None:
        on_connect = getattr(self.server, "on_client_connect", None)
        if on_connect is None:
            return

        @on_connect
        def _(client) -> None:
            root = self._latest_root_position
            client.camera.look_at = tuple(root.tolist())
            client.camera.position = tuple((root + np.array([2.5, 2.5, 1.5], dtype=np.float64)).tolist())
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def update(
        self,
        *,
        root_states: np.ndarray,
        dof_pos: np.ndarray,
        dones: np.ndarray,
        env_origins: Optional[np.ndarray] = None,
        commands: Optional[np.ndarray] = None,
        head_positions: Optional[np.ndarray] = None,
        base_heights: Optional[np.ndarray] = None,
        height_scan_points: Optional[np.ndarray] = None,
        foot_ref_scandot_values: Optional[np.ndarray] = None,
    ) -> None:
        env_idx = int(np.clip(self._selected_env_idx, 0, min(len(root_states), len(dof_pos)) - 1))
        origin = (
            np.asarray(env_origins[env_idx], dtype=np.float64).copy()
            if env_origins is not None
            else np.zeros(3, dtype=np.float64)
        )
        scan_points = (
            np.asarray(
                height_scan_points[env_idx if len(height_scan_points) == len(root_states) else 0],
                dtype=np.float32,
            ).copy()
            if height_scan_points is not None
            else None
        )
        foot_ref_values = (
            np.asarray(
                foot_ref_scandot_values[env_idx if len(foot_ref_scandot_values) == len(root_states) else 0],
                dtype=np.float32,
            ).reshape(-1).copy()
            if foot_ref_scandot_values is not None
            else None
        )
        command = (
            np.asarray(
                commands[env_idx if len(commands) == len(root_states) else 0],
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
            if commands is not None
            else None
        )
        head_position = (
            np.asarray(
                head_positions[env_idx if len(head_positions) == len(root_states) else 0],
                dtype=np.float64,
            ).copy()
            if head_positions is not None
            else None
        )
        base_height = (
            float(
                np.asarray(
                    base_heights[env_idx if len(base_heights) == len(root_states) else 0]
                ).item()
            )
            if base_heights is not None
            else None
        )
        frame = EpisodeFrame(
            root_state=np.asarray(root_states[env_idx], dtype=np.float64).copy(),
            dof_state=np.asarray(dof_pos[env_idx], dtype=np.float64).reshape(-1).copy(),
            env_origin=origin,
            commands=command,
            head_position=head_position,
            base_height=base_height,
            height_scan_points=scan_points,
            foot_ref_scandot_values=foot_ref_values,
        )
        done = bool(np.asarray(dones[env_idx]).item())

        should_render_live = False
        with self._lock:
            if env_idx != self._tracked_env_idx:
                self._tracked_env_idx = env_idx
                self._recording_episode = []
                self._ready_episode = None
                self._last_completed_episode = None
                self._playing_episode = None
                self._playing_index = 0
                self._render_mode = "live"
                self.mode_text.value = self._render_mode

            self._recording_episode.append(frame)
            if self._last_completed_episode is None:
                should_render_live = True

            if done and self._recording_episode:
                completed_episode = self._recording_episode
                self._recording_episode = []
                self._ready_episode = completed_episode
                if self._last_completed_episode is None:
                    self._last_completed_episode = completed_episode
                print(f"[viser] Queued completed episode for env_idx={env_idx} with {len(completed_episode)} frames.")

        if should_render_live:
            self._render_frame(frame, mode="live")

    def _player_loop(self) -> None:
        while not self._stop_event.is_set():
            frame_to_render = None
            mode = "live"
            with self._lock:
                if self._last_completed_episode is not None:
                    if self._playing_episode is None:
                        self._playing_episode = self._ready_episode or self._last_completed_episode
                        self._ready_episode = None
                        self._playing_index = 0

                    if self._playing_episode:
                        frame_to_render = self._playing_episode[self._playing_index]
                        mode = "replay"
                        self._playing_index += 1
                        if self._playing_index >= len(self._playing_episode):
                            self._last_completed_episode = self._playing_episode
                            self._playing_episode = None
                            self._playing_index = 0

            if frame_to_render is not None:
                self._render_frame(frame_to_render, mode=mode)
                time.sleep(self.dt)
            else:
                time.sleep(min(self.dt, 0.01))

    def _render_frame(self, frame: EpisodeFrame, *, mode: str) -> None:
        root_state = frame.root_state
        dof_state = frame.dof_state
        if root_state.shape[0] < 7:
            raise ValueError(f"Expected root state with at least 7 elements, got shape {root_state.shape}.")

        quat_xyzw = root_state[3:7]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        display_pos = root_state[:3].copy()
        if bool(self.recenter_cb.value):
            display_pos -= frame.env_origin

        self._prev_root_position = self._latest_root_position.copy()
        self._latest_root_position = display_pos.copy()
        self.robot_root.position = tuple(display_pos.tolist())
        self.robot_root.wxyz = tuple(quat_wxyz.tolist())
        terrain_pos = np.zeros(3, dtype=np.float64) if bool(self.recenter_cb.value) else frame.env_origin
        self.terrain_root.position = tuple(terrain_pos.tolist())
        self._update_terrain_mesh(frame.env_origin)
        self._follow_robot_camera()
        self._update_command_visuals(frame, display_pos)
        self._update_height_scan(frame)
        self._update_foot_ref_scandot_text(frame)

        if not self._logged_first_pose:
            print(
                "[viser] First pose update: "
                f"env_idx={self._tracked_env_idx}, "
                f"root_pos_world={np.round(root_state[:3], 4).tolist()}, "
                f"root_pos_display={np.round(display_pos, 4).tolist()}"
            )
            self._logged_first_pose = True

        expected_input_dof = len(self._sim_to_viser_indices) if self._sim_to_viser_indices is not None else self._expected_dof
        if dof_state.shape[0] != expected_input_dof and not self._warned_dof_mismatch:
            print(
                f"[viser] Robot DOF mismatch: simulator has {dof_state.shape[0]}, "
                f"viewer mapping expects {expected_input_dof}. Using the overlapping prefix."
            )
            self._warned_dof_mismatch = True

        if self._sim_to_viser_indices is not None and dof_state.shape[0] >= len(self._sim_to_viser_indices):
            cfg = dof_state[self._sim_to_viser_indices]
        else:
            cfg = dof_state[: self._expected_dof]
        self.robot.update_cfg(cfg)
        if mode != self._render_mode:
            self._render_mode = mode
            self.mode_text.value = mode

    def _update_command_visuals(self, frame: EpisodeFrame, display_pos: np.ndarray) -> None:
        if frame.commands is None:
            return
        if frame.commands.shape[0] < 5:
            return

        origin_offset = frame.env_origin if bool(self.recenter_cb.value) else np.zeros(3, dtype=np.float64)
        anchor = (
            frame.head_position.copy() - origin_offset
            if frame.head_position is not None
            else display_pos.copy()
        )
        anchor[2] += 1.2

        self._update_velocity_command(frame, anchor)
        self._update_yaw_command(frame, anchor)
        self._update_height_command(frame, display_pos)

    def _update_velocity_command(self, frame: EpisodeFrame, anchor: np.ndarray) -> None:
        lin_cmd = np.array([frame.commands[0], frame.commands[1], 0.0], dtype=np.float64)
        world_cmd = self._rotate_by_root_yaw(lin_cmd, frame.root_state[3:7])
        length = np.linalg.norm(world_cmd[:2])
        if length > 1e-8:
            direction = world_cmd / length
            tip = anchor + world_cmd
            left = self._rotate_z(direction, 2.5)
            right = self._rotate_z(direction, -2.5)
            arrow_head_len = min(0.2, max(0.05, 0.25 * length))
            points = np.array(
                [
                    [anchor, tip],
                    [tip, tip - arrow_head_len * left],
                    [tip, tip - arrow_head_len * right],
                ],
                dtype=np.float32,
            )
        else:
            points = np.zeros((3, 2, 3), dtype=np.float32)
            points[:, :, :] = anchor.astype(np.float32)
        colors = np.zeros_like(points, dtype=np.float32)
        colors[:] = np.array([0.0, 180.0, 255.0], dtype=np.float32)
        if self._velocity_command_handle is None:
            self._velocity_command_handle = self.server.scene.add_line_segments(
                "/commands/velocity_xy",
                points=points,
                colors=colors,
                line_width=4.0,
            )
        else:
            self._velocity_command_handle.points = points
            self._velocity_command_handle.colors = colors

    def _update_yaw_command(self, frame: EpisodeFrame, anchor: np.ndarray) -> None:
        yaw_cmd = float(frame.commands[2])
        start = anchor + np.array([0.35, 0.0, 0.0], dtype=np.float64)
        z_cmd = -yaw_cmd
        end = start + np.array([0.0, 0.0, z_cmd], dtype=np.float64)
        if abs(z_cmd) > 1e-8:
            direction = 1.0 if z_cmd >= 0.0 else -1.0
            arrow_head_len = min(0.2, max(0.05, 0.25 * abs(z_cmd)))
        else:
            direction = 1.0
            arrow_head_len = 0.0
        points = np.array(
            [
                [start, end],
                [end, end + np.array([arrow_head_len, 0.0, -direction * arrow_head_len], dtype=np.float64)],
                [end, end + np.array([-arrow_head_len, 0.0, -direction * arrow_head_len], dtype=np.float64)],
            ],
            dtype=np.float32,
        )
        colors = np.zeros_like(points, dtype=np.float32)
        colors[:] = np.array([255.0, 160.0, 0.0], dtype=np.float32)
        if self._yaw_command_handle is None:
            self._yaw_command_handle = self.server.scene.add_line_segments(
                "/commands/yaw",
                points=points,
                colors=colors,
                line_width=4.0,
            )
        else:
            self._yaw_command_handle.points = points
            self._yaw_command_handle.colors = colors

    def _update_height_command(self, frame: EpisodeFrame, display_pos: np.ndarray) -> None:
        if frame.base_height is None:
            return
        current_height = float(frame.base_height)
        command_height = float(frame.commands[4])
        ground_z = float(display_pos[2]) - current_height
        center_xy = display_pos[:2]
        current_z = ground_z + current_height
        command_z = ground_z + command_height
        half_x = 0.35
        half_y = 0.25
        points = np.array(
            [
                self._rectangle_segments(center_xy, current_z, half_x, half_y),
                self._rectangle_segments(center_xy, command_z, half_x, half_y),
            ],
            dtype=np.float32,
        ).reshape(8, 2, 3)
        colors = np.zeros_like(points, dtype=np.float32)
        colors[:4] = np.array([255.0, 80.0, 200.0], dtype=np.float32)
        colors[4:] = np.array([160.0, 255.0, 80.0], dtype=np.float32)
        if self._height_command_handle is None:
            self._height_command_handle = self.server.scene.add_line_segments(
                "/commands/base_height",
                points=points,
                colors=colors,
                line_width=3.0,
            )
        else:
            self._height_command_handle.points = points
            self._height_command_handle.colors = colors

    def _rectangle_segments(self, center_xy: np.ndarray, z: float, half_x: float, half_y: float) -> np.ndarray:
        corners = np.array(
            [
                [center_xy[0] - half_x, center_xy[1] - half_y, z],
                [center_xy[0] + half_x, center_xy[1] - half_y, z],
                [center_xy[0] + half_x, center_xy[1] + half_y, z],
                [center_xy[0] - half_x, center_xy[1] + half_y, z],
            ],
            dtype=np.float64,
        )
        return np.array(
            [
                [corners[0], corners[1]],
                [corners[1], corners[2]],
                [corners[2], corners[3]],
                [corners[3], corners[0]],
            ],
            dtype=np.float64,
        )

    def _rotate_by_root_yaw(self, vector: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return self._rotate_z(vector, yaw)

    def _rotate_z(self, vector: np.ndarray, yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array(
            [
                c * vector[0] - s * vector[1],
                s * vector[0] + c * vector[1],
                vector[2],
            ],
            dtype=np.float64,
        )

    def _update_height_scan(self, frame: EpisodeFrame) -> None:
        if frame.height_scan_points is None:
            return

        points = frame.height_scan_points.astype(np.float32, copy=True)
        points -= frame.env_origin.astype(np.float32, copy=False)
        colors = np.zeros_like(points, dtype=np.float32)
        colors[:] = np.array([255.0, 220.0, 0.0], dtype=np.float32)

        if self._height_scan_handle is None:
            self._height_scan_handle = self.server.scene.add_point_cloud(
                "/terrain/height_scan",
                points=points,
                colors=colors,
                point_size=0.035,
                point_shape="circle",
            )
        else:
            self._height_scan_handle.points = points
            self._height_scan_handle.colors = colors

    def _update_foot_ref_scandot_text(self, frame: EpisodeFrame) -> None:
        if frame.foot_ref_scandot_values is None:
            return

        values = frame.foot_ref_scandot_values
        min_text = f"{np.min(values):.3f}"
        max_text = f"{np.max(values):.3f}"
        mean_text = f"{np.mean(values):.3f}"
        if self._foot_ref_scandot_min_text is None:
            self._foot_ref_scandot_min_text = self.server.gui.add_text("Foot-ref scandot min", initial_value=min_text)
            self._foot_ref_scandot_max_text = self.server.gui.add_text("Foot-ref scandot max", initial_value=max_text)
            self._foot_ref_scandot_mean_text = self.server.gui.add_text("Foot-ref scandot mean", initial_value=mean_text)
        else:
            self._foot_ref_scandot_min_text.value = min_text
            self._foot_ref_scandot_max_text.value = max_text
            self._foot_ref_scandot_mean_text.value = mean_text

    def _follow_robot_camera(self) -> None:
        if not bool(self.follow_robot_cb.value):
            return
        delta = self._latest_root_position - self._prev_root_position
        if np.linalg.norm(delta) < 1e-8:
            return
        clients = self.server.get_clients()
        for client in clients.values():
            try:
                cam_pos = np.array(client.camera.position, dtype=np.float64)
                cam_look = np.array(client.camera.look_at, dtype=np.float64)
                client.camera.position = tuple((cam_pos + delta).tolist())
                client.camera.look_at = tuple((cam_look + delta).tolist())
            except Exception:
                pass

    def close(self) -> None:
        self._stop_event.set()
        if self._player_thread.is_alive():
            self._player_thread.join(timeout=1.0)
