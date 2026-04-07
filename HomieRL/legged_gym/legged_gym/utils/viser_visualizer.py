"""Lightweight viser robot-state visualization for HomieRL training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import threading
import time
from typing import ClassVar, Dict, List, Optional

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
        self.server.scene.add_grid("/terrain/grid", width=20.0, height=20.0, position=(0.0, 0.0, 0.0))

        filename_handler = partial(yourdfpy.filename_handler_relative_to_urdf_file, urdf_fname=urdf_path)
        urdf = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=True,
            build_scene_graph=True,
            filename_handler=filename_handler,
        )
        self.robot = ViserUrdf(self.server, urdf_or_path=urdf, root_node_name="/robot")
        self.robot.show_visual = bool(config.show_meshes)
        self._expected_dof = len(self.robot.get_actuated_joint_limits())
        self.robot.update_cfg(np.zeros(self._expected_dof, dtype=np.float64))

        self._selected_env_idx = int(np.clip(config.env_idx, 0, self.num_envs - 1))
        self._tracked_env_idx = self._selected_env_idx
        self._warned_dof_mismatch = False
        self._logged_first_pose = False
        self._latest_root_position = np.array([0.0, 0.0, 1.0], dtype=np.float64)
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
    ) -> None:
        env_idx = int(np.clip(self._selected_env_idx, 0, min(len(root_states), len(dof_pos)) - 1))
        origin = (
            np.asarray(env_origins[env_idx], dtype=np.float64).copy()
            if env_origins is not None
            else np.zeros(3, dtype=np.float64)
        )
        frame = EpisodeFrame(
            root_state=np.asarray(root_states[env_idx], dtype=np.float64).copy(),
            dof_state=np.asarray(dof_pos[env_idx], dtype=np.float64).reshape(-1).copy(),
            env_origin=origin,
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

        self._latest_root_position = display_pos.copy()
        self.robot_root.position = tuple(display_pos.tolist())
        self.robot_root.wxyz = tuple(quat_wxyz.tolist())
        self.terrain_root.position = tuple(np.zeros(3, dtype=np.float64).tolist())

        if not self._logged_first_pose:
            print(
                "[viser] First pose update: "
                f"env_idx={self._tracked_env_idx}, "
                f"root_pos_world={np.round(root_state[:3], 4).tolist()}, "
                f"root_pos_display={np.round(display_pos, 4).tolist()}"
            )
            self._logged_first_pose = True

        if dof_state.shape[0] != self._expected_dof and not self._warned_dof_mismatch:
            print(
                f"[viser] Robot DOF mismatch: simulator has {dof_state.shape[0]}, "
                f"URDF expects {self._expected_dof}. Using the overlapping prefix."
            )
            self._warned_dof_mismatch = True

        self.robot.update_cfg(dof_state[: self._expected_dof])
        if mode != self._render_mode:
            self._render_mode = mode
            self.mode_text.value = mode

    def close(self) -> None:
        self._stop_event.set()
        if self._player_thread.is_alive():
            self._player_thread.join(timeout=1.0)
