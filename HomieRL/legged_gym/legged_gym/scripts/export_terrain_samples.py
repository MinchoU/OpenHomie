"""Export custom5 terrain samples as PLY meshes for visual verification.

Usage (run inside the training env that has isaacgym importable):

    python legged_gym/legged_gym/scripts/export_terrain_samples.py \
        --out legged_gym/logs/custom5_terrain_samples \
        --seed 0

Generates:
    full_arena.ply
    subterrain_type<0..4>_level<0|5|9>.ply    (15 files)

Each individual subterrain PLY re-runs make_terrain with the specified
(choice, difficulty) so the export is reproducible given the seed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from isaacgym import terrain_utils  # noqa: E402

# Complete legged_gym.envs init first to avoid a utils<->envs circular import
# triggered when importing legged_gym.utils.* before legged_gym.envs.
import legged_gym.envs  # noqa: E402, F401

from legged_gym.utils.terrain import Terrain


# ----- custom5 terrain config (mirrors G143dofNoHandObsRoughCustom5TerrainCfg.terrain + pillars) -----

class _PillarCfg:
    enabled = True
    count_range = (0, 5)
    height_range = (0.25, 1.25)
    side_range = (0.4, 1.5)
    top_noise = 0.05
    platform_size = 3.0


class _Custom5TerrainCfg:
    mesh_type = "trimesh"
    horizontal_scale = 0.1
    vertical_scale = 0.005
    border_size = 25.0
    curriculum = True
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.0
    measure_heights = True
    measured_points_x = [0.0]
    measured_points_y = [0.0]
    selected = False
    terrain_kwargs = None
    max_init_terrain_level = 0
    terrain_length = 8.0
    terrain_width = 8.0
    num_rows = 10
    num_cols = 20
    terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
    slope_treshold = 0.75
    pillars = _PillarCfg()


# ----- PLY writer (ASCII, pure numpy -- no trimesh dependency) -----

def _write_ply_ascii(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Write an ASCII PLY with vertices [N,3] float32 and triangles [M,3] int32."""
    n_v = vertices.shape[0]
    n_f = triangles.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("w") as f:
        f.write(header)
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


# ----- individual subterrain export -----

def _export_subterrain_ply(terrain_obj, out_path: Path) -> None:
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        terrain_obj.height_field_raw,
        terrain_obj.horizontal_scale,
        terrain_obj.vertical_scale,
        slope_threshold=0.75,
    )
    _write_ply_ascii(out_path, vertices, triangles)


def _render_single_subterrain(choice: float, difficulty: float, seed: int):
    """Run Terrain.make_terrain in isolation for one subterrain."""
    np.random.seed(seed)  # for top_noise / make_terrain internals
    # We construct a Terrain just to get access to make_terrain; it generates
    # its own internal grid but we only care about a single SubTerrain.
    # Use a tiny 1x1 grid for speed.
    cfg = _Custom5TerrainCfg()
    cfg.num_rows = 1
    cfg.num_cols = 1
    t = Terrain(cfg, num_robots=1)
    # The curriculum path already called make_terrain for (0, 0) with
    # difficulty 0 / choice 0; we re-run make_terrain to get the precise
    # (choice, difficulty) we want.
    sub = t.make_terrain(choice, difficulty)
    return sub


# ----- main -----

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("legged_gym/logs/custom5_terrain_samples"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- full arena ----
    np.random.seed(args.seed)
    cfg = _Custom5TerrainCfg()
    terrain = Terrain(cfg, num_robots=1)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        terrain.height_field_raw,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        slope_threshold=0.75,
    )
    _write_ply_ascii(args.out / "full_arena.ply", vertices, triangles)
    print(f"Wrote {args.out / 'full_arena.ply'}: {vertices.shape[0]} verts, {triangles.shape[0]} tris")

    # ---- sanity: no raised cells inside each subterrain's spawn platform ----
    hs = cfg.horizontal_scale
    half_plat = _PillarCfg.platform_size / 2.0
    border = cfg.border_size / hs
    plat_half_cells = int(half_plat / hs)
    for row in range(cfg.num_rows):
        for col in range(cfg.num_cols):
            sub_origin_x = border + (row + 0.5) * cfg.terrain_length / hs
            sub_origin_y = border + (col + 0.5) * cfg.terrain_width / hs
            xs = slice(int(sub_origin_x - plat_half_cells), int(sub_origin_x + plat_half_cells))
            ys = slice(int(sub_origin_y - plat_half_cells), int(sub_origin_y + plat_half_cells))
            platform_block = terrain.height_field_raw[xs, ys]
            max_here = platform_block.max() * cfg.vertical_scale
            # Platform may be raised by the underlying terrain (slopes peak at center!),
            # but pillars should not dominate. Warn if > 0.5 m.
            if max_here > 0.5:
                print(f"  [warn] subterrain ({row},{col}) platform max height = {max_here:.2f} m")

    # ---- individual subterrains for 5 terrain types x 3 levels ----
    levels = [0, 5, 9]
    # terrain_proportions boundaries: [0.1, 0.3, 0.6, 0.9, 1.0]
    # type_choice mapping (choice in [0,1]):
    #   < 0.1 -> smooth slope
    #   < 0.3 -> rough slope
    #   < 0.6 -> stairs up
    #   < 0.9 -> stairs down
    #   else   -> discrete obstacles
    choice_per_type = [0.05, 0.20, 0.45, 0.75, 0.95]
    type_names = ["smooth_slope", "rough_slope", "stairs_up", "stairs_down", "discrete"]

    for t_idx, (choice, tname) in enumerate(zip(choice_per_type, type_names)):
        for lvl in levels:
            difficulty = lvl / max(1, cfg.num_rows - 1)
            sub = _render_single_subterrain(choice, difficulty, seed=args.seed + 100 * t_idx + lvl)
            fname = args.out / f"subterrain_type{t_idx}_{tname}_level{lvl}.ply"
            _export_subterrain_ply(sub, fname)
            n_pillars = len(getattr(sub, "pillars", []))
            print(f"Wrote {fname}: {n_pillars} pillars")


if __name__ == "__main__":
    main()
