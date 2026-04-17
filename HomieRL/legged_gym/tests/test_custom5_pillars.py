"""Unit tests for custom5 pillar terrain generation helpers."""

import numpy as np
import pytest

from legged_gym.utils.terrain import _rasterize_pillars


# ---------- helpers ----------

def _empty_heightfield(width_cells: int = 80, length_cells: int = 80) -> np.ndarray:
    """Return a zeroed int16 heightfield of the given dimensions."""
    return np.zeros((width_cells, length_cells), dtype=np.int16)


# ---------- tests ----------

def test_rasterize_zero_pillars_leaves_field_unchanged():
    """When count_range=(0, 0) we must not touch any cell."""
    hf = _empty_heightfield()
    rng = np.random.default_rng(0)
    pillars = _rasterize_pillars(
        height_field_raw=hf,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        count_range=(0, 0),
        height_range=(0.25, 1.25),
        side_range=(0.4, 1.5),
        top_noise=0.05,
        platform_size=3.0,
        rng=rng,
    )
    assert pillars == []
    assert np.all(hf == 0)


def test_rasterize_exactly_one_pillar_raises_some_cells():
    """A forced 1-pillar run must leave pillar cells strictly taller than zero."""
    hf = _empty_heightfield()
    rng = np.random.default_rng(42)
    pillars = _rasterize_pillars(
        height_field_raw=hf,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        count_range=(1, 1),
        height_range=(0.5, 0.5),
        side_range=(1.0, 1.0),
        top_noise=0.0,
        platform_size=3.0,
        rng=rng,
    )
    assert len(pillars) == 1
    cx, cy, side, yaw = pillars[0]
    assert side == pytest.approx(1.0)
    # at least some cells must have been raised
    raised = hf > 0
    assert raised.any()
    # target height 0.5 m / vertical_scale 0.005 = 100 units
    max_cell = hf.max()
    assert max_cell == 100


def test_rasterize_respects_spawn_platform():
    """No raised cell may lie inside the 3 m x 3 m spawn platform at center."""
    hf = _empty_heightfield(width_cells=80, length_cells=80)  # 8 m x 8 m
    rng = np.random.default_rng(7)
    _rasterize_pillars(
        height_field_raw=hf,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        count_range=(5, 5),     # force max pillars to stress the check
        height_range=(0.25, 1.25),
        side_range=(0.4, 1.5),
        top_noise=0.0,
        platform_size=3.0,
        rng=rng,
    )
    W, L = hf.shape
    # 80 x 0.1 m = 8 m terrain, center at (4, 4) m, half-platform = 1.5 m
    # center cell range: [25, 54] inclusive (2.5-5.5 m)
    platform_mask = np.zeros_like(hf, dtype=bool)
    platform_mask[25:55, 25:55] = True
    assert not (hf[platform_mask] > 0).any(), \
        "pillar cells found inside the spawn platform"


def test_rasterize_does_not_lower_existing_terrain():
    """Running on a non-zero base must never DECREASE any cell."""
    hf = np.full((80, 80), 10, dtype=np.int16)  # flat terrain at 10 units
    rng = np.random.default_rng(3)
    _rasterize_pillars(
        height_field_raw=hf,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        count_range=(3, 3),
        height_range=(0.25, 1.25),
        side_range=(0.4, 1.0),
        top_noise=0.05,
        platform_size=3.0,
        rng=rng,
    )
    assert (hf >= 10).all(), "pillar rasterization lowered an existing cell"


def test_rasterize_uneven_top_produces_variation():
    """Non-zero top_noise must introduce height variation across pillar cells."""
    hf = _empty_heightfield()
    rng = np.random.default_rng(11)
    _rasterize_pillars(
        height_field_raw=hf,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        count_range=(1, 1),
        height_range=(1.0, 1.0),
        side_range=(1.5, 1.5),
        top_noise=0.05,
        platform_size=3.0,
        rng=rng,
    )
    # pillar cells have heights above 0; they should vary due to noise
    pillar_heights = hf[hf > 0]
    assert pillar_heights.size > 0
    assert pillar_heights.max() - pillar_heights.min() > 1  # > 1 vertical_scale unit


# ---------- end-to-end terrain pipeline test ----------

class _PillarCfg:
    enabled = True
    count_range = (2, 2)
    height_range = (0.5, 0.5)
    side_range = (1.0, 1.0)
    top_noise = 0.0
    platform_size = 3.0


class _TerrainCfg:
    """Minimal stand-in for LeggedRobotCfg.terrain used in the pipeline test."""
    mesh_type = "trimesh"
    horizontal_scale = 0.1
    vertical_scale = 0.005
    border_size = 0.0  # simplifies index math in the test
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
    num_rows = 2
    num_cols = 2
    # custom5 terrain_proportions -- forces exactly one terrain type (pyramid slope)
    terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    slope_treshold = 0.75
    pillars = _PillarCfg()


def test_terrain_pipeline_records_world_pillars_per_subterrain():
    """Running the curriculum path must populate Terrain.pillars
    with exactly count_range[0] entries per subterrain, in world coords."""
    from legged_gym.utils.terrain import Terrain

    terrain = Terrain(_TerrainCfg(), num_robots=1)
    for row in range(_TerrainCfg.num_rows):
        for col in range(_TerrainCfg.num_cols):
            entries = terrain.pillars[row][col]
            assert len(entries) == 2, f"subterrain ({row},{col}) has {len(entries)} pillars"
            for (cx, cy, side, yaw) in entries:
                # world x must fall within [row*env_length, (row+1)*env_length]
                assert row * _TerrainCfg.terrain_length <= cx <= (row + 1) * _TerrainCfg.terrain_length
                assert col * _TerrainCfg.terrain_width <= cy <= (col + 1) * _TerrainCfg.terrain_width
                assert side == pytest.approx(1.0)
                assert 0.0 <= yaw <= np.pi / 2.0
