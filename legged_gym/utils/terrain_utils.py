# SPDX-License-Identifier: BSD-3-Clause
"""NumPy terrain generators with legged_gym-style SubTerrain height samples.

Samples use x-first indexing and integer vertical_scale units. No simulator API
or global random state is involved; Terrain provides a seeded generator.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubTerrain:
    length: int
    width: int
    horizontal_scale: float
    vertical_scale: float
    height_field_raw: np.ndarray = field(init=False)

    def __post_init__(self):
        self.height_field_raw = np.zeros((self.length + 1, self.width + 1), dtype=np.int16)

    def set_heights(self, heights):
        samples = np.rint(np.asarray(heights) / self.vertical_scale)
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > np.iinfo(np.int16).max):
            raise ValueError("Terrain elevations exceed the int16 vertical scale range")
        self.height_field_raw[:] = samples
        return self


def _coordinates(terrain):
    x = np.arange(terrain.length + 1) * terrain.horizontal_scale
    y = np.arange(terrain.width + 1) * terrain.horizontal_scale
    return np.meshgrid(x, y, indexing="ij")


def _platform(terrain, heights, platform_size, height=None):
    x, y = _coordinates(terrain)
    mask = (np.abs(x - x.max() / 2) <= platform_size / 2) & (np.abs(y - y.max() / 2) <= platform_size / 2)
    heights[mask] = heights[terrain.length // 2, terrain.width // 2] if height is None else height
    return heights


def pyramid_sloped_terrain(terrain, slope, platform_size=2.0):
    x, y = _coordinates(terrain)
    distance = np.minimum.reduce((x, x.max() - x, y, y.max() - y))
    limit = (min(x.max(), y.max()) - platform_size) / 2
    return terrain.set_heights(slope * np.minimum(distance, limit))


def random_uniform_terrain(terrain, rng, min_height, max_height, platform_size=2.0, downsampled_scale=0.2):
    nx = max(2, int(terrain.length * terrain.horizontal_scale / downsampled_scale) + 1)
    ny = max(2, int(terrain.width * terrain.horizontal_scale / downsampled_scale) + 1)
    coarse = rng.uniform(min_height, max_height, (nx, ny))
    along_x = np.stack(
        [np.interp(np.linspace(0, 1, terrain.length + 1), np.linspace(0, 1, nx), c) for c in coarse.T], axis=1
    )
    heights = np.stack([np.interp(np.linspace(0, 1, terrain.width + 1), np.linspace(0, 1, ny), r) for r in along_x])
    heights += terrain.height_field_raw * terrain.vertical_scale
    return terrain.set_heights(_platform(terrain, heights, platform_size))


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=2.0):
    if step_width < terrain.horizontal_scale:
        raise ValueError("step_width must be at least horizontal_scale")
    x, y = _coordinates(terrain)
    distance = np.minimum.reduce((x, x.max() - x, y, y.max() - y))
    limit = (min(x.max(), y.max()) - platform_size) / 2
    heights = np.floor(np.minimum(distance, limit) / step_width + 1e-9) * step_height
    return terrain.set_heights(heights)


def discrete_obstacles_terrain(terrain, rng, max_height, num_rectangles=20, platform_size=2.0):
    heights = np.zeros_like(terrain.height_field_raw, dtype=float)
    for _ in range(num_rectangles):
        sx = int(rng.integers(1, max(2, terrain.length // 4)))
        sy = int(rng.integers(1, max(2, terrain.width // 4)))
        x = int(rng.integers(0, terrain.length - sx + 1))
        y = int(rng.integers(0, terrain.width - sy + 1))
        heights[x : x + sx, y : y + sy] = rng.uniform(-max_height, max_height)
    return terrain.set_heights(_platform(terrain, heights, platform_size, 0.0))


def stepping_stones_terrain(terrain, rng, stone_size, stone_distance, pit_depth=0.25, platform_size=2.0):
    stone = max(1, round(stone_size / terrain.horizontal_scale))
    gap = max(1, round(stone_distance / terrain.horizontal_scale))
    heights = np.full_like(terrain.height_field_raw, -pit_depth, dtype=float)
    for x in range(0, terrain.length + 1, stone + gap):
        for y in range(0, terrain.width + 1, stone + gap):
            heights[x : x + stone, y : y + stone] = rng.uniform(-0.01, 0.01)
    return terrain.set_heights(_platform(terrain, heights, platform_size, 0.0))
