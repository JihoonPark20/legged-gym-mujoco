# SPDX-License-Identifier: BSD-3-Clause
"""Terrain atlas, levels/types and spawn origins following legged_gym's layout.

MuJoCo uses the same atlas in every batch simulation. Only the assigned origin
changes on reset, so terrain curriculum needs no model recompilation or copies.
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import terrain_utils


class Terrain:
    terrain_names = ("slope", "rough", "stairs_up", "stairs_down", "obstacles", "stepping_stones")

    def __init__(self, cfg, num_robots, seed=42):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type not in ("plane", "heightfield"):
            raise ValueError("terrain.mesh_type must be plane or heightfield")
        self.rng = np.random.default_rng(seed)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.height_field_raw = None
        if self.type == "plane":
            return
        self._validate()
        self.length_pixels = round(cfg.terrain_length / cfg.horizontal_scale)
        self.width_pixels = round(cfg.terrain_width / cfg.horizontal_scale)
        self.border = round(cfg.border_size / cfg.horizontal_scale)
        self.height_field_raw = np.zeros(
            (
                cfg.num_rows * self.length_pixels + 2 * self.border + 1,
                cfg.num_cols * self.width_pixels + 2 * self.border + 1,
            ),
            dtype=np.int16,
        )
        self.terrain_types = np.empty((cfg.num_rows, cfg.num_cols), dtype=object)
        proportions = np.asarray(cfg.terrain_proportions, dtype=float)
        cumulative = np.cumsum(proportions / proportions.sum())
        for row in range(cfg.num_rows):
            for column in range(cfg.num_cols):
                if cfg.selected:
                    kwargs = dict(cfg.terrain_kwargs)
                    kind = kwargs.pop("type")
                    difficulty = row / max(1, cfg.num_rows - 1)
                else:
                    choice = (column + 0.5) / cfg.num_cols if cfg.curriculum else self.rng.random()
                    kind = self.terrain_names[
                        min(np.searchsorted(cumulative, choice, side="right"), len(cumulative) - 1)
                    ]
                    difficulty = row / max(1, cfg.num_rows - 1) if cfg.curriculum else self.rng.uniform(0.0, 1.0)
                    kwargs = {}
                tile = self.make_terrain(kind, difficulty, column, **kwargs)
                self.add_terrain_to_map(tile, row, column)
                self.terrain_types[row, column] = kind
        self.heightsamples = self.height_field_raw

    def _validate(self):
        c = self.cfg
        if min(c.horizontal_scale, c.vertical_scale, c.terrain_length, c.terrain_width) <= 0:
            raise ValueError("Terrain dimensions and scales must be positive")
        if min(c.num_rows, c.num_cols) < 1 or c.border_size < c.horizontal_scale:
            raise ValueError("Terrain needs positive row/column counts and a border of at least one sample")
        if not 1.0 < c.platform_size < min(c.terrain_length, c.terrain_width):
            raise ValueError("platform_size must exceed the spawn jitter and fit inside the terrain tile")
        for value in (c.terrain_length, c.terrain_width, c.border_size):
            if not np.isclose(value / c.horizontal_scale, round(value / c.horizontal_scale)):
                raise ValueError("Terrain dimensions must be multiples of horizontal_scale")
        p = np.asarray(c.terrain_proportions)
        if p.shape != (len(self.terrain_names),) or np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
            raise ValueError("terrain_proportions needs six nonnegative weights with a positive sum")
        if c.selected and (not isinstance(c.terrain_kwargs, dict) or "type" not in c.terrain_kwargs):
            raise ValueError("Selected terrain requires terrain_kwargs with a type")

    def make_terrain(self, kind, difficulty, column=0, **overrides):
        c = self.cfg
        tile = terrain_utils.SubTerrain(self.length_pixels, self.width_pixels, c.horizontal_scale, c.vertical_scale)
        parameters = {"platform_size": c.platform_size}
        if kind in ("slope", "rough_slope"):
            parameters["slope"] = c.max_slope * difficulty * (-1 if column % 2 else 1)
            parameters.update(overrides)
            terrain_utils.pyramid_sloped_terrain(tile, **parameters)
            if kind == "rough_slope":
                terrain_utils.random_uniform_terrain(
                    tile,
                    self.rng,
                    -c.max_roughness * difficulty,
                    c.max_roughness * difficulty,
                    platform_size=c.platform_size,
                )
        elif kind == "rough":
            parameters.update(min_height=-c.max_roughness * difficulty, max_height=c.max_roughness * difficulty)
            parameters.update(overrides)
            terrain_utils.random_uniform_terrain(tile, self.rng, **parameters)
        elif kind in ("stairs_up", "stairs_down"):
            parameters.update(
                step_width=c.step_width,
                step_height=(c.min_step_height + (c.max_step_height - c.min_step_height) * difficulty)
                * (-1 if kind == "stairs_down" else 1),
            )
            parameters.update(overrides)
            terrain_utils.pyramid_stairs_terrain(tile, **parameters)
        elif kind == "obstacles":
            parameters.update(max_height=c.max_obstacle_height * difficulty, num_rectangles=c.num_obstacles)
            parameters.update(overrides)
            terrain_utils.discrete_obstacles_terrain(tile, self.rng, **parameters)
        elif kind == "stepping_stones":
            parameters.update(
                stone_size=np.interp(difficulty, [0, 1], c.stone_size[::-1]),
                stone_distance=c.stone_distance,
                pit_depth=c.pit_depth,
            )
            parameters.update(overrides)
            terrain_utils.stepping_stones_terrain(tile, self.rng, **parameters)
        elif kind != "flat":
            raise ValueError(f"Unknown terrain type: {kind}")
        elif overrides:
            raise ValueError("Flat terrain accepts no extra parameters")
        return tile

    def add_terrain_to_map(self, tile, row, column):
        x = self.border + row * self.length_pixels
        y = self.border + column * self.width_pixels
        self.height_field_raw[x : x + self.length_pixels + 1, y : y + self.width_pixels + 1] = tile.height_field_raw
        center = tile.height_field_raw[self.length_pixels // 2, self.width_pixels // 2] * self.cfg.vertical_scale
        self.env_origins[row, column] = [
            (row + 0.5) * self.cfg.terrain_length,
            (column + 0.5) * self.cfg.terrain_width,
            center,
        ]

    def add_to_mjcf(self, root):
        world = root.find("worldbody")
        common = dict(name="floor", friction="1 0.005 0.0001", rgba="0.25 0.3 0.35 1", condim="3")
        if self.type == "plane":
            ET.SubElement(world, "geom", type="plane", size="0 0 0.1", **common)
            return
        heights = self.height_field_raw * self.cfg.vertical_scale
        self.min_height = float(heights.min())
        self.height_span = max(float(np.ptp(heights)), self.cfg.vertical_scale)
        nx, ny = heights.shape
        radius_x = (nx - 1) * self.cfg.horizontal_scale / 2
        radius_y = (ny - 1) * self.cfg.horizontal_scale / 2
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "hfield",
            name="terrain",
            nrow=str(ny),
            ncol=str(nx),
            size=f"{radius_x} {radius_y} {self.height_span} 0.5",
        )
        ET.SubElement(
            world,
            "geom",
            type="hfield",
            hfield="terrain",
            pos=f"{radius_x - self.cfg.border_size} {radius_y - self.cfg.border_size} {self.min_height}",
            **common,
        )

    def populate_model(self, model):
        if self.type == "heightfield":
            data = (self.height_field_raw * self.cfg.vertical_scale - self.min_height) / self.height_span
            model.hfield_data[:] = data.T.ravel()  # MuJoCo row=y, column=x, bottom-to-top.

    def heights_at(self, points):
        """Exact triangular interpolation of MuJoCo's hfield surface, not bilinear."""
        points = np.asarray(points)
        if self.type == "plane":
            return np.zeros(points.shape[:-1])
        coordinates = (points[..., :2] + self.cfg.border_size) / self.cfg.horizontal_scale
        maximum = np.array(self.height_field_raw.shape) - 1
        coordinates = np.clip(coordinates, 0, maximum)
        index = np.minimum(np.floor(coordinates).astype(int), maximum - 1)
        u, v = np.moveaxis(coordinates - index, -1, 0)
        i, j = np.moveaxis(index, -1, 0)
        h00 = self.height_field_raw[i, j].astype(float)
        h10 = self.height_field_raw[i + 1, j].astype(float)
        h01 = self.height_field_raw[i, j + 1].astype(float)
        h11 = self.height_field_raw[i + 1, j + 1].astype(float)
        lower = h00 + u * (h10 - h00) + v * (h11 - h10)
        upper = h00 + v * (h01 - h00) + u * (h11 - h01)
        return np.where(u >= v, lower, upper) * self.cfg.vertical_scale

    def contains(self, points):
        points = np.asarray(points)
        if self.type == "plane":
            return np.ones(points.shape[:-1], dtype=bool)
        low = -self.cfg.border_size + self.cfg.horizontal_scale
        high = (
            np.array([self.cfg.num_rows * self.cfg.terrain_length, self.cfg.num_cols * self.cfg.terrain_width])
            + self.cfg.border_size
            - self.cfg.horizontal_scale
        )
        return np.all((points[..., :2] >= low) & (points[..., :2] <= high), axis=-1)
