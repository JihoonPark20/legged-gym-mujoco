"""Seeded legged_gym-style terrain tiles on one shared native MuJoCo heightfield."""

import xml.etree.ElementTree as ET

import numpy as np


class Terrain:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.mesh_type not in ("plane", "heightfield"):
            raise ValueError("terrain.mesh_type must be plane or heightfield")
        self.rough = cfg.mesh_type == "heightfield"
        self.scan_points = np.empty((0, 2))
        if not self.rough:
            if cfg.measure_heights:
                raise ValueError("Height observations require heightfield terrain")
            return
        for name in ("terrain_length", "horizontal_scale", "border_size", "platform_size", "step_width", "height_clip"):
            value = getattr(cfg, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"terrain.{name} must be finite and positive")
        if type(cfg.num_rows) is not int or cfg.num_rows < 1:
            raise ValueError("terrain.num_rows must be a positive integer")
        if type(cfg.max_init_terrain_level) is not int or not 0 <= cfg.max_init_terrain_level < cfg.num_rows:
            raise ValueError("Initial terrain level must be within num_rows")
        supported = {"slope", "slope_down", "stairs", "stairs_down", "noise", "obstacles"}
        if not cfg.types or any(kind not in supported for kind in cfg.types):
            raise ValueError("Unsupported or empty terrain.types")
        for name in ("max_slope", "max_step_height", "max_noise_height", "max_obstacle_height", "height_noise"):
            value = getattr(cfg, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"terrain.{name} must be finite and nonnegative")
        if not 1.8 <= cfg.platform_size < cfg.terrain_length - 1.0:
            raise ValueError("Terrain needs a spawn platform >= 1.8 m and a surrounding traversal area")
        self.cells = round(cfg.terrain_length / cfg.horizontal_scale)
        border = round(cfg.border_size / cfg.horizontal_scale)
        if self.cells < 10 or border < 1 or cfg.step_width < 2 * cfg.horizontal_scale:
            raise ValueError("Terrain resolution is too coarse for the tiles, border or stairs")
        if not np.isclose(self.cells * cfg.horizontal_scale, cfg.terrain_length):
            raise ValueError("terrain_length must be a multiple of horizontal_scale")
        self.spacing = cfg.horizontal_scale
        nx = cfg.num_rows * self.cells + 2 * border + 1
        ny = len(cfg.types) * self.cells + 2 * border + 1
        self.half_size = np.array([(nx - 1), (ny - 1)]) * self.spacing / 2
        self.heights = np.zeros((ny, nx))
        self.origins = np.zeros((cfg.num_rows, len(cfg.types), 3))
        rng = np.random.default_rng(cfg.seed)
        axis = np.linspace(-cfg.terrain_length / 2, cfg.terrain_length / 2, self.cells + 1)
        x, y = np.meshgrid(axis, axis)
        distance = np.maximum(np.abs(x), np.abs(y))
        outside = np.maximum(distance - cfg.platform_size / 2, 0)
        # Flat spawn area and level seams keep neighboring patches traversable.
        edge = np.clip((cfg.terrain_length / 2 - distance) / 0.3, 0, 1)
        for level in range(cfg.num_rows):
            difficulty = (level + 1) / cfg.num_rows
            for column, kind in enumerate(cfg.types):
                if kind.startswith("slope"):
                    values = np.maximum(np.abs(x) - cfg.platform_size / 2, 0) * cfg.max_slope * difficulty
                elif kind.startswith("stairs"):
                    values = np.floor(outside / cfg.step_width) * cfg.max_step_height * difficulty
                elif kind == "noise":
                    values = rng.uniform(-1, 1, x.shape) * cfg.max_noise_height * difficulty
                else:
                    block = max(2, round(0.5 / self.spacing))
                    coarse = rng.uniform(-1, 1, ((self.cells // block) + 1,) * 2)
                    values = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[: x.shape[0], : x.shape[1]]
                    values *= cfg.max_obstacle_height * difficulty
                if kind.endswith("_down"):
                    values = -values
                values *= edge
                values[distance <= cfg.platform_size / 2] = 0
                ix, iy = border + level * self.cells, border + column * self.cells
                self.heights[iy : iy + self.cells + 1, ix : ix + self.cells + 1] = values
                self.origins[level, column, :2] = (np.array([ix, iy]) + self.cells / 2) * self.spacing - self.half_size
        self.z_min = float(self.heights.min())
        self.z_scale = max(float(np.ptp(self.heights)), 0.01)
        self.data = ((self.heights - self.z_min) / self.z_scale).astype(np.float32)
        # Queries use precisely the float32 elevations installed in MuJoCo.
        self.heights = self.data.astype(np.float64) * self.z_scale + self.z_min
        self.origins[:, :, 2] = self.height(self.origins[:, :, :2])
        if cfg.measure_heights:
            axes = [np.asarray(cfg.measured_points_x), np.asarray(cfg.measured_points_y)]
            if any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in axes):
                raise ValueError("Terrain scan axes must be nonempty finite vectors")
            self.scan_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)

    def add_to_xml(self, root):
        attributes = dict(name="floor", friction="1 0.005 0.0001", rgba="0.25 0.3 0.35 1", condim="3")
        if self.rough:
            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            ET.SubElement(
                asset,
                "hfield",
                name="terrain",
                nrow=str(self.data.shape[0]),
                ncol=str(self.data.shape[1]),
                size=f"{self.half_size[0]} {self.half_size[1]} {self.z_scale} 0.5",
            )
            attributes.update(type="hfield", hfield="terrain", pos=f"0 0 {self.z_min}")
        else:
            attributes.update(type="plane", size="0 0 0.1")
        ET.SubElement(root.find("worldbody"), "geom", **attributes)

    def install(self, model):
        if self.rough:
            address = model.hfield("terrain").adr[0]
            model.hfield_data[address : address + self.data.size] = self.data.ravel()

    def height(self, xy):
        """Vectorized height matching MuJoCo's lower-left/upper-right triangle split."""
        xy = np.asarray(xy)
        if not self.rough:
            return np.zeros(xy.shape[:-1])
        grid = (xy + self.half_size) / self.spacing
        grid = np.clip(grid, 0, np.array(self.heights.shape[::-1]) - 1 - 1e-9)
        indices = np.floor(grid).astype(np.int64)
        u, v = np.moveaxis(grid - indices, -1, 0)
        x, y = np.moveaxis(indices, -1, 0)
        a, b = self.heights[y, x], self.heights[y, x + 1]
        c, d = self.heights[y + 1, x], self.heights[y + 1, x + 1]
        return np.where(u >= v, a + u * (b - a) + v * (d - b), a + u * (d - c) + v * (c - a))
