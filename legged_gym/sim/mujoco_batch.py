"""Official MuJoCo 3.17 batch API; no per-environment Python stepping."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch

from legged_gym import LEGGED_GYM_RESOURCES_DIR
from legged_gym.utils.terrain import Terrain


def joint_parameters(names, patterns, default=None):
    """Resolve regex settings in insertion order; more specific entries come last."""
    values = np.full(len(names), np.nan if default is None else default, dtype=np.float64)
    for pattern, value in patterns.items():
        mask = [re.fullmatch(pattern, name) is not None for name in names]
        if not any(mask):
            raise ValueError(f"Joint pattern matches no joints: {pattern}")
        values[mask] = value
    if not np.isfinite(values).all():
        raise ValueError("Every joint must have a finite configured value")
    return values


def load_model(cfg, terrain=None):
    """Add task terrain and native fixed-size sensors to the official Unitree MJCF."""
    terrain = Terrain(cfg.terrain, cfg.env.num_envs, cfg.seed) if terrain is None else terrain
    path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(path.parent / compiler.get("meshdir", "")))
    compiler.set("fusestatic", "false")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep=str(cfg.sim.dt), integrator="implicitfast", iterations=str(cfg.sim.iterations), cone="elliptic"
    )
    # Remove the vendor's DDS/noisy sensors and poses; the task owns policy I/O.
    for tag in ("sensor", "actuator", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    world = root.find("worldbody")
    terrain.add_to_mjcf(root)
    ET.SubElement(world, "light", name="task_light", pos="0 0 4", dir="0 0 -1", directional="true")
    bodies = {b.get("name"): b for b in world.iter("body")}
    bodies[cfg.asset.root_body].set("pos", " ".join(map(str, cfg.init_state.pos)))
    # Vendor Go2 foot collision geoms live on the calves. Move them to their
    # welded foot frames so calf collision penalties do not punish foot contact.
    for foot in cfg.asset.feet:
        if foot.endswith("_foot"):
            leg = foot[:-5]
            calf = bodies[f"{leg}_calf"]
            geom = calf.find(f"geom[@name='{leg}']")
            if geom is not None:
                calf.remove(geom)
                geom.set("pos", "-0.002 0 0")
                bodies[foot].append(geom)
    for joint in world.iter("joint"):
        joint.set("damping", "0")  # Damping is applied by task PD, exactly once.
        joint.set("armature", str(cfg.control.armature))
        joint.set("frictionloss", str(cfg.control.frictionloss))
        # Vendor G1 joint force ranges differ from the training actuator limits.
        # The task's motor ctrlrange is the single source of torque saturation.
        joint.set("actuatorfrclimited", "false")
    limits = joint_parameters(cfg.asset.joint_names, cfg.control.effort_limits)
    actuators = ET.SubElement(root, "actuator")
    for name, limit in zip(cfg.asset.joint_names, limits):
        ET.SubElement(actuators, "motor", name=name, joint=name, ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    sensors = ET.SubElement(root, "sensor")
    for name in bodies:
        ET.SubElement(sensors, "contact", name=f"contact_{name}", body2=name, data="force", reduce="netforce")
    for name in cfg.asset.feet:
        ET.SubElement(sensors, "framelinvel", name=f"velocity_{name}", objtype="body", objname=name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    terrain.populate_model(model)
    if model.nq != 7 + cfg.env.num_actions or model.nv != 6 + cfg.env.num_actions:
        raise ValueError("Expected one floating base and the configured number of hinge joints")
    return model


class MujocoBatch:
    def __init__(self, cfg, rng):
        self.terrain = Terrain(cfg.terrain, cfg.env.num_envs, cfg.seed)
        self.model = load_model(cfg, self.terrain)
        self.batch = batch.Batch(
            self.model, nsim=cfg.env.num_envs, nthread=cfg.sim.nthread, persistent=cfg.sim.persistent
        )
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.xpos = self.batch.bind("xpos")
        self.sensordata = self.batch.bind("sensordata")
        self.joint_ids = np.array([self.model.joint(n).id for n in cfg.asset.joint_names])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody)]
        self.feet_ids = np.array([self.model.body(n).id for n in cfg.asset.feet])
        self.contact_columns = np.array(
            [
                np.arange(self.model.sensor(f"contact_{n}").adr[0], self.model.sensor(f"contact_{n}").adr[0] + 3)
                for n in self.body_names
            ]
        )
        self.velocity_columns = np.array(
            [
                np.arange(self.model.sensor(f"velocity_{n}").adr[0], self.model.sensor(f"velocity_{n}").adr[0] + 3)
                for n in cfg.asset.feet
            ]
        )
        self._randomize_model(cfg, rng)
        self.batch.reset()

    def _randomize_model(self, cfg, rng):
        if cfg.domain_rand.randomize_friction:
            friction = self.batch.expand("geom_friction")
            friction[:, :, 0] = rng.uniform(*cfg.domain_rand.friction_range, (cfg.env.num_envs, 1))
        if cfg.domain_rand.randomize_base_mass:
            body_id = self.model.body(cfg.asset.mass_body).id
            masses = self.batch.expand("body_mass")
            inertias = self.batch.expand("body_inertia")
            original = self.model.body_mass[body_id]
            randomized = original + rng.uniform(*cfg.domain_rand.added_mass_range, cfg.env.num_envs)
            if np.any(randomized <= 0):
                raise ValueError("Base mass randomization must leave strictly positive masses")
            masses[:, body_id] = randomized
            inertias[:, body_id] *= (randomized / original)[:, None]
            self.batch.set_const()

    def step(self, torque, *, observe=False):
        self.ctrl[:] = torque
        self.batch.step(stage=mujoco.mjtStage.mjSTAGE_VEL if observe else mujoco.mjtStage.mjSTAGE_NONE)
        bad_state_warnings = [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        failed = np.any(self.batch.warning[:, bad_state_warnings, 1] > 0, axis=1)
        if failed.any():
            raise FloatingPointError(
                f"MuJoCo reported invalid dynamics in environments {np.flatnonzero(failed).tolist()}"
            )

    def reset(self, ids):
        self.batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)

    def forward(self, ids=None):
        self.batch.forward(ids)

    @property
    def contact_forces(self):
        return self.sensordata[:, self.contact_columns]

    @property
    def feet_velocities(self):
        return self.sensordata[:, self.velocity_columns]

    def close(self):
        # The public Batch API owns its pool through object lifetime. Release
        # exported views too: their base objects keep the native batch alive.
        self.qpos = self.qvel = self.ctrl = self.xpos = self.sensordata = None
        self.batch = None
