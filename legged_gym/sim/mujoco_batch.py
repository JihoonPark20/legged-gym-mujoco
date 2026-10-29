"""Native CPU equivalent of Unitree's mjlab 1.2 model construction.

Sensors use the same primary/secondary contact
ordering as mjlab, and step() leaves derived state at the pre-integration stage.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch

from legged_gym import LEGGED_GYM_RESOURCES_DIR


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


def foot_geom_names(cfg):
    return getattr(cfg.asset, "foot_geoms", [f"{name}_foot_collision" for name in cfg.asset.foot_sites])


def load_model(cfg):
    path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(path.parent / compiler.get("meshdir", "")))
    ET.SubElement(
        root,
        "option",
        timestep=str(cfg.sim.dt),
        integrator="implicitfast",
        cone=cfg.sim.cone,
        solver="Newton",
        jacobian="auto",
        impratio="1",
        iterations=str(cfg.sim.iterations),
        ls_iterations=str(cfg.sim.ls_iterations),
        tolerance=str(cfg.sim.tolerance),
        ls_tolerance=str(cfg.sim.ls_tolerance),
        ccd_iterations=str(cfg.sim.ccd_iterations),
        gravity="0 0 -9.81",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="1 0.005 0.0001",
        rgba="0.25 0.3 0.35 1",
        condim="3",
    )
    ET.SubElement(world, "light", name="task_light", pos="0 0 4", dir="0 0 -1", directional="true")
    bodies = {body.get("name"): body for body in world.iter("body")}
    bodies[cfg.asset.root_body].set("pos", " ".join(map(str, cfg.init_state.pos)))
    feet = set(foot_geom_names(cfg))
    self_collisions = getattr(cfg.asset, "self_collisions", False)
    collision_geoms = []
    for geom in world.iter("geom"):
        name = geom.get("name", "")
        if name.endswith("_collision"):
            collision_geoms.append(name)
            geom.attrib.update(
                contype="1",
                conaffinity="1" if self_collisions else "0",
                condim="3" if name in feet else "1",
                priority="1" if name in feet else "0",
            )
            if name in feet:
                geom.set("friction", "0.6 0.005 0.0001")
                solimp = getattr(cfg.asset, "foot_solimp", [0.9, 0.95, 0.023, 0.5, 2])
                if solimp is not None:
                    geom.set("solimp", " ".join(map(str, solimp)))
    arms = joint_parameters(cfg.asset.joint_names, cfg.control.armature)
    kp = joint_parameters(cfg.asset.joint_names, cfg.control.stiffness)
    kd = joint_parameters(cfg.asset.joint_names, cfg.control.damping)
    limits = joint_parameters(cfg.asset.joint_names, cfg.control.effort_limits)
    actuators = ET.SubElement(root, "actuator")
    for name, arm, p, d, limit in zip(cfg.asset.joint_names, arms, kp, kd, limits):
        joint = world.find(f".//joint[@name='{name}']")
        joint.attrib.update(armature=str(arm), frictionloss=str(cfg.control.frictionloss))
        ET.SubElement(
            actuators,
            "position",
            name=name,
            joint=name,
            kp=str(p),
            kv=str(d),
            forcelimited="true",
            forcerange=f"{-limit} {limit}",
            ctrllimited="false",
        )
    sensors = root.find("sensor")
    ET.SubElement(sensors, "framelinvel", name="root_link_velocity", objtype="xbody", objname=cfg.asset.root_body)
    ET.SubElement(
        sensors, "frameangvel", name="root_link_angular_velocity", objtype="xbody", objname=cfg.asset.root_body
    )
    reward_body = getattr(cfg.asset, "reward_body", cfg.asset.root_body)
    if reward_body != cfg.asset.root_body:
        ET.SubElement(sensors, "frameangvel", name="reward_body_angular_velocity", objtype="xbody", objname=reward_body)
    for site in cfg.asset.foot_sites:
        ET.SubElement(sensors, "framelinvel", name=f"site_velocity_{site}", objtype="site", objname=site)
    foot_bodies = getattr(cfg.asset, "foot_contact_bodies", None)
    for name in collision_geoms if foot_bodies is None else ():
        ET.SubElement(
            sensors,
            "contact",
            name=f"geom_contact_{name}",
            geom1=name,
            geom2="floor",
            data="found force",
            reduce="netforce" if name in feet else "none",
            num="1",
        )
    if foot_bodies is not None:
        for body in foot_bodies:
            ET.SubElement(
                sensors,
                "contact",
                name=f"foot_contact_{body}",
                subtree1=body,
                geom2="floor",
                data="found force",
                reduce="netforce",
                num="1",
            )
    if self_collisions:
        ET.SubElement(
            sensors,
            "contact",
            name="self_collision",
            subtree1=cfg.asset.root_body,
            subtree2=cfg.asset.root_body,
            data="found force",
            reduce="none",
            num="1",
        )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


class MujocoBatch:
    def __init__(self, cfg, rng):
        path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
        source = ET.parse(path).getroot()
        digest = hashlib.sha256(path.read_bytes())
        meshdir = path.parent / source.find("compiler").get("meshdir", "")
        for mesh in source.findall("asset/mesh"):
            digest.update((meshdir / mesh.get("file")).read_bytes())
        self.asset_digest = digest.hexdigest()
        self.model = load_model(cfg)
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
        self._randomize_model(cfg, rng)
        self.batch.reset()
        self.qacc = self.batch.bind("qacc")
        self.xquat = self.batch.bind("xquat")
        self.site_xpos = self.batch.bind("site_xpos")
        self.root_id = self.model.body(cfg.asset.root_body).id
        self.reward_body_id = self.model.body(getattr(cfg.asset, "reward_body", cfg.asset.root_body)).id
        self.site_ids = np.array([self.model.site(name).id for name in cfg.asset.foot_sites])
        self.foot_geoms = foot_geom_names(cfg)
        self.nonfoot_geoms = [
            self.model.geom(i).name
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.endswith("_collision") and self.model.geom(i).name not in self.foot_geoms
        ]
        foot_bodies = getattr(cfg.asset, "foot_contact_bodies", None)
        self.foot_columns = self._columns(
            [f"foot_contact_{body}" for body in foot_bodies]
            if foot_bodies is not None
            else [f"geom_contact_{name}" for name in self.foot_geoms]
        )
        self.history_columns = self._columns(
            ["self_collision"]
            if getattr(cfg.asset, "self_collisions", False)
            else [f"geom_contact_{name}" for name in self.nonfoot_geoms]
        )
        self.history_contact_count = len(self.history_columns)
        self.site_velocity_columns = self._columns([f"site_velocity_{site}" for site in cfg.asset.foot_sites])

    def _columns(self, names):
        return np.array(
            [
                np.arange(
                    self.model.sensor(name).adr[0], self.model.sensor(name).adr[0] + self.model.sensor(name).dim[0]
                )
                for name in names
            ]
        )

    def sensor(self, name):
        sensor = self.model.sensor(name)
        return self.sensordata[:, sensor.adr[0] : sensor.adr[0] + sensor.dim[0]]

    def _randomize_model(self, cfg, rng):
        if cfg.domain_rand.randomize_friction:
            friction = self.batch.expand("geom_friction")
            ids = [self.model.geom(name).id for name in foot_geom_names(cfg)]
            friction[:, ids, 0] = rng.uniform(*cfg.domain_rand.friction_range, (cfg.env.num_envs, 1))
        self.encoder_bias = np.zeros((cfg.env.num_envs, cfg.env.num_actions))
        if cfg.domain_rand.randomize_encoder_bias:
            self.encoder_bias[:] = rng.uniform(*cfg.domain_rand.encoder_bias_range, self.encoder_bias.shape)
        if cfg.domain_rand.randomize_com:
            com = self.batch.expand("body_ipos")
            com[:, self.model.body(cfg.asset.mass_body).id] += rng.uniform(
                *cfg.domain_rand.com_range, (cfg.env.num_envs, 3)
            )
            self.batch.set_const()

    @property
    def foot_contacts(self):
        return self.sensordata[:, self.foot_columns]

    @property
    def history_contacts(self):
        return self.sensordata[:, self.history_columns]

    def close(self):
        self.qacc = self.xquat = self.site_xpos = None
        self.qpos = self.qvel = self.ctrl = self.xpos = self.sensordata = None
        self.batch = None

    def step(self, targets):
        self.ctrl[:] = targets
        self.batch.step(stage=mujoco.mjtStage.mjSTAGE_NONE)
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
