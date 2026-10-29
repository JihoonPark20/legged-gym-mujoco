"""Unitree RL Mjlab G1 29-DoF Flat @ 1425b15f / mjlab 1.2.0.

Motor specifications and posture tolerances follow the pinned Apache-2.0 source.
See resources/robots/g1/unitree_mjlab.json and licenses/unitree_rl_mjlab_LICENSE.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


def _reflected_inertia(rotors, gears):
    return rotors[0] * (gears[1] * gears[2]) ** 2 + rotors[1] * gears[2] ** 2 + rotors[2]


_ARM_5020 = _reflected_inertia((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
_ARM_7520_14 = _reflected_inertia((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
_ARM_7520_22 = _reflected_inertia((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARM_4010 = _reflected_inertia((0.068e-4, 0, 0), (1, 5, 5))
_FREQ = 10 * 2.0 * 3.1415926535
# Pattern, reflected inertia, effort limit. Ankle/waist linkages use two 5020 motors.
_MOTORS = [
    (r".*_(elbow|shoulder_pitch|shoulder_roll|shoulder_yaw|wrist_roll)_joint", _ARM_5020, 25.0),
    (r"(.*_hip_pitch|.*_hip_yaw|waist_yaw)_joint", _ARM_7520_14, 88.0),
    (r".*_(hip_roll|knee)_joint", _ARM_7520_22, 139.0),
    (r".*_wrist_(pitch|yaw)_joint", _ARM_4010, 5.0),
    (r"(waist_pitch|waist_roll|.*_ankle_pitch|.*_ankle_roll)_joint", _ARM_5020 * 2, 50.0),
]


class G1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 29

    class asset(LeggedRobotCfg.asset):
        file = "g1/g1_mjlab.xml"
        root_body = "pelvis"
        mass_body = reward_body = "torso_link"
        joint_names = (
            [
                f"{side}_{joint}_joint"
                for side in ("left", "right")
                for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
            ]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [
                f"{side}_{joint}_joint"
                for side in ("left", "right")
                for joint in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
        )
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_contact_bodies = feet
        foot_sites = ["left_foot", "right_foot"]
        foot_geoms = [f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)]
        self_collisions = True
        foot_solimp = None  # Keep the source default (0.9, 0.95, 0.001, 0.5, 2).

    class init_state(LeggedRobotCfg.init_state):
        pos = [0, 0, 0.8]
        default_joint_angles = {
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {pattern: arm * _FREQ**2 for pattern, arm, effort in _MOTORS}
        damping = {pattern: 4 * arm * _FREQ for pattern, arm, effort in _MOTORS}
        armature = {pattern: arm for pattern, arm, effort in _MOTORS}
        effort_limits = {pattern: effort for pattern, arm, effort in _MOTORS}
        action_scale = {pattern: 0.25 * effort / (arm * _FREQ**2) for pattern, arm, effort in _MOTORS}

    class rewards(LeggedRobotCfg.rewards):
        pose_std_standing = {".*": 0.05}
        pose_std_walking = {
            ".*hip_pitch.*": 0.5,
            ".*hip_roll.*": 0.15,
            ".*hip_yaw.*": 0.15,
            ".*knee.*": 0.5,
            ".*ankle_pitch.*": 0.15,
            ".*ankle_roll.*": 0.1,
            ".*waist_yaw.*": 0.15,
            ".*waist_roll.*": 0.1,
            ".*waist_pitch.*": 0.1,
            ".*shoulder_pitch.*": 0.15,
            ".*shoulder_roll.*": 0.1,
            ".*shoulder_yaw.*": 0.1,
            ".*elbow.*": 0.1,
            ".*wrist.*": 0.1,
        }
        pose_std_running = {
            ".*hip_pitch.*": 0.5,
            ".*hip_roll.*": 0.25,
            ".*hip_yaw.*": 0.25,
            ".*knee.*": 0.5,
            ".*ankle_pitch.*": 0.25,
            ".*ankle_roll.*": 0.1,
            ".*waist_yaw.*": 0.25,
            ".*waist_roll.*": 0.1,
            ".*waist_pitch.*": 0.1,
            ".*shoulder_pitch.*": 0.25,
            ".*shoulder_roll.*": 0.1,
            ".*shoulder_yaw.*": 0.1,
            ".*elbow.*": 0.1,
            ".*wrist.*": 0.1,
        }
        self_collision_threshold = 10.0
        scales = {**LeggedRobotCfg.rewards.scales, "self_collisions": -1.0}


class G1CfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "g1"
