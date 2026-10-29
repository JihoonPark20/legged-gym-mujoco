"""Unitree RL Mjlab Go2 Flat @ 1425b15f, mjlab 1.2.0 reference settings."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 12

    class asset(LeggedRobotCfg.asset):
        file = "go2/go2_mjlab.xml"
        root_body = mass_body = "base_link"
        joint_names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
        feet = [f"{leg}_calf" for leg in ("FL", "FR", "RL", "RR")]
        foot_sites = ["FL", "FR", "RL", "RR"]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0, 0, 0.32]
        default_joint_angles = {
            ".*R_hip_joint": 0.1,
            ".*L_hip_joint": -0.1,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {".*": 20, ".*_calf_joint": 40}
        damping = {".*": 1, ".*_calf_joint": 2}
        effort_limits = {".*": 23.5, ".*_calf_joint": 45}
        armature = {".*": 0.01, ".*_calf_joint": 0.02}

    class termination(LeggedRobotCfg.termination):
        illegal_contact_threshold = 10

    class rewards(LeggedRobotCfg.rewards):
        gait_offsets = [0, 0.5, 0.5, 0]
        pose_std_standing = {".*_hip_joint": 0.05, ".*_thigh_joint": 0.1, ".*_calf_joint": 0.15}
        pose_std_walking = {".*_hip_joint": 0.15, ".*_thigh_joint": 0.35, ".*_calf_joint": 0.5}
        pose_std_running = {".*_hip_joint": 0.15, ".*_thigh_joint": 0.35, ".*_calf_joint": 0.5}


class Go2CfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "go2"
