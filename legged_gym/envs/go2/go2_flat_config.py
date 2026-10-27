"""Unitree Go2 flat-ground velocity tracking (12 actuated joints)."""

from legged_gym.envs.base.mujoco_config import MujocoPPOCfg, MujocoRobotCfg


class Go2FlatCfg(MujocoRobotCfg):
    class env(MujocoRobotCfg.env):
        num_actions = 12

    class asset(MujocoRobotCfg.asset):
        file = "go2/go2.xml"
        root_body = "base_link"
        mass_body = "base_link"
        joint_names = [f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL") for joint in ("hip", "thigh", "calf")]
        feet = [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]
        penalize_contacts_on = [r".*_hip", r".*_thigh", r".*_calf"]

    class init_state(MujocoRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]
        default_joint_angles = {
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }

    class control(MujocoRobotCfg.control):
        torque_speed_curve = True  # Unitree Go2HV: X1=13.5, X2=30, Y1=20.2, Y2=23.4.

    class observations(MujocoRobotCfg.observations):
        critic_joint_effort = True

    class termination(MujocoRobotCfg.termination):
        min_height = 0.0
        base_contact = True

    class rewards(MujocoRobotCfg.rewards):
        scales = {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "dof_vel": -0.001,
            "dof_acc": -2.5e-7,
            "torques": -2e-4,
            "action_rate": -0.1,
            "dof_pos_limits": -10.0,
            "energy": -2e-5,
            "orientation": -2.5,
            "joint_position": -0.7,
            "feet_air_time": 0.1,
            "air_time_variance": -1.0,
            "feet_slide": -0.1,
            "collision": -1.0,
        }


class Go2FlatCfgPPO(MujocoPPOCfg):
    class runner(MujocoPPOCfg.runner):
        experiment_name = "go2_flat"
