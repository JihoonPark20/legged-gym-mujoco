"""Class-style settings for CPU MuJoCo and device-independent rsl-rl PPO.

Locomotion settings follow unitree_rl_lab @ 4960b84732b0c2ec593dccbfe963fda1bcd7b1e3.
Physics/contact parameters are MuJoCo adaptations, not PhysX equivalents.
"""

from .base_config import BaseConfig


class MujocoRobotCfg(BaseConfig):
    seed = 42

    class env:
        num_envs = 4096
        num_actions = 0
        history_length = 1
        episode_length_s = 20.0

    class sim:
        dt = 0.005
        nthread = 10  # Set to 0 to let MuJoCo choose the CPU count.
        persistent = False  # Memory scales with worker count, not environment count.
        iterations = 50

    class terrain:
        mesh_type = "plane"  # plane or heightfield; flat tasks retain their existing I/O.
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 4.0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 10
        curriculum = False
        max_init_terrain_level = 1
        # slope, rough, stairs up, stairs down, obstacles, stepping stones
        terrain_proportions = [0.15, 0.15, 0.25, 0.25, 0.1, 0.1]
        selected = False
        terrain_kwargs = {"type": "rough"}
        platform_size = 2.0
        max_slope = 0.4
        max_roughness = 0.06
        min_step_height = 0.02
        max_step_height = 0.18
        step_width = 0.3
        max_obstacle_height = 0.2
        num_obstacles = 20
        stone_size = [0.3, 1.0]
        stone_distance = 0.1
        pit_depth = 0.25
        measure_heights = False
        measured_points_x = [round(-0.8 + i * 0.1, 2) for i in range(17)]
        measured_points_y = [round(-0.5 + i * 0.1, 2) for i in range(11)]
        height_observation_offset = 0.5
        height_observation_scale = 5.0

    class asset:
        file = ""
        root_body = ""
        mass_body = ""
        joint_names = []
        feet = []
        penalize_contacts_on = []

    class init_state:
        pos = [0.0, 0.0, 0.4]
        default_joint_angles = {}
        joint_velocity_range = [-1.0, 1.0]

    class control:
        decimation = 4
        action_scale = 0.25
        clip_actions = 100.0
        stiffness = {".*": 25.0}
        damping = {".*": 0.5}
        effort_limits = {".*": 23.4}
        armature = 0.01
        frictionloss = 0.01
        torque_speed_curve = False

    class commands:
        resampling_time = 10.0
        standing_fraction = 0.1
        ranges = [[-0.1, 0.1], [-0.1, 0.1], [-1.0, 1.0]]
        limit_ranges = [[-1.0, 1.0], [-0.4, 0.4], [-1.0, 1.0]]
        curriculum = True
        curriculum_step = 0.1
        curriculum_threshold = 0.8

    class noise:
        add_noise = True
        ang_vel = 0.2
        gravity = 0.05
        joint_pos = 0.01
        joint_vel = 1.5
        height_measurements = 0.1

    class observations:
        critic_joint_effort = False
        actor_height_scan = False
        critic_height_scan = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = [5.0, 10.0]
        max_push_vel_xy = 0.5

    class rewards:
        tracking_sigma = 0.25
        tracking_frame = "body"
        soft_dof_pos_limit = 0.9
        contact_threshold = 1.0
        contact_history_length = 3
        base_height_target = 0.78
        feet_air_time_threshold = 0.5
        stand_still_scale = 5.0
        velocity_threshold = 0.3
        gait_period = 0.8
        gait_offsets = [0.0, 0.5]
        gait_stance_fraction = 0.55
        feet_clearance_target = 0.1
        feet_clearance_std = 0.05
        feet_clearance_tanh_mult = 2.0
        scales = {}

    class termination:
        min_height = 0.2
        max_tilt = 0.8
        base_contact = False


class MujocoPPOCfg(BaseConfig):
    seed = 42

    class runner:
        num_steps_per_env = 24
        max_iterations = 50000
        save_interval = 100
        experiment_name = "locomotion"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
