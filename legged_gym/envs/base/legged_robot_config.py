"""Go2/G1 locomotion settings from Unitree RL Mjlab @ 1425b15f."""

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    seed = 42

    class env:
        num_envs = 4096
        episode_length_s = 20.0
        num_actions = 0

    class sim:
        dt = 0.005
        nthread = 10
        persistent = False
        iterations = 10
        cone = "pyramidal"
        ls_iterations = 20
        tolerance = 1e-08
        ls_tolerance = 0.01
        ccd_iterations = 50

    class control:
        decimation = 4
        action_scale = 0.25
        clip_actions = None
        frictionloss = 0.0
        stiffness = {}
        damping = {}
        effort_limits = {}
        armature = {}

    class commands:
        ranges = [[-1, 2], [-1, 1], [-1, 1]]
        limit_ranges = [[-1, 2], [-1, 1], [-1, 1]]
        initial_stage_ranges = [[-0.5, 1], [-0.5, 0.5], [-1, 1]]
        expansion_step = 120000
        resampling_time_range = [3, 8]
        standing_fraction = 0.05
        heading_command = True
        heading_fraction = 1.0
        heading_range = [-3.141592653589793, 3.141592653589793]
        heading_stiffness = 0.5
        curriculum = True

    class noise:
        add_noise = True
        ang_vel = 0.2
        gravity = 0.05
        joint_pos = 0.01
        joint_vel = 1.5

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.6]
        randomize_com = True
        com_range = [-0.05, 0.05]
        randomize_encoder_bias = True
        encoder_bias_range = [-0.015, 0.015]
        push_robots = True
        push_interval_s = [5, 6]
        push_velocity_ranges = [[-0.5, 0.5], [-0.5, 0.5], [-0.4, 0.4], [-0.52, 0.52], [-0.52, 0.52], [-0.78, 0.78]]

    class rewards:
        soft_dof_pos_limit = 0.9
        contact_history_length = 4
        gait_period = 0.6
        command_threshold = 0.1
        feet_clearance_target = 0.1
        scales = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 1.0,
            "orientation": -1.0,
            "pose": 1.0,
            "body_ang_vel": -0.05,
            "angular_momentum": -0.025,
            "is_terminated": -200.0,
            "dof_acc": -2.5e-07,
            "dof_pos_limits": -10.0,
            "action_rate": -0.05,
            "foot_gait": 0.5,
            "foot_clearance": -1.0,
            "foot_slip": -0.25,
            "soft_landing": -0.001,
            "stand_still": -1.0,
        }
        gait_offsets = [0, 0.5]
        gait_stance_fraction = 0.56

    class asset:
        file = ""
        root_body = ""
        mass_body = ""
        joint_names = []
        feet = []
        foot_sites = []

    class init_state:
        pos = [0, 0, 0.8]
        default_joint_angles = {}

    class termination:
        max_tilt = 1.2217304763960306
        illegal_contact_threshold = None


class LeggedRobotCfgPPO(BaseConfig):
    seed = 42

    # Agent definitions select external implementations; construction is shared
    # by policy family, rather than implemented in a module per algorithm.
    agents = {
        "ppo": {
            "kind": "on_policy",
            "runner_class": "rsl_rl.runners:OnPolicyRunner",
            "algorithm_class": "rsl_rl.algorithms:PPO",
            "actor_class": "rsl_rl.models:MLPModel",
            "critic_class": "rsl_rl.models:MLPModel",
            "parameter_group": "algorithm",
            "construction": "upstream",
            "distribution": "gaussian",
            "learning_rate_fields": ["learning_rate"],
            "install_hint": "python -m pip install -e '.[ppo]'",
        },
        "sac": {
            "kind": "off_policy",
            "runner_class": "rsl_rl.runners:OffPolicyRunner",
            "algorithm_class": "rsl_rl.algorithms:SAC",
            "actor_class": "rsl_rl.models:SACActorModel",
            "critic_class": "rsl_rl.models:SACCriticModel",
            "storage_class": "rsl_rl.storage:ReplayBuffer",
            "parameter_group": "sac",
            "construction": "actor_critic_replay",
            "distribution": "squashed_gaussian",
            "action_transform": "joint_limits",
            "learning_rate_fields": ["actor_learning_rate", "critic_learning_rate", "alpha_learning_rate"],
            "refill_replay_on_resume": True,
            "checkpoint_counters": ["update_step"],
            "install_hint": "python -m pip install -e '.[sac]' (in a separate environment)",
        },
    }

    class evaluation:
        num_envs = 16
        seeds = [0, 1, 2]
        warmup_s = 2.0
        duration_s = 5.0
        settling_hold_s = 0.5
        velocity_tolerance = [0.1, 0.1, 0.15]  # X/Y m/s, yaw rad/s; all axes must settle.
        # Scenario: (warmup command, measured command), both in X/Y/yaw order.
        commands = {
            "stand": ([0, 0, 0], [0, 0, 0]),
            "forward": ([0, 0, 0], [0.5, 0, 0]),
            "backward": ([0, 0, 0], [-0.5, 0, 0]),
            "left": ([0, 0, 0], [0, 0.3, 0]),
            "right": ([0, 0, 0], [0, -0.3, 0]),
            "turn_left": ([0, 0, 0], [0, 0, 0.5]),
            "turn_right": ([0, 0, 0], [0, 0, -0.5]),
            "combined": ([0, 0, 0], [0.5, 0.3, 0.5]),
            "stop": ([0.5, 0, 0], [0, 0, 0]),
        }

    class viewer:
        enabled = True
        history_points = 500
        host = "127.0.0.1"
        max_fps = 15
        port = 8080

    class runner:
        algorithm = "ppo"
        experiment_name = "locomotion"
        max_iterations = 10001
        num_steps_per_env = 24
        run_name = ""
        save_interval = 100

    class policy:
        activation = "elu"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 1.0
        obs_normalization = True

    class algorithm:
        clip_param = 0.2
        desired_kl = 0.01
        entropy_coef = 0.01
        gamma = 0.99
        lam = 0.95
        learning_rate = 0.001
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = "adaptive"
        use_clipped_value_loss = True
        value_loss_coef = 1.0

    class sac:
        # Separate from PPO so unsupported PPO parameters never reach SAC.
        replay_buffer_size = 262144  # Total transitions across all environments.
        mini_batch_size = 256
        num_learning_epochs = 1
        num_mini_batches = 1
        actor_learning_rate = 0.0003
        critic_learning_rate = 0.0003
        alpha_learning_rate = 0.0003
        auto_alpha = True
        alpha = 0.05
        tau = 0.005
        gamma = 0.99
        target_entropy_scale = 1.0
        max_grad_norm = 1.0
        policy_frequency = 2
        n_steps = 1

    class off_policy:
        num_steps_per_env = 1
        start_training = 100  # Collection iterations before updates; also after resume.
        log_interval = 20
