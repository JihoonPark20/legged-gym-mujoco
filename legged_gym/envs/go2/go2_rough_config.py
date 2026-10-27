"""Go2 terrain locomotion with legged_gym's height-scan actor/critic inputs."""

from .go2_flat_config import Go2FlatCfg, Go2FlatCfgPPO


class Go2RoughCfg(Go2FlatCfg):
    class terrain(Go2FlatCfg.terrain):
        mesh_type = "heightfield"
        curriculum = True
        measure_heights = True

    class observations(Go2FlatCfg.observations):
        actor_height_scan = True
        critic_height_scan = True


class Go2RoughCfgPPO(Go2FlatCfgPPO):
    class runner(Go2FlatCfgPPO.runner):
        experiment_name = "go2_rough"
