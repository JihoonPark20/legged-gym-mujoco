"""G1 terrain locomotion; conservative initial terrain levels for a humanoid."""

from .g1_config import G1FlatCfg, G1FlatCfgPPO


class G1RoughCfg(G1FlatCfg):
    class terrain(G1FlatCfg.terrain):
        mesh_type = "heightfield"
        curriculum = True
        measure_heights = True
        max_init_terrain_level = 0
        max_slope = 0.25
        max_step_height = 0.12
        max_obstacle_height = 0.12

    class observations(G1FlatCfg.observations):
        actor_height_scan = True
        critic_height_scan = True


class G1RoughCfgPPO(G1FlatCfgPPO):
    class runner(G1FlatCfgPPO.runner):
        experiment_name = "g1_29dof_rough"
