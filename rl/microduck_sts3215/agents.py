"""PPO runner for the STS3215 Microduck: Isaac Lab's MicroDuck recipe with left/right symmetry augmentation."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckSymmetricPPORunnerCfg


@configclass
class Sts3215PPORunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "microduck_sts3215_flat"
        self.run_name = "bam"
        self.max_iterations = 8000
        self.save_interval = 250


@configclass
class Sts3215TurnPPORunnerCfg(Sts3215PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "microduck_sts3215_flat_turn"  # keeps its logs apart from the base task's
