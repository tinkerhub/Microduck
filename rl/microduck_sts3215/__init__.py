"""Isaac Lab tasks for the STS3215 Microduck (Newton / MuJoCo-Warp physics)."""

import gymnasium as gym

gym.register(
    id="Microduck-STS3215-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Sts3215VelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:Sts3215PPORunnerCfg",
    },
)

gym.register(
    id="Microduck-STS3215-Velocity-Flat-Turn",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Sts3215TurnVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:Sts3215TurnPPORunnerCfg",
    },
)
