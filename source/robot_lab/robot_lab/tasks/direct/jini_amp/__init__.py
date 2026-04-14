# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-JiNi-AMP-Moonwalk-Direct-v0",
    entry_point=f"{__name__}.jini_amp_env:JiNiAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jini_amp_env_cfg:JiNiAmpMoonwalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_moonwalk_amp_cfg.yaml",
    },
)

gym.register(
    id="RobotLab-Isaac-JiNi-AMP-StandWithOneFoot-Direct-v0",
    entry_point=f"{__name__}.jini_amp_env:JiNiAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jini_amp_env_cfg:JiNiAmpStandWithOneFootEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_stand_with_one_foot_amp_cfg.yaml",
    },
)
