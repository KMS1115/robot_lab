# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.agents.cusrl_ppo_cfg import (
    UnitreeGo2WFlatTrainerCfg,
    UnitreeGo2WRoughTrainerCfg,
)


@dataclass
class MonsterRoughTrainerCfg(UnitreeGo2WRoughTrainerCfg):
    experiment_name = "monster_rough"


@dataclass
class MonsterFlatTrainerCfg(UnitreeGo2WFlatTrainerCfg):
    experiment_name = "monster_flat"
