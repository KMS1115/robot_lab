# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.assets.monster import MONSTER_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRoughEnvCfg,
)


@configclass
class MonsterRoughEnvCfg(UnitreeGo2WRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = "wheel_.*"

    leg_joint_names = [
        "hip_roll_joint_FR",
        "hip_pitch_joint_FR",
        "knee_pitch_joint_FR",
        "hip_roll_joint_FL",
        "hip_pitch_joint_FL",
        "knee_pitch_joint_FL",
        "hip_roll_joint_HR",
        "hip_pitch_joint_HR",
        "knee_pitch_joint_HR",
        "hip_roll_joint_HL",
        "hip_pitch_joint_HL",
        "knee_pitch_joint_HL",
    ]
    wheel_joint_names = [
        "wheel_joint_FR",
        "wheel_joint_FL",
        "wheel_joint_HR",
        "wheel_joint_HL",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MONSTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.actions.joint_pos.scale = {".*hip_roll_joint_.*": 0.125, "^(?!.*hip_roll_joint_.*).*": 0.25}

        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["(hip_roll|hip_pitch|knee_pitch)_joint_FR", "(hip_roll|hip_pitch|knee_pitch)_joint_HL"],
            ["(hip_roll|hip_pitch|knee_pitch)_joint_FL", "(hip_roll|hip_pitch|knee_pitch)_joint_HR"],
        ]
        self.rewards.action_mirror.params["mirror_joints"] = [
            ["(hip_roll|hip_pitch|knee_pitch)_joint_FR", "(hip_roll|hip_pitch|knee_pitch)_joint_HL"],
            ["(hip_roll|hip_pitch|knee_pitch)_joint_FL", "(hip_roll|hip_pitch|knee_pitch)_joint_HR"],
        ]
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("wheel_FL", "wheel_HR"), ("wheel_FR", "wheel_HL"))
