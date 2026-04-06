# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.jini import JINI_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


@configclass
class JiNiBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = JINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 0.25

        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/jini_motion.npz"
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "JiNi-Left-Link-1",
            "JiNi-Left-Link-2",
            "JiNi-Left-Link-3",
            "JiNi-Left-Link-4",
            "JiNi-Left-Link-5",
            "JiNi-Right-Link-1",
            "JiNi-Right-Link-2",
            "JiNi-Right-Link-3",
            "JiNi-Right-Link-4",
            "JiNi-Right-Link-5",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.events.randomize_com_positions.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[r"^(?!JiNi-Left-Link-5$)(?!JiNi-Right-Link-5$).+$"],
        )

        self.terminations.ee_body_pos.params["body_names"] = [
            "JiNi-Left-Link-5",
            "JiNi-Right-Link-5",
        ]

        self.episode_length_s = 30.0
