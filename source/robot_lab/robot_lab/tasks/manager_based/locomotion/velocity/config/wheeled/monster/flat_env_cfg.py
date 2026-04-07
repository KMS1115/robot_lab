# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import MonsterRoughEnvCfg


@configclass
class MonsterFlatEnvCfg(MonsterRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Flat-only tuning: bias the robot towards wheel-driven locomotion and suppress stepping.
        self.actions.joint_pos.scale = {".*hip_roll_joint_.*": 0.06, "^(?!.*hip_roll_joint_.*).*": 0.12}

        # Make flat resets easy: near-upright spawn, near-zero initial twist, and no reset-time disturbances.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
        }
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.base_height_l2.params["target_height"] = 0.43
        self.rewards.joint_power.weight = -8.0e-5
        self.rewards.joint_pos_penalty.weight = -3.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 8.0
        self.rewards.action_rate_l2.weight = -0.04

        # Keep all wheels planted on flat ground and penalize stepping behavior.
        self.rewards.undesired_contacts.weight = -4.0
        self.rewards.undesired_contacts.params["threshold"] = 0.5
        self.rewards.feet_air_time.weight = -1.0
        self.rewards.feet_air_time.params["threshold"] = 0.05
        self.rewards.feet_contact.weight = -0.5
        self.rewards.feet_contact.params["expect_contact_num"] = 4
        self.rewards.feet_contact_without_cmd.weight = 0.25
        self.rewards.feet_gait.weight = 0.0
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
                "threshold": 1.0,
            },
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "MonsterFlatEnvCfg":
            self.disable_zero_weight_rewards()
