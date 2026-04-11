# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from robot_lab.assets.jini import JINI_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

JINI_KEY_BODY_NAMES = [
    "JiNi_Left_Link_1",
    "JiNi_Left_Link_2",
    "JiNi_Left_Link_3",
    "JiNi_Left_Link_4",
    "JiNi_Left_Link_5",
    "JiNi_Right_Link_1",
    "JiNi_Right_Link_2",
    "JiNi_Right_Link_3",
    "JiNi_Right_Link_4",
    "JiNi_Right_Link_5",
]


@configclass
class JiNiAmpMoonwalkEnvCfg(DirectRLEnvCfg):
    """JiNi AMP environment config."""

    # basic reward
    rew_termination = -0.0
    rew_action_l2 = -0.1
    rew_joint_pos_limits = -10.0
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    # imitation reward parameters
    rew_imitation_pos = 1.0
    rew_imitation_rot = 0.5
    rew_imitation_joint_pos = 2.5
    rew_imitation_joint_vel = 1.0
    imitation_sigma_pos = 1.2
    imitation_sigma_rot = 0.5
    imitation_sigma_joint_pos = 1.5
    imitation_sigma_joint_vel = 8.0

    # env
    episode_length_s = 10.0
    decimation = 1
    dt = 1 / 60

    # spaces
    action_space = 10
    observation_space = action_space * 2 + 1 + 6 + len(JINI_KEY_BODY_NAMES) * 3 + 1
    state_space = 0
    num_amp_observations = 3
    amp_observation_space = observation_space

    early_termination = True
    termination_height = 0.32

    motion_file = os.path.join(MOTIONS_DIR, "jini_moonwalk_30hz.npz")
    reference_body = "base_link"
    key_body_names = JINI_KEY_BODY_NAMES
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid pose and joint states)."""

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = JINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
