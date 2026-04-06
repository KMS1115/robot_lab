# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

JINI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/jini/jini.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={
            ".*Hip-Pitch": -0.25,
            ".*Knee-Pitch": 0.5,
            ".*Ankle-Pitch": -0.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Hip-Yaw",
                ".*Hip-Roll",
                ".*Hip-Pitch",
                ".*Knee-Pitch",
                ".*Ankle-Pitch",
            ],
            effort_limit_sim={
                ".*Hip-Yaw": 60.0,
                ".*Hip-Roll": 60.0,
                ".*Hip-Pitch": 100.0,
                ".*Knee-Pitch": 120.0,
                ".*Ankle-Pitch": 60.0,
            },
            velocity_limit_sim=12.0,
            stiffness={
                ".*Hip-Yaw": 80.0,
                ".*Hip-Roll": 80.0,
                ".*Hip-Pitch": 100.0,
                ".*Knee-Pitch": 140.0,
                ".*Ankle-Pitch": 60.0,
            },
            damping={
                ".*Hip-Yaw": 3.0,
                ".*Hip-Roll": 3.0,
                ".*Hip-Pitch": 4.0,
                ".*Knee-Pitch": 6.0,
                ".*Ankle-Pitch": 2.5,
            },
            armature=0.01,
        ),
    },
)
"""Configuration for the JiNi biped robot."""
