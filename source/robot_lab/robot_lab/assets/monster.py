# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

MONSTER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/monster/monster.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*hip_roll_joint_.*": 0.0,
            ".*hip_pitch_joint_F.*": 0.8,
            ".*knee_pitch_joint_F.*": -1.5,
            ".*hip_pitch_joint_H.*": -0.8,
            ".*knee_pitch_joint_H.*": 1.5,
            ".*wheel_joint_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll_joint_.*", ".*hip_pitch_joint_.*", ".*knee_pitch_joint_.*"],
            effort_limit_sim={
                ".*hip_roll_joint_.*": 120.0,
                ".*hip_pitch_joint_.*": 120.0,
                ".*knee_pitch_joint_.*": 140.0,
            },
            velocity_limit_sim={
                ".*hip_roll_joint_.*": 20.0,
                ".*hip_pitch_joint_.*": 20.0,
                ".*knee_pitch_joint_.*": 20.0,
            },
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel_joint_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=80.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of the Monster wheeled quadruped."""
