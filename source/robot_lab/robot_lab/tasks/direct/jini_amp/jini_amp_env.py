# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from robot_lab.tasks.direct.g1_amp.g1_amp_env import compute_obs, compute_rewards, exp_reward_with_floor
from robot_lab.tasks.direct.g1_amp.motions.motion_loader import MotionLoader

from .jini_amp_env_cfg import JiNiAmpMoonwalkEnvCfg


class JiNiAmpEnv(DirectRLEnv):
    cfg: JiNiAmpMoonwalkEnvCfg

    def __init__(self, cfg: JiNiAmpMoonwalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in self.cfg.key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(self.cfg.key_body_names)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.ground_static_friction,
                    dynamic_friction=self.cfg.ground_dynamic_friction,
                    restitution=self.cfg.ground_restitution,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)
        root_pos_relative = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        key_body_pos_relative = self.robot.data.body_pos_w[:, self.key_body_indexes] - self.scene.env_origins.unsqueeze(
            1
        )
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            root_pos_relative,
            self.robot.data.body_quat_w[:, self.ref_body_index],
            key_body_pos_relative,
            progress,
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        with torch.no_grad():
            current_times = (self.episode_length_buf * self.physics_dt).cpu().numpy()
            (
                ref_dof_positions,
                ref_dof_velocities,
                ref_body_positions,
                ref_body_rotations,
                _,
                _,
            ) = self._motion_loader.sample(num_samples=self.num_envs, times=current_times)

            ref_joint_pos = ref_dof_positions[:, self.motion_dof_indexes]
            ref_joint_vel = ref_dof_velocities[:, self.motion_dof_indexes]
            ref_root_pos = ref_body_positions[:, self.motion_ref_body_index]
            ref_root_quat = ref_body_rotations[:, self.motion_ref_body_index]

        joint_pos_error = torch.square(self.robot.data.joint_pos - ref_joint_pos).sum(dim=-1)
        rew_joint_pos = exp_reward_with_floor(
            joint_pos_error, self.cfg.rew_imitation_joint_pos, self.cfg.imitation_sigma_joint_pos, floor=4.0
        )
        rew_joint_pos = torch.clamp(rew_joint_pos, min=-1.0)

        joint_vel_error = torch.square(self.robot.data.joint_vel - ref_joint_vel).sum(dim=-1)
        rew_joint_vel = exp_reward_with_floor(
            joint_vel_error, self.cfg.rew_imitation_joint_vel, self.cfg.imitation_sigma_joint_vel, floor=6.0
        )
        rew_joint_vel = torch.clamp(rew_joint_vel, min=-1.0)

        current_relative_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        pos_err = torch.square(current_relative_pos - ref_root_pos).sum(dim=-1)
        rew_pos = exp_reward_with_floor(pos_err, self.cfg.rew_imitation_pos, self.cfg.imitation_sigma_pos, floor=4.0)
        rew_pos = torch.clamp(rew_pos, min=-1.0)

        quat_dot = torch.abs(torch.sum(self.robot.data.body_quat_w[:, self.ref_body_index] * ref_root_quat, dim=-1))
        ang_err = 2 * torch.arccos(torch.clamp(quat_dot, -1.0, 1.0))
        rew_rot = self.cfg.rew_imitation_rot * torch.exp(-torch.square(ang_err) / (self.cfg.imitation_sigma_rot**2))

        imitation_reward = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot

        basic_reward, basic_reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,
        )

        total_reward = imitation_reward + basic_reward

        log_dict = {
            "rew_imitation": imitation_reward.mean().item(),
            "rew_joint_pos": rew_joint_pos.mean().item(),
            "rew_joint_vel": rew_joint_vel.mean().item(),
            "rew_pos": rew_pos.mean().item(),
            "rew_rot": rew_rot.mean().item(),
            "error_joint_pos": joint_pos_error.mean().item(),
            "error_joint_vel": joint_vel_error.mean().item(),
            "error_root_pos": pos_err.mean().item(),
            "error_ang": ang_err.mean().item(),
            "total_reward": total_reward.mean().item(),
        }

        for key, value in basic_reward_log.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.mean().item()
            else:
                log_dict[key] = float(value)

        self.extras["log"] = log_dict

        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                for key, value in log_dict.items():
                    agent.track_data(f"Reward / {key}", value)
            except Exception:
                pass

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.03
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]

        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            _,
            _,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        progress = (
            torch.as_tensor(times, device=dof_positions.device, dtype=dof_positions.dtype).unsqueeze(-1)
            / self._motion_loader.duration
        )
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            progress,
        )
        return amp_observation.view(-1, self.amp_observation_size)
