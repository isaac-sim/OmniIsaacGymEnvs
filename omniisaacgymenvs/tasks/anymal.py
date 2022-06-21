# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.anymal import Anymal
from omniisaacgymenvs.robots.articulations.views.anymal_view import AnymalView
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math


class AnymalTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._anymal_translation = torch.tensor([0.0, 0.0, 0.62])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 48
        self._num_actions = 12

        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self.get_anymal()
        super().set_up_scene(scene)
        self._anymals = AnymalView(prim_paths_expr="/World/envs/.*/anymal", name="anymalview")
        scene.add(self._anymals)

        return

    def get_anymal(self):
        anymal = Anymal(prim_path=self.default_zero_env_path + "/anymal", name="Anymal", translation=self._anymal_translation)
        self._sim_config.apply_articulation_settings("Anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("Anymal"))

        # Configure joint properties
        joint_paths = []
        for quadrant in ["LF", "LH", "RF", "RH"]:
            for component, abbrev in [("HIP", "H"), ("THIGH", "K")]:
                joint_paths.append(f"{quadrant}_{component}/{quadrant}_{abbrev}FE")
            joint_paths.append(f"base/{quadrant}_HAA")
        for joint_path in joint_paths:
            set_drive(f"{anymal.prim_path}/{joint_path}", "angular", "position", 0, 400, 40, 1000)

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._anymals.get_world_poses(clone=False)
        root_velocities = self._anymals.get_velocities(clone=False)
        dof_pos = self._anymals.get_joint_positions(clone=False)
        dof_vel = self._anymals.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        dof_pos_scaled = (dof_pos - self.initial_dof_pos) * self.dof_pos_scale

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

        obs = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf[:] = obs

        observations = {
            self._anymals.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self._anymals.count, dtype=torch.int32, device=self._device)
        self.actions[:] = actions.clone().to(self._device)
        targets = self.action_scale * self.actions + self.initial_dof_pos
        self._anymals.set_joint_position_targets(targets, indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions and velocities
        positions_offset = torch_rand_float(0.5, 1.5, (num_resets, self._anymals.num_dof), device=self._device)
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._anymals.num_dof), device=self._device)
        dof_pos = self.initial_dof_pos[env_ids] * positions_offset
        dof_vel = velocities

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._anymals.set_joint_positions(dof_pos, indices)
        self._anymals.set_joint_velocities(dof_vel, indices)

        self._anymals.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices)
        self._anymals.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._anymals.get_world_poses()
        self.initial_dof_pos = self._anymals.get_joint_positions().clone()

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(self._anymals.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._anymals.get_world_poses(clone=False)
        root_velocities = self._anymals.get_velocities(clone=False)
        dof_pos = self._anymals.get_joint_positions(clone=False)
        dof_vel = self._anymals.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        # prepare quantities (TODO: return from obs ?)
        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # torque penalty
        # rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        total_reward = rew_lin_vel_xy + rew_ang_vel_z  # + rew_torque
        total_reward = torch.clip(total_reward, 0.0, None)

        # self.fallen_over = root_transforms[:, 2] < 0.45
        self.fallen_over = self._anymals.is_knee_below_threshold(threshold=0.1)
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()


    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf > self.max_episode_length
        self.reset_buf[:] = time_out | self.fallen_over

