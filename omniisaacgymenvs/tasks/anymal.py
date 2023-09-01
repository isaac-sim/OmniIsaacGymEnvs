# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
import torch
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.anymal import Anymal
from omniisaacgymenvs.robots.articulations.views.anymal_view import AnymalView
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive


class AnymalTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._num_observations = 48
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
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
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]

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

    def set_up_scene(self, scene) -> None:
        self.get_anymal()
        super().set_up_scene(scene)
        self._anymals = AnymalView(prim_paths_expr="/World/envs/.*/anymal", name="anymalview")
        scene.add(self._anymals)
        scene.add(self._anymals._knees)
        scene.add(self._anymals._base)

        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("anymalview"):
            scene.remove_object("anymalview", registry_only=True)
        if scene.object_exists("knees_view"):
            scene.remove_object("knees_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        self._anymals = AnymalView(prim_paths_expr="/World/envs/.*/anymal", name="anymalview")
        scene.add(self._anymals)
        scene.add(self._anymals._knees)
        scene.add(self._anymals._base)

    def get_anymal(self):
        anymal = Anymal(
            prim_path=self.default_zero_env_path + "/anymal", name="Anymal", translation=self._anymal_translation
        )
        self._sim_config.apply_articulation_settings(
            "Anymal", get_prim_at_path(anymal.prim_path), self._sim_config.parse_actor_config("Anymal")
        )

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
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale

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

        observations = {self._anymals.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self._anymals.count, dtype=torch.int32, device=self._device)
        self.actions[:] = actions.clone().to(self._device)
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt
        self.current_targets[:] = tensor_clamp(
            current_targets, self.anymal_dof_lower_limits, self.anymal_dof_upper_limits
        )
        self._anymals.set_joint_position_targets(self.current_targets, indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._anymals.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities

        self.current_targets[env_ids] = dof_pos[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._anymals.set_joint_positions(dof_pos, indices)
        self._anymals.set_joint_velocities(dof_vel, indices)

        self._anymals.set_world_poses(
            self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices
        )
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
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

    def post_reset(self):
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )
        dof_names = self._anymals.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.initial_root_pos, self.initial_root_rot = self._anymals.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._anymals.get_dof_limits()
        self.anymal_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.anymal_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(
            (self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_actions = torch.zeros(
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

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )
        rew_cosmetic = (
            torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["cosmetic"]
        )

        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_joint_acc + rew_action_rate + rew_cosmetic + rew_lin_vel_z
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = self._anymals.is_base_below_threshold(threshold=0.51, ground_heights=0.0)
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()

    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
