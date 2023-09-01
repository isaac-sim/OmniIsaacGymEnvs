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
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.quadcopter import Quadcopter
from omniisaacgymenvs.robots.articulations.views.quadcopter_view import QuadcopterView


class QuadcopterTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self._num_observations = 21
        self._num_actions = 12

        self._copter_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name=name, env=env)

        max_thrust = 2.0
        self.thrust_lower_limits = -max_thrust * torch.ones(4, device=self._device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

    def set_up_scene(self, scene) -> None:
        self.get_copter()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._copters = QuadcopterView(prim_paths_expr="/World/envs/.*/Quadcopter", name="quadcopter_view")
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False
        )
        self._balls._non_root_link = True # do not set states for kinematics
        scene.add(self._copters)
        scene.add(self._copters.rotors)
        scene.add(self._balls)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("quadcopter_view"):
            scene.remove_object("quadcopter_view", registry_only=True)
        if scene.object_exists("rotors_view"):
            scene.remove_object("rotors_view", registry_only=True)
        if scene.object_exists("targets_view"):
            scene.remove_object("targets_view", registry_only=True)
        self._copters = QuadcopterView(prim_paths_expr="/World/envs/.*/Quadcopter", name="quadcopter_view")
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False
        )
        scene.add(self._copters)
        scene.add(self._copters.rotors)
        scene.add(self._balls)

    def get_copter(self):
        copter = Quadcopter(
            prim_path=self.default_zero_env_path + "/Quadcopter", name="quadcopter", translation=self._copter_position
        )
        self._sim_config.apply_articulation_settings(
            "copter", get_prim_at_path(copter.prim_path), self._sim_config.parse_actor_config("copter")
        )

    def get_target(self):
        radius = 0.05
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)
        self.dof_pos = self._copters.get_joint_positions(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = (self.target_positions - root_positions) / 3
        self.obs_buf[..., 3:7] = root_quats
        self.obs_buf[..., 7:10] = root_linvels / 2
        self.obs_buf[..., 10:13] = root_angvels / math.pi
        self.obs_buf[..., 13:21] = self.dof_pos

        observations = {self._copters.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)

        dof_action_speed_scale = 8 * math.pi
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[:, 0:8]
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.dof_lower_limits, self.dof_upper_limits
        )

        thrust_action_speed_scale = 100
        self.thrusts += self.dt * thrust_action_speed_scale * actions[:, 8:12]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)

        self.forces[:, 0, 2] = self.thrusts[:, 0]
        self.forces[:, 1, 2] = self.thrusts[:, 1]
        self.forces[:, 2, 2] = self.thrusts[:, 2]
        self.forces[:, 3, 2] = self.thrusts[:, 3]

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.dof_position_targets[reset_env_ids] = self.dof_pos[reset_env_ids]

        # apply actions
        self._copters.set_joint_position_targets(self.dof_position_targets)
        self._copters.rotors.apply_forces(self.forces, is_global=False)

    def post_reset(self):
        # control tensors
        self.dof_position_targets = torch.zeros(
            (self._num_envs, self._copters.num_dof), dtype=torch.float32, device=self._device, requires_grad=False
        )
        self.thrusts = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device, requires_grad=False)
        self.forces = torch.zeros(
            (self._num_envs, self._copters.rotors.count // self._num_envs, 3),
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device)
        self.target_positions[:, 2] = 1.0

        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)
        self.dof_pos = self._copters.get_joint_positions(clone=False)
        self.dof_vel = self._copters.get_joint_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        dof_limits = self._copters.get_dof_limits()
        self.dof_lower_limits = dof_limits[0][:, 0].to(device=self._device)
        self.dof_upper_limits = dof_limits[0][:, 1].to(device=self._device)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.2, 0.2, (num_resets, self._copters.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        self._balls.set_world_poses(positions=self.target_positions[:, 0:3] + self._env_pos)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        # distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + 3 * target_dist * target_dist)  # 2
        self.target_dist = target_dist
        self.root_positions = root_positions

        # uprightness
        ups = quat_axis(root_quats, 2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + 10 * tiltage * tiltage)

        # spinning
        spinnage = torch.abs(root_angvels[..., 2])
        spinnage_reward = 1.0 / (1.0 + 0.001 * spinnage * spinnage)

        rew = pos_reward + pos_reward * (up_reward + spinnage_reward + spinnage * spinnage * (-1 / 400))
        rew = torch.clip(rew, 0.0, None)
        self.rew_buf[:] = rew

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 3.0, ones, die)
        die = torch.where(self.root_positions[..., 2] < 0.3, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
