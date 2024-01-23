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


from omniisaacgymenvs.robots.articulations.ingenuity import Ingenuity
from omniisaacgymenvs.robots.articulations.views.ingenuity_view import IngenuityView

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask

import numpy as np
import torch
import math


class IngenuityTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        
        self.update_config(sim_config)

        self.thrust_limit = 2000
        self.thrust_lateral_component = 0.2

        self._num_observations = 13
        self._num_actions = 6

        self._ingenuity_position = torch.tensor([0, 0, 1.0])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name=name, env=env)
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
        self.get_ingenuity()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._copters = IngenuityView(prim_paths_expr="/World/envs/.*/Ingenuity", name="ingenuity_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)
        self._balls._non_root_link = True # do not set states for kinematics
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(2):
            scene.add(self._copters.physics_rotors[i])
            scene.add(self._copters.visual_rotors[i])
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("ingenuity_view"):
            scene.remove_object("ingenuity_view", registry_only=True)
        for i in range(2):
            if scene.object_exists(f"physics_rotor_{i}_view"):
                scene.remove_object(f"physics_rotor_{i}_view", registry_only=True)
            if scene.object_exists(f"visual_rotor_{i}_view"):
                scene.remove_object(f"visual_rotor_{i}_view", registry_only=True)
        if scene.object_exists("targets_view"):
            scene.remove_object("targets_view", registry_only=True)
        self._copters = IngenuityView(prim_paths_expr="/World/envs/.*/Ingenuity", name="ingenuity_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False)
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(2):
            scene.add(self._copters.physics_rotors[i])
            scene.add(self._copters.visual_rotors[i])

    def get_ingenuity(self):
        copter = Ingenuity(prim_path=self.default_zero_env_path + "/Ingenuity", name="ingenuity", translation=self._ingenuity_position)
        self._sim_config.apply_articulation_settings("ingenuity", get_prim_at_path(copter.prim_path), self._sim_config.parse_actor_config("ingenuity"))

    def get_target(self):
        radius = 0.1
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball", 
            translation=self._ball_position, 
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = (self.target_positions - root_positions) / 3
        self.obs_buf[..., 3:7] = root_quats
        self.obs_buf[..., 7:10] = root_linvels / 2
        self.obs_buf[..., 10:13] = root_angvels / math.pi

        observations = {
            self._copters.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        vertical_thrust_prop_0 = torch.clamp(actions[:, 2] * self.thrust_limit, -self.thrust_limit, self.thrust_limit)
        vertical_thrust_prop_1 = torch.clamp(actions[:, 5] * self.thrust_limit, -self.thrust_limit, self.thrust_limit)
        lateral_fraction_prop_0 = torch.clamp(
            actions[:, 0:2] * self.thrust_lateral_component,
            -self.thrust_lateral_component,
            self.thrust_lateral_component,
        )
        lateral_fraction_prop_1 = torch.clamp(
            actions[:, 3:5] * self.thrust_lateral_component,
            -self.thrust_lateral_component,
            self.thrust_lateral_component,
        )

        self.thrusts[:, 0, 2] = self.dt * vertical_thrust_prop_0
        self.thrusts[:, 0, 0:2] = self.thrusts[:, 0, 2, None] * lateral_fraction_prop_0
        self.thrusts[:, 1, 2] = self.dt * vertical_thrust_prop_1
        self.thrusts[:, 1, 0:2] = self.thrusts[:, 1, 2, None] * lateral_fraction_prop_1

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        self.dof_vel[:, self.spinning_indices[0]] = 50
        self.dof_vel[:, self.spinning_indices[1]] = -50
        self._copters.set_joint_velocities(self.dof_vel)

        # apply actions
        for i in range(2):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def post_reset(self):
        self.spinning_indices = torch.tensor([1, 3], device=self._device)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1

        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control tensors
        self.thrusts = torch.zeros((self._num_envs, 2, 3), dtype=torch.float32, device=self._device)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (-1, 1) and z in (1, 2)
        self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 2 - 1
        self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 1

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.4
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, 1] = torch_rand_float(-0.2, 0.2, (num_resets, 1), device=self._device).squeeze()
        self.dof_pos[env_ids, 3] = torch_rand_float(-0.2, 0.2, (num_resets, 1), device=self._device).squeeze()
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        # distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + 2.5 * target_dist * target_dist)
        self.target_dist = target_dist
        self.root_positions = root_positions

        # uprightness
        ups = quat_axis(root_quats, 2)
        
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + 30 * tiltage * tiltage)
  
        # spinning
        spinnage = torch.abs(root_angvels[..., 2])
        spinnage_reward = 1.0 / (1.0 + 10 * spinnage * spinnage)

        # combined reward
        # uprightness and spinning only matter when close to the target
        self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spinnage_reward)

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 20.0, ones, die)
        die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
