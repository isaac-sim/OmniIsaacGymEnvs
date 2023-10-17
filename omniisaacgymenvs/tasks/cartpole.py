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
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole


class CartpoleTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("cartpole_view"):
            scene.remove_object("cartpole_view", registry_only=True)
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)

    def get_cartpole(self):
        cartpole = Cartpole(
            prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole")
        )

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        self.cart_pos = dof_pos[:, self._cart_dof_idx]
        self.cart_vel = dof_vel[:, self._cart_dof_idx]
        self.pole_pos = dof_pos[:, self._pole_dof_idx]
        self.pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs_buf[:, 0] = self.cart_pos
        self.obs_buf[:, 1] = self.cart_vel
        self.obs_buf[:, 2] = self.pole_pos
        self.obs_buf[:, 3] = self.pole_vel

        observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        reward = 1.0 - self.pole_pos * self.pole_pos - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)
        reward = torch.where(torch.abs(self.cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(self.pole_pos) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.where(torch.abs(self.cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(self.pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
