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


from omniisaacgymenvs.robots.articulations.cartpole import Cartpole

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.warp as warp_utils
from omniisaacgymenvs.tasks.base.rl_task import RLTaskWarp

import numpy as np
import torch
import warp as wp
import math


class CartpoleTask(RLTaskWarp):
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

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = wp.array([0.0, 0.0, 2.0], dtype=wp.float32)

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        RLTaskWarp.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False)
        scene.add(self._cartpoles)
        return

    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        wp.launch(get_observations, dim=self._num_envs, 
            inputs=[self.obs_buf, dof_pos, dof_vel, self._cart_dof_idx, self._pole_dof_idx], device=self._device)

        observations = {
            self._cartpoles.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        self.reset_idx()

        actions_wp = wp.from_torch(actions)

        forces = wp.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=wp.float32, device=self._device)
        wp.launch(compute_forces, dim=self._num_envs, 
            inputs=[forces, actions_wp, self._cart_dof_idx, self._max_push_effort], device=self._device)
        self._cartpoles.set_joint_efforts(forces)

    def reset_idx(self):
        reset_env_ids = wp.to_torch(self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        num_resets = len(reset_env_ids)
        indices = wp.from_torch(reset_env_ids.to(dtype=torch.int32), dtype=wp.int32)

        if num_resets > 0:
            wp.launch(reset_idx, num_resets, 
                inputs=[self.dof_pos, self.dof_vel, indices, self.reset_buf, self.progress_buf, self._cart_dof_idx, self._pole_dof_idx, self._rand_seed], 
                device=self._device)

            # apply resets
            self._cartpoles.set_joint_positions(self.dof_pos[indices], indices=indices)
            self._cartpoles.set_joint_velocities(self.dof_vel[indices], indices=indices)

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")

        self.dof_pos = wp.zeros((self._num_envs, self._cartpoles.num_dof), device=self._device, dtype=wp.float32)
        self.dof_vel = wp.zeros((self._num_envs, self._cartpoles.num_dof), device=self._device, dtype=wp.float32)

        # randomize all envs
        self.reset_idx()

    def calculate_metrics(self) -> None:
        wp.launch(calculate_metrics, dim=self._num_envs, 
            inputs=[self.obs_buf, self.rew_buf, self._reset_dist], device=self._device)

    def is_done(self) -> None:
        wp.launch(is_done, dim=self._num_envs, 
            inputs=[self.obs_buf, self.reset_buf, self.progress_buf, self._reset_dist, self._max_episode_length], 
            device=self._device)


@wp.kernel
def reset_idx(dof_pos: wp.array(dtype=wp.float32, ndim=2), 
              dof_vel: wp.array(dtype=wp.float32, ndim=2), 
              indices: wp.array(dtype=wp.int32), 
              reset_buf: wp.array(dtype=wp.int32), 
              progress_buf: wp.array(dtype=wp.int32),
              cart_dof_idx: int,
              pole_dof_idx: int,
              rand_seed: int):
    i = wp.tid()
    idx = indices[i]

    rand_state = wp.rand_init(rand_seed, i)

    # randomize DOF positions
    dof_pos[idx, cart_dof_idx] = 1.0 * (1.0 - 2.0 * wp.randf(rand_state))
    dof_pos[idx, pole_dof_idx] = 0.125 * warp_utils.PI * (1.0 - 2.0 * wp.randf(rand_state))

    # randomize DOF velocities
    dof_vel[idx, cart_dof_idx] = 0.5 * (1.0 - 2.0 * wp.randf(rand_state))
    dof_vel[idx, pole_dof_idx] = 0.25 * warp_utils.PI * (1.0 - 2.0 * wp.randf(rand_state))

    # bookkeeping
    progress_buf[idx] = 0
    reset_buf[idx] = 0

@wp.kernel
def compute_forces(forces: wp.array(dtype=wp.float32, ndim=2), 
                   actions: wp.array(dtype=wp.float32, ndim=2), 
                   cart_dof_idx: int,
                   max_push_effort: float):
    i = wp.tid()
    forces[i, cart_dof_idx] = max_push_effort * actions[i, 0]


@wp.kernel
def get_observations(obs_buf: wp.array(dtype=wp.float32, ndim=2),
                     dof_pos: wp.indexedarray(dtype=wp.float32, ndim=2),
                     dof_vel: wp.indexedarray(dtype=wp.float32, ndim=2),
                     cart_dof_idx: int,
                     pole_dof_idx: int):
    i = wp.tid()
    obs_buf[i, 0] = dof_pos[i, cart_dof_idx]
    obs_buf[i, 1] = dof_vel[i, cart_dof_idx]
    obs_buf[i, 2] = dof_pos[i, pole_dof_idx]
    obs_buf[i, 3] = dof_vel[i, pole_dof_idx]


@wp.kernel
def calculate_metrics(obs_buf: wp.array(dtype=wp.float32, ndim=2),
                    rew_buf: wp.array(dtype=wp.float32),
                    reset_dist: float):
    i = wp.tid()

    cart_pos = obs_buf[i, 0]
    cart_vel = obs_buf[i, 1]
    pole_angle = obs_buf[i, 2]
    pole_vel = obs_buf[i, 3]

    rew_buf[i] = 1.0 - pole_angle * pole_angle - 0.01 * wp.abs(cart_vel) - 0.005 * wp.abs(pole_vel)
    if wp.abs(cart_pos) > reset_dist or wp.abs(pole_angle) > warp_utils.PI / 2.0:
        rew_buf[i] = -2.0


@wp.kernel
def is_done(obs_buf: wp.array(dtype=wp.float32, ndim=2),
            reset_buf: wp.array(dtype=wp.int32),
            progress_buf: wp.array(dtype=wp.int32),
            reset_dist: float,
            max_episode_length: int):
    
    i = wp.tid()

    cart_pos = obs_buf[i, 0]
    pole_pos = obs_buf[i, 2]

    if wp.abs(cart_pos) > reset_dist or wp.abs(pole_pos) > warp_utils.PI / 2.0 or progress_buf[i] > max_episode_length:
        reset_buf[i] = 1
    else:
        reset_buf[i] = 0
