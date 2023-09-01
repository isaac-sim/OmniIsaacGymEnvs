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


from omniisaacgymenvs.robots.articulations.ant import Ant
from omniisaacgymenvs.tasks.warp.shared.locomotion import LocomotionTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTaskWarp

import numpy as np
import torch
import warp as wp


class AntLocomotionTask(LocomotionTask):
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
        self._num_observations = 60
        self._num_actions = 8
        self._ant_positions = wp.array([0, 0, 0.5], dtype=wp.float32, device="cpu")

        LocomotionTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_ant()
        RLTaskWarp.set_up_scene(self, scene)
        self._ants = ArticulationView(prim_paths_expr="/World/envs/.*/Ant/torso", name="ant_view", reset_xform_properties=False)
        scene.add(self._ants)
        return

    def get_ant(self):
        ant = Ant(prim_path=self.default_zero_env_path + "/Ant", name="Ant", translation=self._ant_positions)
        self._sim_config.apply_articulation_settings("Ant", get_prim_at_path(ant.prim_path), self._sim_config.parse_actor_config("Ant"))

    def get_robot(self):
        return self._ants

    def post_reset(self):
        self.joint_gears = wp.array([15, 15, 15, 15, 15, 15, 15, 15], dtype=wp.float32, device=self._device)
        dof_limits = self._ants.get_dof_limits().to(self._device)
        self.dof_limits_lower = wp.zeros(self._ants._num_dof, dtype=wp.float32, device=self._device)
        self.dof_limits_upper = wp.zeros(self._ants._num_dof, dtype=wp.float32, device=self._device)
        wp.launch(parse_dof_limits, dim=self._ants._num_dof,
            inputs=[self.dof_limits_lower, self.dof_limits_upper, dof_limits], device=self._device)
        self.motor_effort_ratio = wp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=wp.float32, device=self._device)
        self.dof_at_limit_cost = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        force_links = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
        self._sensor_indices = wp.array([self._ants._body_indices[j] for j in force_links], device=self._device, dtype=wp.int32)

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        wp.launch(get_dof_at_limit_cost, dim=(self._num_envs, self._ants._num_dof),
            inputs=[self.dof_at_limit_cost, self.obs_buf, self.motor_effort_ratio])
        return self.dof_at_limit_cost

@wp.kernel
def get_dof_at_limit_cost(dof_at_limit_cost: wp.array(dtype=wp.float32),
                          obs_buf: wp.array(dtype=wp.float32, ndim=2), 
                          motor_effort_ratio: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    dof_i = j + 12

    cost = 0.0
    if wp.abs(obs_buf[i, dof_i]) > 0.99:
        cost = 1.0
    dof_at_limit_cost[i] = cost

@wp.kernel
def parse_dof_limits(dof_limits_lower: wp.array(dtype=wp.float32),
                     dof_limits_upper: wp.array(dtype=wp.float32),
                     dof_limits: wp.array(dtype=wp.float32, ndim=3)):
    tid = wp.tid()
    dof_limits_lower[tid] = dof_limits[0, tid, 0]
    dof_limits_upper[tid] = dof_limits[0, tid, 1]