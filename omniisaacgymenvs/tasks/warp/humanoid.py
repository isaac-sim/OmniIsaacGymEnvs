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


from omniisaacgymenvs.tasks.warp.shared.locomotion import LocomotionTask
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTaskWarp

import numpy as np
import torch
import warp as wp
import math


class HumanoidLocomotionTask(LocomotionTask):
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
        self._num_observations = 87
        self._num_actions = 21
        self._humanoid_positions = torch.tensor([0, 0, 1.34])

        LocomotionTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_humanoid()
        RLTaskWarp.set_up_scene(self, scene)
        self._humanoids = ArticulationView(prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False)
        scene.add(self._humanoids)
        return

    def get_humanoid(self):
        humanoid = Humanoid(prim_path=self.default_zero_env_path + "/Humanoid", name="Humanoid", translation=self._humanoid_positions)
        self._sim_config.apply_articulation_settings("Humanoid", get_prim_at_path(humanoid.prim_path), 
            self._sim_config.parse_actor_config("Humanoid"))

    def get_robot(self):
        return self._humanoids

    def post_reset(self):
        self.joint_gears = wp.array(
            [
                67.5000, # lower_waist
                67.5000, # lower_waist
                67.5000, # right_upper_arm
                67.5000, # right_upper_arm
                67.5000, # left_upper_arm
                67.5000, # left_upper_arm
                67.5000, # pelvis
                45.0000, # right_lower_arm
                45.0000, # left_lower_arm
                45.0000, # right_thigh: x
                135.0000, # right_thigh: y
                45.0000, # right_thigh: z
                45.0000, # left_thigh: x
                135.0000, # left_thigh: y
                45.0000, # left_thigh: z
                90.0000, # right_knee
                90.0000, # left_knee
                22.5, # right_foot
                22.5, # right_foot
                22.5, # left_foot
                22.5, # left_foot
            ],
            device=self._device,
            dtype=wp.float32
        )
        self.max_motor_effort = 135.0
        self.motor_effort_ratio = wp.zeros(self._humanoids._num_dof, dtype=wp.float32, device=self._device)
        wp.launch(compute_effort_ratio, dim=self._humanoids._num_dof, 
            inputs=[self.motor_effort_ratio, self.joint_gears, self.max_motor_effort], device=self._device)

        dof_limits = self._humanoids.get_dof_limits().to(self._device)
        self.dof_limits_lower = wp.zeros(self._humanoids._num_dof, dtype=wp.float32, device=self._device)
        self.dof_limits_upper = wp.zeros(self._humanoids._num_dof, dtype=wp.float32, device=self._device)
        wp.launch(parse_dof_limits, dim=self._humanoids._num_dof,
            inputs=[self.dof_limits_lower, self.dof_limits_upper, dof_limits], device=self._device)
        self.dof_at_limit_cost = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        force_links = ["left_foot", "right_foot"]
        self._sensor_indices = wp.array([self._humanoids._body_indices[j] for j in force_links], device=self._device, dtype=wp.int32)

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        wp.launch(get_dof_at_limit_cost, dim=(self._num_envs, self._humanoids._num_dof),
            inputs=[self.dof_at_limit_cost, self.obs_buf, self.motor_effort_ratio, self.joints_at_limit_cost_scale], device=self._device)
        return self.dof_at_limit_cost

@wp.kernel
def compute_effort_ratio(motor_effort_ratio: wp.array(dtype=wp.float32),
                         joint_gears: wp.array(dtype=wp.float32),
                         max_motor_effort: float):
    tid = wp.tid()
    motor_effort_ratio[tid] = joint_gears[tid] / max_motor_effort

@wp.kernel
def parse_dof_limits(dof_limits_lower: wp.array(dtype=wp.float32),
                     dof_limits_upper: wp.array(dtype=wp.float32),
                     dof_limits: wp.array(dtype=wp.float32, ndim=3)):
    tid = wp.tid()
    dof_limits_lower[tid] = dof_limits[0, tid, 0]
    dof_limits_upper[tid] = dof_limits[0, tid, 1]

@wp.kernel
def get_dof_at_limit_cost(dof_at_limit_cost: wp.array(dtype=wp.float32),
                          obs_buf: wp.array(dtype=wp.float32, ndim=2), 
                          motor_effort_ratio: wp.array(dtype=wp.float32), 
                          joints_at_limit_cost_scale: float):
    i, j = wp.tid()
    dof_i = j + 12

    scaled_cost = joints_at_limit_cost_scale * (wp.abs(obs_buf[i, dof_i]) - 0.98) / 0.02
    cost = 0.0
    if wp.abs(obs_buf[i, dof_i]) > 0.98:
        cost = scaled_cost * motor_effort_ratio[j]
    dof_at_limit_cost[i] = cost
