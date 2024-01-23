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
from omni.isaac.core.utils.torch.maths import tensor_clamp, torch_rand_float, unscale
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.humanoid import Humanoid
from omniisaacgymenvs.tasks.shared.locomotion import LocomotionTask
from pxr import PhysxSchema


class HumanoidLocomotionTask(LocomotionTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._num_observations = 87
        self._num_actions = 21
        self._humanoid_positions = torch.tensor([0, 0, 1.34])

        LocomotionTask.__init__(self, name=name, env=env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        LocomotionTask.update_config(self)

    def set_up_scene(self, scene) -> None:
        self.get_humanoid()
        RLTask.set_up_scene(self, scene)
        self._humanoids = ArticulationView(
            prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False
        )
        scene.add(self._humanoids)
        return

    def initialize_views(self, scene):
        RLTask.initialize_views(self, scene)
        if scene.object_exists("humanoid_view"):
            scene.remove_object("humanoid_view", registry_only=True)
        self._humanoids = ArticulationView(
            prim_paths_expr="/World/envs/.*/Humanoid/torso", name="humanoid_view", reset_xform_properties=False
        )
        scene.add(self._humanoids)

    def get_humanoid(self):
        humanoid = Humanoid(
            prim_path=self.default_zero_env_path + "/Humanoid", name="Humanoid", translation=self._humanoid_positions
        )
        self._sim_config.apply_articulation_settings(
            "Humanoid", get_prim_at_path(humanoid.prim_path), self._sim_config.parse_actor_config("Humanoid")
        )

    def get_robot(self):
        return self._humanoids

    def post_reset(self):
        self.joint_gears = torch.tensor(
            [
                67.5000,  # lower_waist
                67.5000,  # lower_waist
                67.5000,  # right_upper_arm
                67.5000,  # right_upper_arm
                67.5000,  # left_upper_arm
                67.5000,  # left_upper_arm
                67.5000,  # pelvis
                45.0000,  # right_lower_arm
                45.0000,  # left_lower_arm
                45.0000,  # right_thigh: x
                135.0000,  # right_thigh: y
                45.0000,  # right_thigh: z
                45.0000,  # left_thigh: x
                135.0000,  # left_thigh: y
                45.0000,  # left_thigh: z
                90.0000,  # right_knee
                90.0000,  # left_knee
                22.5,  # right_foot
                22.5,  # right_foot
                22.5,  # left_foot
                22.5,  # left_foot
            ],
            device=self._device,
        )
        self.max_motor_effort = torch.max(self.joint_gears)
        self.motor_effort_ratio = self.joint_gears / self.max_motor_effort
        dof_limits = self._humanoids.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        force_links = ["left_foot", "right_foot"]
        self._sensor_indices = torch.tensor(
            [self._humanoids._body_indices[j] for j in force_links], device=self._device, dtype=torch.long
        )

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self.motor_effort_ratio, self.joints_at_limit_cost_scale)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, motor_effort_ratio, joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, float) -> Tensor
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1
    )
    return dof_at_limit_cost
