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
from omni.isaac.core.utils.torch import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.allegro_hand import AllegroHand
from omniisaacgymenvs.robots.articulations.views.allegro_hand_view import AllegroHandView
from omniisaacgymenvs.tasks.shared.in_hand_manipulation import InHandManipulationTask


class AllegroHandTask(InHandManipulationTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)

        InHandManipulationTask.__init__(self, name=name, env=env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full_no_vel", "full"]):
            raise Exception("Unknown type of observations!\nobservationType should be one of: [full_no_vel, full]")
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
        }

        self.object_scale = torch.tensor([1.0, 1.0, 1.0])

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 16
        self._num_states = 0

        InHandManipulationTask.update_config(self)

    def get_starting_positions(self):
        self.hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.hand_start_orientation = torch.tensor([0.257551, 0.283045, 0.683330, -0.621782], device=self.device)
        self.pose_dy, self.pose_dz = -0.2, 0.06

    def get_hand(self):
        allegro_hand = AllegroHand(
            prim_path=self.default_zero_env_path + "/allegro_hand",
            name="allegro_hand",
            translation=self.hand_start_translation,
            orientation=self.hand_start_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "allegro_hand",
            get_prim_at_path(allegro_hand.prim_path),
            self._sim_config.parse_actor_config("allegro_hand"),
        )
        allegro_hand_prim = self._stage.GetPrimAtPath(allegro_hand.prim_path)
        allegro_hand.set_allegro_hand_properties(stage=self._stage, allegro_hand_prim=allegro_hand_prim)
        allegro_hand.set_motor_control_mode(
            stage=self._stage, allegro_hand_path=self.default_zero_env_path + "/allegro_hand"
        )

    def get_hand_view(self, scene):
        return AllegroHandView(prim_paths_expr="/World/envs/.*/allegro_hand", name="allegro_hand_view")

    def get_observations(self):
        self.get_object_goal_observations()

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unkown observations type!")

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )

            self.obs_buf[:, 16:19] = self.object_pos
            self.obs_buf[:, 19:23] = self.object_rot
            self.obs_buf[:, 23:26] = self.goal_pos
            self.obs_buf[:, 26:30] = self.goal_rot
            self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 34:50] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel

            self.obs_buf[:, 32:35] = self.object_pos
            self.obs_buf[:, 35:39] = self.object_rot
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:48] = self.goal_pos
            self.obs_buf[:, 48:52] = self.goal_rot
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 56:72] = self.actions
