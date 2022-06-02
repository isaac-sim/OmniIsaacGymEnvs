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


from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.shared.in_hand_manipulation import InHandManipulationTask
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *

import numpy as np
import torch
import math


class ShadowHandTask(InHandManipulationTask):
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

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full_no_vel": 77,
            "full": 157,
        }

        self.fingertip_obs = True
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.num_fingertips = len(self.fingertips)

        self.object_scale = torch.tensor([0.8, 0.8, 0.8])

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 20
        self._num_states = 0

        InHandManipulationTask.__init__(self, name=name, env=env)
        return

    def get_hand(self):
        hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        hand_start_orientation = torch.tensor([0.0, 0.0, -0.70711, 0.70711], device=self.device)

        shadow_hand = ShadowHand(prim_path=self.default_zero_env_path + "/shadow_hand", 
                                 name="shadow_hand",
                                 translation=hand_start_translation, 
                                 orientation=hand_start_orientation,
                                )
        self._sim_config.apply_articulation_settings("shadow_hand", get_prim_at_path(shadow_hand.prim_path), self._sim_config.parse_actor_config("shadow_hand"))
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)
        pose_dy, pose_dz = -0.39, 0.10
        return hand_start_translation, pose_dy, pose_dz
    
    def get_hand_view(self, scene):
        hand_view = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(hand_view._fingers)
        return hand_view

    def get_observations(self):
        self.get_object_goal_observations()

        self.fingertip_pos, self.fingertip_rot = self._hands._fingers.get_world_poses()
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(self.num_envs * self.num_fingertips, 3)
        self.fingertip_velocities = self._hands._fingers.get_velocities()

        self.hand_dof_pos = self._hands.get_joint_positions()
        self.hand_dof_vel = self._hands.get_joint_velocities()

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unkown observations type!")
        
        observations = {
            self._hands.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:self.num_hand_dofs] = unscale(self.hand_dof_pos,
                self.hand_dof_lower_limits, self.hand_dof_upper_limits)
            
            self.obs_buf[:, 24:37] = self.object_pos
            self.obs_buf[:, 27:31] = self.object_rot
            self.obs_buf[:, 31:34] = self.goal_pos
            self.obs_buf[:, 34:38] = self.goal_rot
            self.obs_buf[:, 38:42] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 3*self.num_fingertips)
            self.obs_buf[:, 57:77] = self.actions
        else:
            self.obs_buf[:, 0:self.num_hand_dofs] = unscale(self.hand_dof_pos,
                self.hand_dof_lower_limits, self.hand_dof_upper_limits)
            self.obs_buf[:, self.num_hand_dofs:2*self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
            
            self.obs_buf[:, 48:51] = self.object_pos
            self.obs_buf[:, 51:55] = self.object_rot
            self.obs_buf[:, 55:58] = self.object_linvel
            self.obs_buf[:, 58:61] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 61:64] = self.goal_pos
            self.obs_buf[:, 64:68] = self.goal_rot
            self.obs_buf[:, 68:72] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # (7+6)*self.num_fingertips = 65
            self.obs_buf[:, 72:87] = self.fingertip_pos.reshape(self.num_envs, 3*self.num_fingertips)
            self.obs_buf[:, 87:107] = self.fingertip_rot.reshape(self.num_envs, 4*self.num_fingertips)
            self.obs_buf[:, 107:137] = self.fingertip_velocities.reshape(self.num_envs, 6*self.num_fingertips)
           
            self.obs_buf[:, 137:157] = self.actions