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
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView
from omniisaacgymenvs.tasks.shared.in_hand_manipulation import InHandManipulationTask


class ShadowHandTask(InHandManipulationTask):
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
        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 187,
        }

        self.asymmetric_obs = self._task_cfg["env"]["asymmetric_observations"]
        self.use_vel_obs = False

        self.fingertip_obs = True
        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10.0

        num_states = 0
        if self.asymmetric_obs:
            num_states = 187

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 20
        self._num_states = num_states
        InHandManipulationTask.update_config(self)

    def get_starting_positions(self):
        self.hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.hand_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self.pose_dy, self.pose_dz = -0.39, 0.10

    def get_hand(self):
        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=self.hand_start_translation,
            orientation=self.hand_start_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)

    def get_hand_view(self, scene):
        hand_view = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(hand_view._fingers)
        return hand_view

    def get_observations(self):
        self.get_object_goal_observations()

        self.fingertip_pos, self.fingertip_rot = self._hands._fingers.get_world_poses(clone=False)
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_velocities = self._hands._fingers.get_velocities(clone=False)

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.vec_sensor_tensor = self._hands.get_measured_joint_forces(
                joint_indices=self._hands._sensor_indices
            ).view(self._num_envs, -1)

        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state(False)
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations

    def compute_fingertip_observations(self, no_vel=False):
        if no_vel:
            # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
            #   Fingertip positions
            #   Object Position, but not orientation
            #   Relative target orientation

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 15)
            self.obs_buf[:, 15:18] = self.object_pos
            self.obs_buf[:, 18:22] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 22:42] = self.actions
        else:
            # 13*self.num_fingertips = 65
            self.obs_buf[:, 0:65] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 15:35] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[:, 35:65] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.obs_buf[:, 65:68] = self.object_pos
            self.obs_buf[:, 68:72] = self.object_rot
            self.obs_buf[:, 72:75] = self.object_linvel
            self.obs_buf[:, 75:78] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 78:81] = self.goal_pos
            self.obs_buf[:, 81:85] = self.goal_rot
            self.obs_buf[:, 85:89] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 89:109] = self.actions

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )

            self.obs_buf[:, 24:37] = self.object_pos
            self.obs_buf[:, 27:31] = self.object_rot
            self.obs_buf[:, 31:34] = self.goal_pos
            self.obs_buf[:, 34:38] = self.goal_rot
            self.obs_buf[:, 38:42] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 57:77] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel

            self.obs_buf[:, 48:51] = self.object_pos
            self.obs_buf[:, 51:55] = self.object_rot
            self.obs_buf[:, 55:58] = self.object_linvel
            self.obs_buf[:, 58:61] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 61:64] = self.goal_pos
            self.obs_buf[:, 64:68] = self.goal_rot
            self.obs_buf[:, 68:72] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # (7+6)*self.num_fingertips = 65
            self.obs_buf[:, 72:87] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 87:107] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[:, 107:137] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.obs_buf[:, 137:157] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.states_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
            # self.states_buf[:, 2*self.num_hand_dofs:3*self.num_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 2 * self.num_hand_dofs  # 48
            self.states_buf[:, obj_obs_start : obj_obs_start + 3] = self.object_pos
            self.states_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.object_rot
            self.states_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start : goal_obs_start + 3] = self.goal_pos
            self.states_buf[:, goal_obs_start + 3 : goal_obs_start + 7] = self.goal_rot
            self.states_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            self.states_buf[
                :, fingertip_obs_start : fingertip_obs_start + 3 * self.num_fingertips
            ] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.states_buf[
                :, fingertip_obs_start + 3 * self.num_fingertips : fingertip_obs_start + 7 * self.num_fingertips
            ] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.states_buf[
                :, fingertip_obs_start + 7 * self.num_fingertips : fingertip_obs_start + 13 * self.num_fingertips
            ] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.states_buf[
                :, fingertip_obs_start + num_ft_states : fingertip_obs_start + num_ft_states + num_ft_force_torques
            ] = (self.force_torque_obs_scale * self.vec_sensor_tensor)

            # obs_end = 72 + 65 + 30 = 167
            # obs_total = obs_end + num_actions = 187
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
            self.obs_buf[:, 2 * self.num_hand_dofs : 3 * self.num_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            obj_obs_start = 3 * self.num_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start : obj_obs_start + 3] = self.object_pos
            self.obs_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.object_rot
            self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start : goal_obs_start + 3] = self.goal_pos
            self.obs_buf[:, goal_obs_start + 3 : goal_obs_start + 7] = self.goal_rot
            self.obs_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            self.obs_buf[
                :, fingertip_obs_start : fingertip_obs_start + 3 * self.num_fingertips
            ] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + 3 * self.num_fingertips : fingertip_obs_start + 7 * self.num_fingertips
            ] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + 7 * self.num_fingertips : fingertip_obs_start + 13 * self.num_fingertips
            ] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + num_ft_states : fingertip_obs_start + num_ft_states + num_ft_force_torques
            ] = (self.force_torque_obs_scale * self.vec_sensor_tensor)

            # obs_end = 96 + 65 + 30 = 167
            # obs_total = obs_end + num_actions = 187
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions
