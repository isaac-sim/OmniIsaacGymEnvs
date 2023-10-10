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

from omniisaacgymenvs.tasks.anymal_terrain import AnymalTerrainTask, wrap_to_pi

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import tf_combine

import numpy as np
import torch
import math

import omni
import carb

from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.kit.viewport.utility import get_viewport_from_window_name
from pxr import Sdf


class AnymalTerrainDemo(AnymalTerrainTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        max_num_envs = 128
        if sim_config.task_config["env"]["numEnvs"] >= max_num_envs:
            print(f"num_envs reduced to {max_num_envs} for this demo.")
            sim_config.task_config["env"]["numEnvs"] = max_num_envs
        sim_config.task_config["env"]["learn"]["episodeLength_s"] = 120
        AnymalTerrainTask.__init__(self, name, sim_config, env)
        self.add_noise = False
        self.knee_threshold = 0.05

        self.create_camera()
        self._current_command = [0.0, 0.0, 0.0, 0.0]
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        return
    
    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        self.view_port = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.view_port.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 1
        self._key_to_control = {
            "UP": [T, 0.0, 0.0, 0.0],
            "DOWN": [-T, 0.0, 0.0, 0.0],
            "LEFT": [0.0, T, 0.0, 0.0],
            "RIGHT": [0.0, -T, 0.0, 0.0],
            "Z": [0.0, 0.0, R, 0.0],
            "X": [0.0, 0.0, -R, 0.0],
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                self._current_command = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.view_port.get_active_camera() == self.camera_path:
                        self.view_port.set_active_camera(self.perspective_path)
                    else:
                        self.view_port.set_active_camera(self.camera_path)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._current_command = [0.0, 0.0, 0.0, 0.0]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.view_port.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.view_port.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not an Anymal")
        
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.commands[self._previous_selected_id, 0] = np.random.uniform(self.command_x_range[0], self.command_x_range[1])
            self.commands[self._previous_selected_id, 1] = np.random.uniform(self.command_y_range[0], self.command_y_range[1])
            self.commands[self._previous_selected_id, 2] = 0.0
    
    def _update_camera(self):
        base_pos = self.base_pos[self._selected_id, :].clone()
        base_quat = self.base_quat[self._selected_id, :].clone()

        camera_local_transform = torch.tensor([-1.8, 0.0, 0.6], device=self.device)
        camera_pos = quat_apply(base_quat, camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.view_port)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item()+0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def post_physics_step(self):
        self.progress_buf[:] += 1

        self.refresh_dof_state_tensors()
        self.refresh_body_state_tensors()

        self.update_selected_object()

        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval == 0:
            self.push_robots()
        
        # prepare quantities
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        self.check_termination()

        if self._selected_id is not None:
            self.commands[self._selected_id, :] = torch.tensor(self._current_command, device=self.device)
            self.timeout_buf[self._selected_id] = 0
            self.reset_buf[self._selected_id] = 0

        self.get_states()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.get_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras