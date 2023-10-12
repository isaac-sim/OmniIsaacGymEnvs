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


from omni.isaac.kit import SimulationApp
from omni.isaac.gym.vec_env import VecEnvBase
import os
import carb

import torch
import numpy as np

from datetime import datetime

def _new__init__(
        self, headless: bool, sim_device: int = 0, enable_livestream: bool = False, enable_viewport: bool = False,stream_type: str = "webRTC"
    ) -> None:
        """ Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
        """

        experience = ""
        if headless:
            if enable_livestream:
                experience = ""
            elif enable_viewport:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.render.kit'
            else:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp({"headless": headless, "physics_gpu": sim_device}, experience=experience)
        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._render = not headless or enable_livestream or enable_viewport
        self.sim_frame_count = 0

        if enable_livestream:
            from omni.isaac.core.utils.extensions import enable_extension
            if stream_type == "webRTC":
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.services.streamclient.webrtc")
            elif stream_type == "native":
                self._simulation_app.set_setting("/app/livestream/enabled", True)
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.kit.livestream.native")
                enable_extension("omni.services.streaming.manager")
            elif stream_type == "webSocket":
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.services.streamclient.websocket")

            else:
                raise NotImplementedError("unsopported stream type")


VecEnvBase.__init__ = _new__init__

# VecEnv Wrapper for RL training
class VecEnvRLGames(VecEnvBase):

    def _process_data(self):
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._rew = self._rew.to(self._task.rl_device).clone()
        self._states = torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions):
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        self._task.apply_control(actions)
        
        for _ in range(self._task.control_frequency_inv):
            self._task.pre_physics_step()
            self._world.step(render=self._render)
            self.sim_frame_count += 1

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()
        
        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """ Resets the task and applies default zero actions to recompute observations and states. """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.rl_device)
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict
