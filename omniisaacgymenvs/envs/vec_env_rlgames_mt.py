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


from omni.isaac.gym.vec_env import VecEnvMT
from omni.isaac.gym.vec_env import TaskStopException

from .vec_env_rlgames import VecEnvRLGames

import torch
import numpy as np


# VecEnv Wrapper for RL training
class VecEnvRLGamesMT(VecEnvRLGames, VecEnvMT):

    def _parse_data(self, data):
        self._obs = data["obs"].clone()
        self._rew = data["rew"].to(self._task.rl_device).clone()
        self._states = torch.clamp(data["states"], -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
        self._resets = data["reset"].to(self._task.rl_device).clone()
        self._extras = data["extras"].copy()

    def step(self, actions):
        if self._stop:
            raise TaskStopException()

        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        self.send_actions(actions)
        data = self.get_data()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(observations=self._obs.to(self._task.rl_device), reset_buf=self._task.reset_buf)
        
        self._obs = torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device)
        
        obs_dict = {}
        obs_dict["obs"] = self._obs
        obs_dict["states"] = self._states

        return obs_dict, self._rew, self._resets, self._extras
