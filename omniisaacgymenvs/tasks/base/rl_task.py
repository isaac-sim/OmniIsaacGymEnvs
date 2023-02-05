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


from abc import abstractmethod
import numpy as np
import torch
from gym import spaces
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner
from omniisaacgymenvs.tasks.utils.usd_utils import create_distant_light
from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer
import omni.kit
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.kit.viewport.utility import get_viewport_from_window_name
from pxr import Gf

class RLTask(BaseTask):

    """ This class provides a PyTorch RL-specific interface for setting up RL tasks. 
        It includes utilities for setting up RL task related parameters,
        cloning environments, and data collection for RL algorithms.
    """

    def __init__(self, name, env, offset=None) -> None:

        """ Initializes RL parameters, cloner object, and buffers.

        Args:
            name (str): name of the task.
            env (VecEnvBase): an instance of the environment wrapper class to register task.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, offset=offset)

        # optimization flags for pytorch JIT
        torch._C._jit_set_nvfuser_enabled(False)

        self.test = self._cfg["test"]
        self._device = self._cfg["sim_device"]
        self._dr_randomizer = Randomizer(self._sim_config)
        print("Task Device:", self._device)

        self.randomize_actions = False
        self.randomize_observations = False

        self.clip_obs = self._cfg["task"]["env"].get("clipObservations", np.Inf)
        self.clip_actions = self._cfg["task"]["env"].get("clipActions", np.Inf)
        self.rl_device = self._cfg.get("rl_device", "cuda:0")

        self.control_frequency_inv = self._cfg["task"]["env"].get("controlFrequencyInv", 1)

        print("RL device: ", self.rl_device)

        self._env = env

        if not hasattr(self, "_num_agents"):
            self._num_agents = 1  # used for multi-agent environments
        if not hasattr(self, "_num_states"):
            self._num_states = 0

        # initialize data spaces (defaults to gym.Box)
        if not hasattr(self, "action_space"):
            self.action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        if not hasattr(self, "state_space"):
            self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self.cleanup()

    def cleanup(self) -> None:
        """ Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(self, scene, replicate_physics=True) -> None:
        """ Clones environments based on value provided in task config and applies collision filters to mask 
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
            replicate_physics (bool): Clone physics using PhysX API for better performance
        """

        super().set_up_scene(scene)

        collision_filter_global_paths = list()
        if self._sim_config.task_config["sim"].get("add_ground_plane", True):
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics)
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)
        self.set_initial_camera_params(camera_position=[10, 10, 3], camera_target=[0, 0, 0])
        if self._sim_config.task_config["sim"].get("add_distant_light", True):
            create_distant_light()
    
    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        if self._env._render:
            viewport_api_2 = get_viewport_from_window_name("Viewport")
            viewport_api_2.set_active_camera("/OmniverseKit_Persp")
            camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api_2)
            camera_state.set_position_world(Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]), True)
            camera_state.set_target_world(Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]), True)

    @property
    def default_base_env_path(self):
        """ Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """ Retrieves default path to the first env prim (index 0).

        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"

    @property
    def num_envs(self):
        """ Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_actions(self):
        """ Retrieves dimension of actions.

        Returns:
            num_actions(int): Dimension of actions.
        """
        return self._num_actions

    @property
    def num_observations(self):
        """ Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_observations

    @property
    def num_states(self):
        """ Retrieves dimesion of states.

        Returns:
            num_states(int): Dimension of states.
        """
        return self._num_states

    @property
    def num_agents(self):
        """ Retrieves number of agents for multi-agent environments.

        Returns:
            num_agents(int): Dimension of states.
        """
        return self._num_agents

    def get_states(self):
        """ API for retrieving states buffer, used for asymmetric AC training.

        Returns:
            states_buf(torch.Tensor): States buffer.
        """
        return self.states_buf

    def get_extras(self):
        """ API for retrieving extras data for RL.

        Returns:
            extras(dict): Dictionary containing extras data.
        """
        return self.extras

    def reset(self):
        """ Flags all environments for reset.
        """
        self.reset_buf = torch.ones_like(self.reset_buf)

    def pre_physics_step(self, actions):
        """ Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        pass

    def post_physics_step(self):
        """ Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
