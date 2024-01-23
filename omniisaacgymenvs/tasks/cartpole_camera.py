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


from gym import spaces
import numpy as np
import torch
import omni.usd
from pxr import UsdGeom

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.cartpole import CartpoleTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole


class CartpoleCameraTask(CartpoleTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = self.camera_width * self.camera_height * 3
        self._num_actions = 1

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)

        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)

    def add_camera(self) -> None:
        stage = get_current_stage()
        camera_path = f"/World/envs/env_0/Camera"
        camera_xform = stage.DefinePrim(f'{camera_path}_Xform', 'Xform')
        # set up transforms for parent and camera prims
        position = (-4.2, 0.0, 3.0)
        rotation = (0, -6.1155, -180)
        UsdGeom.Xformable(camera_xform).AddTranslateOp()
        UsdGeom.Xformable(camera_xform).AddRotateXYZOp()
        camera_xform.GetAttribute('xformOp:translate').Set(position)
        camera_xform.GetAttribute('xformOp:rotateXYZ').Set(rotation)
        camera = stage.DefinePrim(f'{camera_path}_Xform/Camera', 'Camera')
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 90))
        # set camera properties
        camera.GetAttribute('focalLength').Set(24)
        camera.GetAttribute('focusDistance').Set(400)
        # hide other environments in the background
        camera.GetAttribute("clippingRange").Set((0.01, 20.0))

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        self.add_camera()

        RLTask.set_up_scene(self, scene)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        self.render_products = []
        env_pos = self._env_pos.cpu()
        camera_paths = [f"/World/envs/env_{i}/Camera_Xform/Camera" for i in range(self._num_envs)]
        for i in range(self._num_envs):
            render_product = self.rep.create.render_product(camera_paths[i], resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)

        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)
        return

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        self.cart_pos = dof_pos[:, self._cart_dof_idx]
        self.cart_vel = dof_vel[:, self._cart_dof_idx]
        self.pole_pos = dof_pos[:, self._pole_dof_idx]
        self.pole_vel = dof_vel[:, self._pole_dof_idx]

        # retrieve RGB data from all render products
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                img = images/255
                save_image(make_grid(img, nrows = 2), 'cartpole_export.png')

            self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0
        else:
            print("Image tensor is NONE!")

        return self.obs_buf
