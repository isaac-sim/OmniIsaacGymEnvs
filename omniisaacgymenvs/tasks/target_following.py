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
import torch, math

from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdLux
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats

enable_extension("omni.isaac.sensor")

class TargetFollowingTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        # parse configurations, set task-specific members
        self.update_config(sim_config)
        self._max_episode_length = 512 * 4
        self._num_observations = self.camera_width * self.camera_height * 3
        self._num_actions = 2
        self._use_reward_seg = True  ## If using Segmentation reward shaping

        # use multi-dimensional observation for camera RGB
        self.observation_space = spaces.Box(
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * -np.Inf, 
            np.ones((self.camera_width, self.camera_height, 3), dtype=np.float32) * np.Inf)
        
        self.max_angular_velocity = math.pi
        self.action_space = spaces.Box(-self.max_angular_velocity, self.max_angular_velocity, shape=(2,), dtype=np.float32)
        self.weight_reg = 0.0005 ## discourage robot to move in circle, and encourage to move faster
        self.weight = 10  ## encourage robot close to target
        self.weight_seg = 1  ## encourage target in the camera view
        import warp as wp
        self.wp = wp

        # call parent class’s __init__
        RLTask.__init__(self, name, env)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._robot_positions = torch.tensor([0.0, 0.0, 0.25])
        self._farthest_distance = self._task_cfg["env"]["farthestDistance"]
        self._nearest_distance = self._task_cfg["env"]["nearestDistance"]
        self._target_positions = torch.tensor([self._nearest_distance, 0.0, 0.025])
        self._target_scale = torch.tensor([5.0, 5.0, 5.0])
        
        self.camera_type = self._task_cfg["env"].get("cameraType", 'rgb')
        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]
        self.env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self.camera_channels = 3
        self._export_images = self._task_cfg["env"]["exportImages"]
        
    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("carterv2_view"):
            scene.remove_object("carterv2_view", registry_only=True)
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/CarterV2", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._robots)

        if scene.object_exists("nvidiacube_view"):
            scene.remove_object("nvidiacube_view", registry_only=True)
        self._targets = RigidPrimView(
            prim_paths_expr="/World/envs/.*/NvidiaCube", name="nvidiacube_view", reset_xform_properties=False
        )
        scene.add(self._targets)
        
    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.camera_width, self.camera_height, 3), device=self.device, dtype=torch.float)
        # override segmentation mask buffer for camera data
        self.mask_tensor = torch.zeros((self._num_envs, self.camera_width, self.camera_height), device=self.device)
        if self._use_reward_seg:
            self.last_reward_seg = torch.zeros(self._num_envs, device="cuda")
    
    def get_robot(self):
        from omniisaacgymenvs.robots.articulations.carterv2 import CarterV2
        robot = CarterV2(
            prim_path=self.default_zero_env_path + "/CarterV2", name="CarterV2", position=self._robot_positions
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "CarterV2", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("CarterV2")
        )
    
    def get_target(self):
        from omniisaacgymenvs.robots.articulations.carterv2 import NvidiaCube
        target = NvidiaCube(prim_path=self.default_zero_env_path + "/NvidiaCube", name="NvidiaCube", translation=self._target_positions, scale=self._target_scale
        )

    def _create_dome_light(self, prim_path="/World/defaultDomeLight", intensity=1000):
        stage = get_current_stage()
        light = UsdLux.DomeLight.Define(stage, prim_path)
        light.CreateIntensityAttr().Set(intensity)

    def set_up_scene(self, scene) -> None:
        self.get_robot()
        self.get_target()
        self._create_dome_light()
        super().set_up_scene(scene, replicate_physics=False)
        # RLTask.set_up_scene(self, scene)

        # start replicator to capture image data
        self.rep.orchestrator._orchestrator._is_started = True

        # set up cameras
        self.render_products = []
        self.instance_seg_list = []

        ## Camera positions can be obtained by getting current robot position
        for i in range(self._num_envs):
            camera = f"/World/envs/env_{i}/CarterV2/chassis_link/stereo_cam_right/stereo_cam_right_sensor_frame/camera_sensor_right"
            ## -- Set the camera far clipping range strategically to isolate cloned envs
            from omni.isaac.sensor.scripts.camera import Camera
            camera_sensor = Camera(prim_path=camera)
            camera_sensor.set_clipping_range(far_distance=0.5*self.env_spacing)
            camera_sensor.set_focal_length(0.6)  ## original focal length is 0.24m
            fl = camera_sensor.get_focal_length()
            print(f'focal length: {fl}')
            
            render_product = self.rep.create.render_product(camera, resolution=(self.camera_width, self.camera_height))
            self.render_products.append(render_product)
            if self._use_reward_seg:
                instance_seg = self.rep.AnnotatorRegistry.get_annotator(name="instance_segmentation")
                instance_seg.attach([render_product])
                self.instance_seg_list.append(instance_seg)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
        self.pytorch_writer.attach(self.render_products)
        
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/CarterV2", name="carterv2_view", reset_xform_properties=False
        )
        scene.add(self._robots)

        self._targets = RigidPrimView(
            prim_paths_expr="/World/envs/.*/NvidiaCube", name="nvidiacube_view", reset_xform_properties=False
        )
        scene.add(self._targets)

        return
    
    ## called by post_physics_step()
    def get_observations(self) -> dict:
        self.robot_pos = self._robots.get_world_poses()[0]
        self.target_pos = self._targets.get_world_poses()[0]
        dof_vel = self._robots.get_joint_velocities(clone=False)

        self.robot_vel = dof_vel[:, [self._left_wheel_idx, self._right_wheel_idx]] 

        if self._use_reward_seg:
            ## --- retrieve Segmentation Mask data from all render products as observations
            self.mask_tensor = self.get_seg_masks()
            mask_tensor = self.mask_tensor.clone().unsqueeze(-1).float()
            self.obs_buf = mask_tensor.expand(self.num_envs, self.camera_width, self.camera_height, 3)
            if self._export_images:
                from torchvision.utils import save_image, make_grid
                # save_image(make_grid(torch.swapaxes(self.obs_buf, 1, 3), nrows = 2), 'camera_export.png')
                save_image(make_grid(torch.permute(self.obs_buf, (0, 3, 1, 2)), nrows = 2), 'camera_export.png')  #(B x C x H x W)
        else:
            # --- retrieve RGB data from all render products as observations
            images = self.pytorch_listener.get_rgb_data()
            if images is not None:
                if self._export_images:
                    from torchvision.utils import save_image, make_grid
                    img = images/255
                    save_image(make_grid(img, nrows = 2), 'camera_export.png')
                self.obs_buf = torch.swapaxes(images, 1, 3).clone().float()/255.0  ## NCHW -> NWHC
            else:
                print("Image tensor is NONE!")
        
        return self.obs_buf
    
    def pre_physics_step(self, actions) -> None:  ## actions are [left_wheel_velocity, angular_velocity] in [-1, 1]
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.previous_robot_pos = self._robots.get_world_poses()[0]
        actions = actions.to(self._device)
        ## --- Apply actions to robot wheels ---
        wheel_velocities = torch.zeros((self._num_envs, self._robots.num_dof), dtype=torch.float32, device=self._device)
        wheel_velocities[:, self._left_wheel_idx] = self.max_angular_velocity * actions[:, 0]
        wheel_velocities[:, self._right_wheel_idx] = self.max_angular_velocity * actions[:, 1]

        indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        self._robots.set_joint_velocities(wheel_velocities, indices=indices)


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)        

        # randomize robot orientations
        # quaternion is scalar-first (w, x, y, z). shape is (M, 4).
        zeros = torch.zeros(num_resets, device=self._device).unsqueeze(dim=-1)
        euler_angle_Z_random = math.pi * 2 * torch.rand(num_resets, device=self._device).unsqueeze(dim=-1)
        robot_euler_angles_random = torch.cat((zeros, zeros, euler_angle_Z_random), -1)
        robot_quarternions_random = euler_angles_to_quats(robot_euler_angles_random, device=self._device)

        # randomize target cube positions and orientations
        target_pos = self._target_positions.to(self._device)
        target_pos_X_random = (self._farthest_distance - self._nearest_distance) * torch.rand(num_resets, device=self._device).unsqueeze(dim=-1)
        target_pos_random = torch.cat((target_pos_X_random, zeros, zeros), -1)
        
        euler_angle_Z_random = math.pi * 2 * torch.rand(num_resets, device=self._device).unsqueeze(dim=-1)
        target_euler_angles_random = torch.cat((zeros, zeros, euler_angle_Z_random), -1)
        target_quarternions_random = euler_angles_to_quats(target_euler_angles_random, device=self._device)
 
        # apply resets
        indices = env_ids.to(dtype=torch.int32, device=self._device)
        target_positions_random = target_pos + target_pos_random + self._env_pos[indices] ## global positions, need to add self._env_pos
        robot_pos = self._robot_positions.to(self._device).unsqueeze(0)
        robot_positions = robot_pos.expand(num_resets, -1) + self._env_pos[indices] ## global positions, need to add self._env_pos
        self._targets.set_world_poses(positions=target_positions_random, orientations=target_quarternions_random, indices=indices)  
        self._robots.set_world_poses(positions=robot_positions, orientations=robot_quarternions_random, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self._left_wheel_idx = self._robots.get_dof_index("joint_wheel_left")
        self._right_wheel_idx = self._robots.get_dof_index("joint_wheel_right")
        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:  ## Called by task.post_physics_step()
        reward = torch.zeros(self._num_envs, device="cuda")
        self.current_distance = torch.norm(self.robot_pos - self.target_pos, dim=-1) 
        self.previous_distance = torch.norm(self.previous_robot_pos - self.target_pos, dim=-1) 
        
        ## Reward is to encourage robot be closer to target
        reward = self.weight * (self.previous_distance - self.current_distance)

        ## Add reward penalty for not going straight, and encourage to move faster
        linear_x = (self.robot_vel[:, 0] + self.robot_vel[:, 1]) / 2.0 # (num_envs,)
        angular_z = torch.abs(self.robot_vel[:, 0] - self.robot_vel[:, 1]) # (num_envs,)
        reward_reg = self.weight_reg * (linear_x - angular_z)
        reward = reward + reward_reg

        ## -- Get Segmentation data --
        # retrieve Segmentation data from all render products
        if self._use_reward_seg:      
            reward_seg = torch.zeros(self._num_envs, device="cuda")
            for i in range(self._num_envs):
                reward_seg[i] = torch.sum(self.mask_tensor[i]).float()/(self.camera_width * self.camera_width)               
            reward = reward + self.weight_seg * reward_seg

            ## --- reward shaping ---
            reward = torch.where(reward_seg > 0, reward + 0.1, reward - 0.1)
        
        reward = torch.where(self.current_distance < 1, reward + 100, reward)

        self.rew_buf[:] = reward
    
    def get_seg_masks(self):
        for i in range(self._num_envs):
            if self.progress_buf[i] > 2: ## Step the simulator for several steps before get_data
                instance_seg = self.instance_seg_list[i]
                seg_data = instance_seg.get_data(device="cuda")
                if seg_data["info"]["idToLabels"]:
                    self.instance_data = self.wp.to_torch(seg_data["data"].view(self.wp.int32)).squeeze()
                    # path_to_instance_id = {'BACKGROUND': 0, 'UNLABELLED': 1, '/World/envs/env_0/NvidiaCube': 2, '/World/envs/env_1/NvidiaCube': 3, '/World/envs/env_2/NvidiaCube': 4, '/World/envs/env_3/NvidiaCube': 5}
                    path_to_instance_id = {v: int(k) for k, v in seg_data["info"]["idToLabels"].items()}
                    mask = torch.zeros(*self.instance_data.shape, dtype=bool, device="cuda")
                    instance = f'/World/envs/env_{i}/NvidiaCube'
                    if instance in path_to_instance_id:
                        mask += torch.isin(self.instance_data, path_to_instance_id[instance])
                    self.mask_tensor[i] = mask
                else:
                    self.mask_tensor[i] = torch.zeros((self.camera_width, self.camera_height), device=self.device)
        return self.mask_tensor

    def is_done(self) -> None:
        ## is_done when robot reaches the target...
        # print(f'Curent_distance: {self.current_distance}')
        resets = torch.where(self.current_distance < 0.9, 1, 0)
        for i, idx in enumerate(resets.cpu().numpy()):
            if idx:
                print(f'-- Reached the target in env_{i}')
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
