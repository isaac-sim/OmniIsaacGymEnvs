# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.physx.scripts import deformableUtils, physicsUtils

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema


class FrankaDeformableTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self.update_config(sim_config)
        self.dt = 1/60.
        self._num_observations = 39
        self._num_actions = 9

        RLTask.__init__(self, name, env)
        return
    

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["actionScale"]


    def set_up_scene(self, scene) -> None:
        self.stage = get_current_stage()
        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        self.get_franka()
        self.get_beaker()
        self.get_deformable_tube()
        
        super().set_up_scene(scene=scene, replicate_physics=False)
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self.deformableView = DeformablePrimView(
            prim_paths_expr="/World/envs/.*/deformableTube/tube/mesh", name="deformabletube_view"
        )
        
        scene.add(self.deformableView)
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        return


    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("deformabletube_view"):
            scene.remove_object("deformabletube_view", registry_only=True)
        self._frankas = FrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="franka_view"
        )
        self.deformableView = DeformablePrimView(
            prim_paths_expr="/World/envs/.*/deformableTube/tube/mesh", name="deformabletube_view"
        )
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self.deformableView)


    def get_franka(self):
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka", 
            name="franka", 
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            translation=torch.tensor([0.0, 0.0, 0.0]),
        )
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )
        franka.set_franka_properties(stage=self.stage, prim=franka.prim)


    def get_beaker(self):
        _usd_path = self.assets_root_path + "/Isaac/Props/Beaker/beaker_500ml.usd"
        mesh_path = self.default_zero_env_path + "/beaker"
        add_reference_to_stage(_usd_path, mesh_path)

        beaker = RigidPrim(
            prim_path=mesh_path+"/beaker",
            name="beaker",
            position=torch.tensor([0.5, 0.2, 0.095]),
        )

        self._sim_config.apply_articulation_settings("beaker", beaker.prim, self._sim_config.parse_actor_config("beaker"))


    def get_deformable_tube(self):
        _usd_path = self.assets_root_path + "/Isaac/Props/DeformableTube/tube.usd"
        mesh_path = self.default_zero_env_path + "/deformableTube/tube"
        add_reference_to_stage(_usd_path, mesh_path)

        skin_mesh = get_prim_at_path(mesh_path)
        physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
        physicsUtils.set_or_add_translate_op(skin_mesh, (0.6, 0.0, 0.005))
        physicsUtils.set_or_add_orient_op(skin_mesh, Gf.Rotation(Gf.Vec3d([0, 0, 1]), 90).GetQuat())


    def get_observations(self) -> dict:
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        dof_pos_scaled = (
            2.0 * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        self.lfinger_pos, _ = self._frankas._lfingers.get_world_poses(clone=False)
        self.rfinger_pos, _ = self._frankas._rfingers.get_world_poses(clone=False)
        self.gripper_site_pos = (self.lfinger_pos + self.rfinger_pos)/2 - self._env_pos

        tube_positions = self.deformableView.get_simulation_mesh_nodal_positions(clone=False)
        tube_velocities = self.deformableView.get_simulation_mesh_nodal_velocities(clone=False)

        self.tube_front_positions = tube_positions[:, 200, :] - self._env_pos
        self.tube_front_velocities = tube_velocities[:, 200, :]
        self.tube_back_positions = tube_positions[:, -1, :] - self._env_pos
        self.tube_back_velocities = tube_velocities[:, -1, :]

        front_to_gripper = self.tube_front_positions - self.gripper_site_pos
        to_front_goal = self.front_goal_pos - self.tube_front_positions
        to_back_goal = self.back_goal_pos - self.tube_back_positions

        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                front_to_gripper,
                to_front_goal,
                to_back_goal,
                self.tube_front_positions,
                self.tube_front_velocities,
                self.tube_back_positions,
                self.tube_back_velocities,
            ),
            dim=-1,
        )
       
        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_targets[:, -1] = self.franka_dof_targets[:, -2]

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        pos = self.franka_default_dof_pos 
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        self.deformableView.set_simulation_mesh_nodal_positions(self.initial_tube_positions[env_ids], indices)
        self.deformableView.set_simulation_mesh_nodal_velocities(self.initial_tube_velocities[env_ids], indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):
        self.franka_default_dof_pos = torch.tensor(
            [0.00, 0.63, 0.00, -2.15, 0.00, 2.76, 0.75, 0.02, 0.02], device=self._device
        )
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        self.front_goal_pos = torch.tensor([0.36, 0.0, 0.23], device=self._device).repeat((self._num_envs, 1))
        self.back_goal_pos = torch.tensor([0.5, 0.2, 0.0], device=self._device).repeat((self._num_envs, 1))

        self.goal_hand_rot = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self._device).repeat((self.num_envs, 1))
        self.lfinger_pos, _ = self._frankas._lfingers.get_world_poses(clone=False)
        self.rfinger_pos, _ = self._frankas._rfingers.get_world_poses(clone=False)

        self.gripper_site_pos = (self.lfinger_pos + self.rfinger_pos)/2 - self._env_pos

        self.initial_tube_positions = self.deformableView.get_simulation_mesh_nodal_positions()
        self.initial_tube_velocities = self.deformableView.get_simulation_mesh_nodal_velocities()

        self.tube_front_positions = self.initial_tube_positions[:, 0, :] - self._env_pos
        self.tube_front_velocities = self.initial_tube_velocities[:, 0, :]
        self.tube_back_positions = self.initial_tube_positions[:, -1, :] - self._env_pos
        self.tube_back_velocities = self.initial_tube_velocities[:, -1, :]

        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)



    def calculate_metrics(self) -> None:
        goal_distance_error = torch.norm(self.tube_back_positions[:, 0:2] - self.back_goal_pos[:, 0:2], p = 2, dim = -1)
        goal_dist_reward = 1.0 / (5*goal_distance_error + .025)

        current_z_level = self.tube_back_positions[:, 2:3]
        z_lift_level = torch.where(
            goal_distance_error < 0.07, torch.zeros_like(current_z_level), torch.ones_like(current_z_level)*0.18
        )

        front_lift_error = torch.norm(current_z_level - z_lift_level, p = 2, dim = -1)
        front_lift_reward = 1.0 / (5*front_lift_error + .025)

        rewards = goal_dist_reward + 4*front_lift_reward

        self.rew_buf[:] = rewards


    def is_done(self) -> None:
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.tube_front_positions[:, 0] < 0, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.tube_front_positions[:, 0] > 1.0, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.tube_front_positions[:, 1] < -1.0, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.tube_front_positions[:, 1] > 1.0, torch.ones_like(self.reset_buf), self.reset_buf)

