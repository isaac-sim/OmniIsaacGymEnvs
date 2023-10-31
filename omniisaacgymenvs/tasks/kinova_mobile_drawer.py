# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch
import omni
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.materials import PhysicsMaterial

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
# from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.kinova_mobile import KinovaMobile
from omniisaacgymenvs.robots.articulations.views.kinova_mobile_view import KinovaMobileView
from omniisaacgymenvs.robots.articulations.views.cabinet_view2 import CabinetView
# from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom
from pxr import Usd, UsdPhysics, UsdShade, UsdGeom, PhysxSchema
from typing import Optional, Sequence, Tuple, Union

from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor = None, q12: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01 (torch.Tensor): Position of frame 1 w.r.t. frame 0.
        q01 (torch.Tensor): Quaternion orientation of frame 1 w.r.t. frame 0.
        t12 (torch.Tensor): Position of frame 2 w.r.t. frame 1.
        q12 (torch.Tensor): Quaternion orientation of frame 2 w.r.t. frame 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the position and orientation of
            frame 2 w.r.t. frame 0.
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02

class KinovaMobileDrawerTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 37
        self._num_actions = 16

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

    def set_up_scene(self, scene) -> None:

        self._usd_context = omni.usd.get_context()
        self.get_kinova()
        self.get_cabinet()
        # if self.num_props > 0:
        #     self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        self._kinovas = KinovaMobileView(prim_paths_expr="/World/envs/.*/kinova", name="kinova_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._kinovas)
        scene.add(self._kinovas._hands)
        scene.add(self._kinovas._lfingers)
        scene.add(self._kinovas._rfingers)
        scene.add(self._cabinets)
        # scene.add(self._cabinets._drawers)

        # if self.num_props > 0:
        #     self._props = RigidPrimView(
        #         prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
        #     )
        #     scene.add(self._props)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("kinova_view"):
            scene.remove_object("kinova_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("cabinet_view"):
            scene.remove_object("cabinet_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        # self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._kinovas = KinovaMobileView(prim_paths_expr="/World/envs/.*/kinova", name="kinova_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._kinovas)
        scene.add(self._kinovas._hands)
        scene.add(self._kinovas._lfingers)
        scene.add(self._kinovas._rfingers)
        scene.add(self._cabinets)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_kinova(self):
        kinova = KinovaMobile(prim_path=self.default_zero_env_path + "/kinova", name="kinova", translation=[2.0, 0, 0.01])
        self._sim_config.apply_articulation_settings(
            "kinova", get_prim_at_path(kinova.prim_path), self._sim_config.parse_actor_config("kinova")
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet", 
                          usd_path="/home/nikepupu/Desktop/Orbit/usd/40147/mobility_relabel_gapartnet_instanceable.usd", 
                          translation=[0,0,0.0], orientation=[0,0,0,1])

        # move cabinet to the ground
        prim_path = self.default_zero_env_path + "/cabinet"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes[0])
        zmin = min_box[2]
        drawer = XFormPrim(prim_path=prim_path)
        position, orientation = drawer.get_world_pose()
        position[2] += -zmin 
        drawer.set_world_pose(position, orientation)

        # add physics material
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.default_zero_env_path + "/cabinet")
        _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        material = PhysicsMaterial(
                prim_path=_physicsMaterialPath,
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        # -- enable patch-friction: yields better results!
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
        physx_material_api.CreateImprovePatchFrictionAttr().Set(True)

        # add collision approximation
        prim = stage.GetPrimAtPath( self.default_zero_env_path + "/cabinet/link_4/collisions_xform")
        collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
        if not collision_api:
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        
        collision_api.CreateApproximationAttr().Set("convexDecomposition")
                          
        self._sim_config.apply_articulation_settings(
            "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        device = self._device
        # hand_pos, hand_rot = self.get_ee_pose()
        
        self.bboxes = torch.zeros(( self._num_envs, 8, 3), device=device)
        link_path =  f"/World/envs/env_0/cabinet/link_4"
        min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
        min_pt = torch.tensor(np.array(min_box)).to(self._device) - self._env_pos[0]
        max_pt = torch.tensor(np.array(max_box)).to(self._device) - self._env_pos[0]
        # self.centers = torch.zeros((self._num_envs, 3)).to(self._device)
        self.centers = ((min_pt +  max_pt)/2.0).repeat((self._num_envs, 1)).to(torch.float)
        

        corners = torch.zeros((8, 3))
        # Top right back
        corners[0] = torch.tensor([max_pt[0], min_pt[1], max_pt[2]])
        # Top right front
        corners[1] = torch.tensor([min_pt[0], min_pt[1], max_pt[2]])
        # Top left front
        corners[2] = torch.tensor([min_pt[0], max_pt[1], max_pt[2]])
        # Top left back (Maximum)
        corners[3] = max_pt
        # Bottom right back
        corners[4] = torch.tensor([max_pt[0], min_pt[1], min_pt[2]])
        # Bottom right front (Minimum)
        corners[5] = min_pt
        # Bottom left front
        corners[6] = torch.tensor([min_pt[0], max_pt[1], min_pt[2]])
        # Bottom left back
        corners[7] = torch.tensor([max_pt[0], max_pt[1], min_pt[2]])
        
        corners = corners.to(self._device)
        for idx in range(self._num_envs):
            self.bboxes[idx] = corners + self._env_pos[idx]

        # stage = get_current_stage()
        # hand_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/kinova/robotiq_85_base_link")),
        #     self._device,
        # )
        # lfinger_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/kinova/left_inner_finger")),
        #     self._device,
        # )
        # rfinger_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/kinova/right_inner_finger")),
        #     self._device,
        # )

        # finger_pose = torch.zeros(7, device=self._device)
        # finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        # finger_pose[3:7] = lfinger_pose[3:7]
        # hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # grasp_pose_axis = 1
        # kinova_local_grasp_pose_rot, kinova_local_pose_pos = tf_combine(
        #     hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        # )
        # kinova_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        # self.kinova_local_grasp_pos = kinova_local_pose_pos.repeat((self._num_envs, 1))
        # self.kinova_local_grasp_rot = kinova_local_grasp_pose_rot.repeat((self._num_envs, 1))

        # drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        # self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        # self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        # self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
        #     (self._num_envs, 1)
        # )
        # self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
        #     (self._num_envs, 1)
        # )
        # self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
        #     (self._num_envs, 1)
        # )
        # self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
        #     (self._num_envs, 1)
        # )

        # self.kinova_default_dof_pos = torch.tensor(
        #     [0 ] * 16, device=self._device
        # )

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)
    
    def get_ee_pose(self):
        hand_position_w, hand_quat_w = self._kinovas._hands.get_world_poses(clone=False)
        hand_position_w = hand_position_w - self._env_pos

        ee_pos_offset = torch.tensor([0.03, 0, 0.14]).repeat((self._num_envs, 1)).to(hand_position_w.device)
        ee_rot_offset = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat((self._num_envs, 1)).to(hand_quat_w.device)
        # print(ee_pos_offset.shape)
        # print(ee_rot_offset.shape)
        position_w, quat_w = combine_frame_transforms(
            hand_position_w, hand_quat_w,  ee_pos_offset, ee_rot_offset
        )
        return position_w, quat_w

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self.get_ee_pose()

        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        
        
        
        # drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        kinova_dof_pos = self._kinovas.get_joint_positions(clone=False)
        kinova_dof_vel = self._kinovas.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        # self.kinova_dof_pos = kinova_dof_pos

        # (
        #     self.kinova_grasp_rot,
        #     self.kinova_grasp_pos,
        #     self.drawer_grasp_rot,
        #     self.drawer_grasp_pos,
        # ) = self.compute_grasp_transforms(
        #     hand_rot,
        #     hand_pos,
        #     self.kinova_local_grasp_rot,
        #     self.kinova_local_grasp_pos,
        #     drawer_rot,
        #     drawer_pos,
        #     self.drawer_local_grasp_rot,
        #     self.drawer_local_grasp_pos,
        # )

        # self.kinova_lfinger_pos, self.kinova_lfinger_rot = self._kinovas._lfingers.get_world_poses(clone=False)
        # self.kinova_rfinger_pos, self.kinova_rfinger_rot = self._kinovas._lfingers.get_world_poses(clone=False)
        hand_pos, hand_rot = self.get_ee_pose()
        tool_pos_diff = (hand_pos - self._env_pos) - self.centers
        dof_pos_scaled = (
            2.0
            * (kinova_dof_pos - self.kinova_dof_lower_limits)
            / (self.kinova_dof_upper_limits - self.kinova_dof_lower_limits)
            - 1.0
        )
        # to_target = self.drawer_grasp_pos - self.kinova_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                kinova_dof_vel * self.dof_vel_scale,
                tool_pos_diff,
                self.cabinet_dof_pos[:, 1].unsqueeze(-1),
                self.cabinet_dof_vel[:, 1].unsqueeze(-1),
            ),
            dim=-1,
        )
        observations = {self._kinovas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        # observations = {self._kinovas.name: {"obs_buf": torch.zeros((self._num_envs, self._num_observations))}}
        # print('obs: ', observations)
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.kinova_dof_targets + self.kinova_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.kinova_dof_targets[:] = tensor_clamp(targets, self.kinova_dof_lower_limits, self.kinova_dof_upper_limits)
        env_ids_int32 = torch.arange(self._kinovas.count, dtype=torch.int32, device=self._device)

        self._kinovas.set_joint_position_targets(self.kinova_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # # reset kinova
        # pos = tensor_clamp(
        #     self.kinova_default_dof_pos.unsqueeze(0)
        #     + 0.25 * (torch.rand((len(env_ids), self.num_kinova_dofs), device=self._device) - 0.5),
        #     self.kinova_dof_lower_limits,
        #     self.kinova_dof_upper_limits,
        # )
        dof_pos = torch.zeros((num_indices, self._kinovas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._kinovas.num_dof), device=self._device)
        # dof_pos[:, :] = pos
        # self.kinova_dof_targets[env_ids, :] = pos
        # self.kinova_dof_pos[env_ids, :] = pos

        # # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=indices
        )

        # # reset props
        # if self.num_props > 0:
        #     self._props.set_world_poses(
        #         self.default_prop_pos[self.prop_indices[env_ids].flatten()],
        #         self.default_prop_rot[self.prop_indices[env_ids].flatten()],
        #         self.prop_indices[env_ids].flatten().to(torch.int32),
        #     )

        self._kinovas.set_joint_position_targets(self.kinova_dof_targets[env_ids], indices=indices)
        self._kinovas.set_joint_positions(dof_pos, indices=indices)
        self._kinovas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_kinova_dofs = self._kinovas.num_dof
        self.kinova_dof_pos = torch.zeros((self.num_envs, self.num_kinova_dofs), device=self._device)
        dof_limits = self._kinovas.get_dof_limits()
        self.kinova_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.kinova_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.kinova_dof_speed_scales = torch.ones_like(self.kinova_dof_lower_limits)
        self.kinova_dof_speed_scales[self._kinovas.gripper_indices] = 0.1
        self.kinova_dof_targets = torch.zeros(
            (self._num_envs, self.num_kinova_dofs), dtype=torch.float, device=self._device
        )

        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        hand_pos, hand_rot = self.get_ee_pose()
        # print(hand_pos)
        # print(self.centers)
        # print('========')
        # exit()
        tool_pos_diff = hand_pos - self.centers
        # print('hand_pos: ', hand_pos)
        # print('self envs: ',  self._env_pos  )
        # print('self centers: ',  self.centers  )
        # print('tool_pos_diff: ', tool_pos_diff)

        tool_pos_diff = torch.norm(tool_pos_diff, dim=-1)
        
        self.rew_buf[:] = -tool_pos_diff.to(torch.float32)

        # print()
        # print('reward: ', self.rew_buf )
        # self.rew_buf[:] = self.compute_kinova_reward(
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.cabinet_dof_pos,
        #     self.kinova_grasp_pos,
        #     self.drawer_grasp_pos,
        #     self.kinova_grasp_rot,
        #     self.drawer_grasp_rot,
        #     self.kinova_lfinger_pos,
        #     self.kinova_rfinger_pos,
        #     self.gripper_forward_axis,
        #     self.drawer_inward_axis,
        #     self.gripper_up_axis,
        #     self.drawer_up_axis,
        #     self._num_envs,
        #     self.dist_reward_scale,
        #     self.rot_reward_scale,
        #     self.around_handle_reward_scale,
        #     self.open_reward_scale,
        #     self.finger_dist_reward_scale,
        #     self.action_penalty_scale,
        #     self.distX_offset,
        #     self._max_episode_length,
        #     self.kinova_dof_pos,
        #     self.finger_close_reward_scale,
        # )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        kinova_local_grasp_rot,
        kinova_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_kinova_rot, global_kinova_pos = tf_combine(
            hand_rot, hand_pos, kinova_local_grasp_rot, kinova_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_kinova_rot, global_kinova_pos, global_drawer_rot, global_drawer_pos

    # def compute_kinova_reward(
    #     self,
    #     reset_buf,
    #     progress_buf,
    #     actions,
    #     cabinet_dof_pos,
    #     kinova_grasp_pos,
    #     drawer_grasp_pos,
    #     kinova_grasp_rot,
    #     drawer_grasp_rot,
    #     kinova_lfinger_pos,
    #     kinova_rfinger_pos,
    #     gripper_forward_axis,
    #     drawer_inward_axis,
    #     gripper_up_axis,
    #     drawer_up_axis,
    #     num_envs,
    #     dist_reward_scale,
    #     rot_reward_scale,
    #     around_handle_reward_scale,
    #     open_reward_scale,
    #     finger_dist_reward_scale,
    #     action_penalty_scale,
    #     distX_offset,
    #     max_episode_length,
    #     joint_positions,
    #     finger_close_reward_scale,
    # ):
    #     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

    #     # distance from hand to the drawer
    #     d = torch.norm(kinova_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    #     dist_reward = 1.0 / (1.0 + d**2)
    #     dist_reward *= dist_reward
    #     dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    #     axis1 = tf_vector(kinova_grasp_rot, gripper_forward_axis)
    #     axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    #     axis3 = tf_vector(kinova_grasp_rot, gripper_up_axis)
    #     axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

    #     dot1 = (
    #         torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    #     )  # alignment of forward axis for gripper
    #     dot2 = (
    #         torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    #     )  # alignment of up axis for gripper
    #     # reward for matching the orientation of the hand to the drawer (fingers wrapped)
    #     rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

    #     # bonus if left finger is above the drawer handle and right below
    #     around_handle_reward = torch.zeros_like(rot_reward)
    #     around_handle_reward = torch.where(
    #         kinova_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #         torch.where(
    #             kinova_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
    #         ),
    #         around_handle_reward,
    #     )
    #     # reward for distance of each finger from the drawer
    #     finger_dist_reward = torch.zeros_like(rot_reward)
    #     lfinger_dist = torch.abs(kinova_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    #     rfinger_dist = torch.abs(kinova_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    #     finger_dist_reward = torch.where(
    #         kinova_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #         torch.where(
    #             kinova_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
    #             (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
    #             finger_dist_reward,
    #         ),
    #         finger_dist_reward,
    #     )

    #     finger_close_reward = torch.zeros_like(rot_reward)
    #     finger_close_reward = torch.where(
    #         d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
    #     )

    #     # regularization on the actions (summed for each environment)
    #     action_penalty = torch.sum(actions**2, dim=-1)

    #     # how far the cabinet has been opened out
    #     open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

    #     rewards = (
    #         dist_reward_scale * dist_reward
    #         + rot_reward_scale * rot_reward
    #         + around_handle_reward_scale * around_handle_reward
    #         + open_reward_scale * open_reward
    #         + finger_dist_reward_scale * finger_dist_reward
    #         - action_penalty_scale * action_penalty
    #         + finger_close_reward * finger_close_reward_scale
    #     )

    #     # bonus for opening drawer properly
    #     rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
    #     rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
    #     rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

    #     # # prevent bad style in opening drawer
    #     # rewards = torch.where(kinova_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #     #                       torch.ones_like(rewards) * -1, rewards)
    #     # rewards = torch.where(kinova_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #     #                       torch.ones_like(rewards) * -1, rewards)

    #     return rewards
