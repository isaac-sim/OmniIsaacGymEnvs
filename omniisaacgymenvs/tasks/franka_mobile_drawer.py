# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
import carb
import numpy as np
import torch
import omni
# from omni.isaac.cloner import Cloner
# from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.materials import PhysicsMaterial
# from omni.isaac.core import World
# from omni.debugdraw import get_debug_draw_interface
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
# from omniisaacgymenvs.robots.articulations.franka import Franka
# from omniisaacgymenvs.robots.articulations.franka_mobile import KinovaMobile
from omniisaacgymenvs.robots.articulations.franka_mobile import FrankaMobile
from omniisaacgymenvs.robots.articulations.views.franka_mobile_view import FrankaMobileView
from omniisaacgymenvs.robots.articulations.views.cabinet_view2 import CabinetView
# from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom
from pxr import Usd, UsdPhysics, UsdShade, UsdGeom, PhysxSchema
from typing import Optional, Sequence, Tuple, Union
from omni.isaac.core.utils.prims import get_all_matching_child_prims, get_prim_children, get_prim_at_path
from numpy.linalg import inv
from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)
# from pytorch3d.transforms import quaternion_to_matrix
from omni.physx.scripts import deformableUtils, physicsUtils
# def quat_axis(q, axis=0):
#     '''
#     :func apply rotation represented by quanternion `q`
#     on basis vector(along axis)
#     :return vector after rotation
#     '''
#     basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
#     basis_vec[:, axis] = 1
#     return quat_rotate(q, basis_vec)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))

def quat_axis(q, axis_idx):
    """Extract a specific axis from a quaternion."""
    rotm = quaternion_to_matrix(q)
    axis = rotm[:, axis_idx]

    return axis

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

class FrankaMobileDrawerTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 39#37 + 3 + 7
        self._num_actions = 12  # 10 + 1

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
        self.get_franka()
        self.get_cabinet()
        # if self.num_props > 0:
        #     self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
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
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
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
        self._frankas = KinovaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_franka(self):
        franka = FrankaMobile(prim_path=self.default_zero_env_path + "/franka", name="franka", translation=[-1.5, 0, 0.15])

        # stage = get_current_stage()
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka")
        # _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka")
        # physicsUtils.add_physics_material_to_prim(
        #             stage,
        #             prim,
        #             _physicsMaterialPath,
        #         )
        
        # prim = stage.GetPrimAtPath(self.default_zero_env_path + "/franka/left_inner_finger_pad")
        # physicsUtils.add_physics_material_to_prim(
        #             stage,
        #             prim,
        #             _physicsMaterialPath,
        #         )
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    def get_cabinet(self):
        # cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet", 
        #                   usd_path="/home/nikepupu/Desktop/Orbit/usd/40147/mobility_relabel_gapartnet_instanceable.usd", 
        #                   translation=[0,0,0.0], orientation=[0,0,0,1])
        
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet", 
                          usd_path="/home/nikepupu/Desktop/Orbit/NewUSD/40147/mobility_relabel_gapartnet.usd", 
                          translation=[0,0,0.0], orientation=[1,0,0,0])

        # move cabinet to the ground
        prim_path = self.default_zero_env_path + "/cabinet"
        bboxes = omni.usd.get_context().compute_path_world_bounding_box(prim_path)
        min_box = np.array(bboxes[0])
        zmin = min_box[2]
        drawer = XFormPrim(prim_path=prim_path)
        position, orientation = drawer.get_world_pose()
        position[2] += -zmin +0.05
        self.cabinet_offset = -zmin + 0.05
        drawer.set_world_pose(position, orientation)

        # add physics material
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.default_zero_env_path + "/cabinet/link_4/collisions")
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

        physicsUtils.add_physics_material_to_prim(
                    stage,
                    prim,
                    _physicsMaterialPath,
                )

        # add collision approximation
        prim = stage.GetPrimAtPath( self.default_zero_env_path + "/cabinet/link_4/collisions")
        collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
        if not collision_api:
            collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        
        collision_api.CreateApproximationAttr().Set("convexDecomposition")

        
                          
        # self._sim_config.apply_articulation_settings(
        #     "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        # )

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
        # enable ccd: /physicsScene
        physicsScenePath = '/physicsScene'
        stage = get_current_stage()
        scene = UsdPhysics.Scene.Get(stage, physicsScenePath)
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)

        

        device = self._device
        # hand_pos, hand_rot = self.get_ee_pose()

        file_to_read = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation/40147/link_anno_gapartnet.json'
        import json
        with open(file_to_read) as json_file:
            data = json.load(json_file)
        
        for d in data:
            if d['link_name'] == 'link_4':
                corners = torch.tensor(d['bbox'])

        
        self.bboxes = torch.zeros(( self._num_envs, 8, 3), device=device)
        link_path =  f"/World/envs/env_0/cabinet/link_4"
        min_box, max_box = omni.usd.get_context().compute_path_world_bounding_box(link_path)
        min_pt = torch.tensor(np.array(min_box)).to(self._device) - self._env_pos[0]
        max_pt = torch.tensor(np.array(max_box)).to(self._device) - self._env_pos[0]
        # self.centers = torch.zeros((self._num_envs, 3)).to(self._device)
        self.centers_orig = ((min_pt +  max_pt)/2.0).repeat((self._num_envs, 1)).to(torch.float).to(self._device) 
        self.centers = self.centers_orig.clone() 

        prim = get_prim_at_path(link_path)
        
        matrix = inv(np.array(omni.usd.get_world_transform_matrix(prim)))
        
        self.forwardDir = matrix[0:3, 2]
        self.forwardDir = self.forwardDir/np.linalg.norm(self.forwardDir)
        self.forwardDir = torch.tensor(self.forwardDir).to(self._device).repeat((self._num_envs,1))

        # corners = torch.zeros((8, 3))
        # # Top right back
        # corners[0] = torch.tensor([max_pt[0], min_pt[1], max_pt[2]])
        # # Top right front
        # corners[1] = torch.tensor([min_pt[0], min_pt[1], max_pt[2]])
        # # Top left front
        # corners[2] = torch.tensor([min_pt[0], max_pt[1], max_pt[2]])
        # # Top left back (Maximum)
        # corners[3] = max_pt
        # # Bottom right back
        # corners[4] = torch.tensor([max_pt[0], min_pt[1], min_pt[2]])
        # # Bottom right front (Minimum)
        # corners[5] = min_pt
        # # Bottom left front
        # corners[6] = torch.tensor([min_pt[0], max_pt[1], min_pt[2]])
        # # Bottom left back
        # corners[7] = torch.tensor([max_pt[0], max_pt[1], min_pt[2]])

        
        corners = corners.to(self._device)
        self.handle_short = torch.zeros((self._num_envs, 3))
        self.handle_out = torch.zeros((self._num_envs, 3))
        self.handle_long = torch.zeros((self._num_envs, 3))

        for idx in range(self._num_envs):
            # self.bboxes[idx] = corners + self._env_pos[idx]
            # handle_short = corners[0] - corners[4]
            # handle_out = corners[1] - corners[0]
            # handle_long = corners[3] - corners[0]

            
            # self.handle_short[idx] = handle_short
            # self.handle_out[idx] = handle_out
            # self.handle_long[idx] = handle_long

            handle_out = corners[0] - corners[4]
            handle_long = corners[1] - corners[0]
            handle_short = corners[3] - corners[0]




            self.handle_short[idx] = handle_short
            self.handle_out[idx] = handle_out
            self.handle_long[idx] = handle_long
        
        self.handle_short = self.handle_short.cuda()
        self.handle_out = self.handle_out.cuda()
        self.handle_long = self.handle_long.cuda()
        
        self.corners = corners.unsqueeze(0).repeat((self._num_envs, 1,1))
        for idx in range(self._num_envs):
            self.corners[idx] = self.corners[idx] + self._env_pos[idx] + torch.tensor([0,0, self.cabinet_offset]).to(self._device)


        # stage = get_current_stage()
        # hand_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/robotiq_85_base_link")),
        #     self._device,
        # )
        # lfinger_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/left_inner_finger")),
        #     self._device,
        # )
        # rfinger_pose = get_env_local_pose(
        #     self._env_pos[0],
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/right_inner_finger")),
        #     self._device,
        # )

        # finger_pose = torch.zeros(7, device=self._device)
        # finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        # finger_pose[3:7] = lfinger_pose[3:7]
        # hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # grasp_pose_axis = 1
        # franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
        #     hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        # )
        # franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        # self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        # self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

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

        # self.franka_default_dof_pos = torch.tensor(
        #     [0 ] * 16, device=self._device
        # )

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)
    
    def get_ee_pose(self):
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)
        hand_position_w = hand_position_w - self._env_pos

        ee_pos_offset = torch.tensor([0.0, 0.0, 0.105]).repeat((self._num_envs, 1)).to(hand_position_w.device)
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
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        # self.franka_dof_pos = franka_dof_pos

        # (
        #     self.franka_grasp_rot,
        #     self.franka_grasp_pos,
        #     self.drawer_grasp_rot,
        #     self.drawer_grasp_pos,
        # ) = self.compute_grasp_transforms(
        #     hand_rot,
        #     hand_pos,
        #     self.franka_local_grasp_rot,
        #     self.franka_local_grasp_pos,
        #     drawer_rot,
        #     drawer_pos,
        #     self.drawer_local_grasp_rot,
        #     self.drawer_local_grasp_pos,
        # )

        # self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        # self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        hand_pos, hand_rot = self.get_ee_pose()
        tool_pos_diff = hand_pos  - self.centers
        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        # to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                tool_pos_diff,
                hand_pos,
                hand_rot,
                self.centers,
                self.cabinet_dof_pos[:, 1].unsqueeze(-1),
                self.cabinet_dof_vel[:, 1].unsqueeze(-1),
            ),
            dim=-1,
        )
        observations = {self._frankas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        # observations = {self._frankas.name: {"obs_buf": torch.zeros((self._num_envs, self._num_observations))}}
        # print('obs: ', observations)
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # self.actions = torch.zeros((self._num_envs, self._num_actions+5), device=self._device)
        # import pdb; pdb.set_trace()
        # self.actions[:, :10] = (actions.clone()[:,:10]).to(self._device)

        # self.actions = (actions.clone()).to(self._device)
        # targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        # map -1 to 1 to 0 to 1
        self.actions = actions.clone().to(self._device)
        
        
        

        self.actions = (self.actions + 1.0)/2.0

        # mask = (actions[:, -1] > 0).unsqueeze(-1).float()
        # self.actions[:,-6:] = 1.0 * mask.expand_as(self.actions[:, -6:])

        targets = self.actions *(self.franka_dof_upper_limits - self.franka_dof_lower_limits) + self.franka_dof_lower_limits

        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        # self.franka_dof_targets[:,:3] = 0.0

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # print(self.franka_dof_targets)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # # reset franka
        # pos = tensor_clamp(
        #     self.franka_default_dof_pos.unsqueeze(0)
        #     + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
        #     self.franka_dof_lower_limits,
        #     self.franka_dof_upper_limits,
        # )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        # dof_pos[:, :] = pos
        # self.franka_dof_targets[env_ids, :] = pos
        # self.franka_dof_pos[env_ids, :] = pos

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

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

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

        cabinet_dof_limits = self._cabinets.get_dof_limits()
        self.cabinet_dof_lower_limits = cabinet_dof_limits[0, 1, 0].to(device=self._device)
        self.cabinet_dof_upper_limits = cabinet_dof_limits[0, 1, 1].to(device=self._device)

        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # import pdb; pdb.set_trace()
        # self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.centers = (self.centers_orig +  self.forwardDir * self.cabinet_dof_pos[:, 1].unsqueeze(-1)).to(torch.float32).to(self._device)
      
        # world = World()
        
        # while True:
        #     self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        #     color = 4283782485
        #     my_debugDraw = get_debug_draw_interface()
        #     corners = self.corners.clone()
        #     for idx in range(self._num_envs):
        #     # import pdb; pdb.set_trace()
        #         corners[idx] = (self.corners[idx] + self.forwardDir[idx] * self.cabinet_dof_pos[idx, 1].unsqueeze(-1)).to(torch.float32).to(self._device)
        #     corners = corners.cpu().numpy()
        #     for corner in corners:
        #         my_debugDraw.draw_line(carb.Float3(corner[0]),color, carb.Float3(corner[4]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[1]),color, carb.Float3(corner[0]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[3]),color, carb.Float3(corner[0]), color)
            
        #     world.step(render=True)
        # handle_out = corners[0] - corners[4]
        # handle_long = corners[1] - corners[0]
        # handle_short = corners[3] - corners[0]

        
        # print(self.centers)
        hand_pos, hand_rot = self.get_ee_pose()
        # print(hand_pos)
        # print(self.centers)
        # print('========')
        # exit()
        tcp_to_obj_delta = hand_pos - self.centers
        # print('hand_pos: ', hand_pos)
        # print('self envs: ',  self._env_pos  )
        # print('self centers: ',  self.centers  )
        # print('tool_pos_diff: ', tool_pos_diff)

        tcp_to_obj_dist = torch.norm(tcp_to_obj_delta, dim=-1)

        handle_out_length = torch.norm(self.handle_out, dim = -1).cuda()
        handle_long_length = torch.norm(self.handle_long, dim = -1).cuda()
        handle_short_length = torch.norm(self.handle_short, dim = -1).cuda()

        handle_out = self.handle_out / handle_out_length.unsqueeze(-1).cuda()
        handle_long = self.handle_long / handle_long_length.unsqueeze(-1).cuda()
        handle_short = self.handle_short / handle_short_length.unsqueeze(-1).cuda()


        self.franka_lfinger_pos = self._frankas._lfingers.get_world_poses(clone=False)[0] - self._env_pos
        self.franka_rfinger_pos = self._frankas._rfingers.get_world_poses(clone=False)[0] - self._env_pos
        
        gripper_length = torch.norm(self.franka_lfinger_pos - self.franka_rfinger_pos, dim=-1)

        # print(self.franka_lfinger_pos)
        # print(self.franka_rfinger_pos)
        
       
        short_ltip = ((self.franka_lfinger_pos - self.centers) * handle_short).sum(dim=-1) 
        short_rtip = ((self.franka_rfinger_pos - self.centers) *handle_short).sum(dim=-1)
        is_reached_short = (short_ltip * short_rtip) < 0

        is_reached_long = (tcp_to_obj_delta * handle_long).sum(dim=-1).abs() < (handle_long_length / 2.0 - 0.005)
        is_reached_out = (tcp_to_obj_delta * handle_out).sum(dim=-1).abs() < (handle_out_length / 2.0 - 0.005)


        hand_grip_dir = quat_axis(hand_rot, 1).cuda()
        # hand_grip_dir_length = torch.norm(hand_grip_dir)
        # hand_grip_dir  = hand_grip_dir/ hand_grip_dir_length
        
        hand_sep_dir = quat_axis(hand_rot, 0).cuda()
        # hand_sep_dir_length = torch.norm(hand_sep_dir)
        # hand_sep_dir = hand_sep_dir / hand_sep_dir_length

        hand_down_dir = quat_axis(hand_rot, 2).cuda()
        # hand_down_dir_length = torch.norm(hand_down_dir)
        # hand_down_dir = hand_down_dir / hand_down_dir_length

        # dot1 = (-hand_grip_dir * handle_out).sum(dim=-1)
        dot1 = torch.max((hand_grip_dir * handle_out).sum(dim=-1), (-hand_grip_dir * handle_out).sum(dim=-1))
        dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
        dot2 = (-hand_sep_dir * handle_short).sum(dim=-1)
        dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))
        # dot3 = (hand_down_dir * handle_long).sum(dim=-1)

        rot_reward = dot1 + dot2 + dot3 - 3     
        reaching_reward = - tcp_to_obj_dist +  0.1 * (is_reached_short + is_reached_long + is_reached_out) 

        is_reached =  is_reached_out & is_reached_long & is_reached_short #& (tcp_to_obj_dist < 0.03) 

        # close_reward = is_reached * (gripper_length < 0.02) * 0.1 + 0.1 * (gripper_length > 0.08) * (~is_reached)
        close_reward =  (0.1 - gripper_length ) * is_reached + 0.1 * ( gripper_length -0.1) * (~is_reached)
        # print('close reward: ', close_reward)

        grasp_success = is_reached & (gripper_length < handle_short_length + 0.015) & (rot_reward > -0.2)

        # if torch.any(grasp_success):
        #     print('grasp success')
        #     num_true = grasp_success.sum()
        #     if num_true >= 3:
        #         print(torch.nonzero(grasp_success).squeeze())
        #         print()
        #         timeline = omni.timeline.get_timeline_interface()
        #         timeline.pause()
        #         from omni.isaac.core import World
        #         world = World()
        #         while True:
        #             world.step(render=True)


        normalized_dof_pos = (self.cabinet_dof_pos[:, 1] - self.cabinet_dof_lower_limits) / (self.cabinet_dof_upper_limits - self.cabinet_dof_lower_limits)
        condition_mask = (normalized_dof_pos > 0.95) & grasp_success
        

        self.rew_buf[:] = reaching_reward +  rot_reward * 0.5 + 5 * close_reward + grasp_success *5 * ( 0.2 + normalized_dof_pos)  

        self.rew_buf = self.rew_buf + self.rew_buf.abs() * rot_reward

        # self.rew_buf[condition_mask] += 2.0
        # self.rew_buf[:] = rot_reward
        # self.rew_buf[:] = rot_reward
        self.rew_buf[:] = self.rew_buf[:].to(torch.float32)
        # print()
        # print('reward: ', self.rew_buf )
        # self.rew_buf[:] = self.compute_franka_reward(
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.cabinet_dof_pos,
        #     self.franka_grasp_pos,
        #     self.drawer_grasp_pos,
        #     self.franka_grasp_rot,
        #     self.drawer_grasp_rot,
        #     self.franka_lfinger_pos,
        #     self.franka_rfinger_pos,
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
        #     self.franka_dof_pos,
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
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos

    # def compute_franka_reward(
    #     self,
    #     reset_buf,
    #     progress_buf,
    #     actions,
    #     cabinet_dof_pos,
    #     franka_grasp_pos,
    #     drawer_grasp_pos,
    #     franka_grasp_rot,
    #     drawer_grasp_rot,
    #     franka_lfinger_pos,
    #     franka_rfinger_pos,
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
    #     d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    #     dist_reward = 1.0 / (1.0 + d**2)
    #     dist_reward *= dist_reward
    #     dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    #     axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
    #     axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
    #     axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
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
    #         franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #         torch.where(
    #             franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
    #         ),
    #         around_handle_reward,
    #     )
    #     # reward for distance of each finger from the drawer
    #     finger_dist_reward = torch.zeros_like(rot_reward)
    #     lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    #     rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
    #     finger_dist_reward = torch.where(
    #         franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
    #         torch.where(
    #             franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
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
    #     # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #     #                       torch.ones_like(rewards) * -1, rewards)
    #     # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
    #     #                       torch.ones_like(rewards) * -1, rewards)

    #     return rewards
