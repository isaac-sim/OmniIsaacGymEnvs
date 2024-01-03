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
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core import World
# from omni.debugdraw import get_debug_draw_interface
from omniisaacgymenvs.tasks.base.rl_multi_task import RLMultiTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.views.cabinet_view2 import CabinetView
# from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.franka_mobile import FrankaMobile
from omniisaacgymenvs.robots.articulations.views.franka_mobile_view import FrankaMobileView
# from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
# from pxr import Usd, UsdGeom
from pxr import Usd, UsdPhysics, UsdShade, UsdGeom, PhysxSchema
from typing import Optional, Sequence, Tuple, Union
from omni.isaac.core.utils.prims import get_all_matching_child_prims, get_prim_children, get_prim_at_path, delete_prim
from numpy.linalg import inv
from omni.isaac.core.utils.torch.rotations import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    quat_to_rot_matrices,
    matrices_to_euler_angles,
    quat_from_euler_xyz

)
from typing import List, Type
# from pytorch3d.transforms import quaternion_to_matrix
from omni.physx.scripts import deformableUtils, physicsUtils
import os
import json
from scipy.spatial.transform import Rotation as R
import pxr

def load_annotation_file():
    folder = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation'
    subfolders = sorted(os.listdir(folder))
    
    # filter out files starts with 4 and has 5 digits
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    annotation_json = 'link_anno_gapartnet.json'
    annotation = {}
    for subfolder in subfolders:
        annotation_path = os.path.join(folder, subfolder, annotation_json)
        with open(annotation_path, 'r') as f:
            annotation[int(subfolder)] = json.load(f)
    return annotation

def load_joint_file():
    folder = '/home/nikepupu/Desktop/gapartnet_new_subdivition/partnet_all_annotated_new/annotation'
    subfolders = sorted(os.listdir(folder))
    # filter out files starts with 4 and has 5 digits
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    annotation_json = 'mobility_v2.json'
    annotation = {}
    for subfolder in subfolders:
        annotation_path = os.path.join(folder, subfolder, annotation_json)
        with open(annotation_path, 'r') as f:
            annotation[int(subfolder)] = json.load(f)
    return annotation

def load_usd_paths():
    folder = '/home/nikepupu/Desktop/Orbit/NewUSD'
    subfolders = os.listdir(folder)
    subfolders = [f for f in subfolders if f.startswith('4') and len(f) == 5]
    usds = [os.path.join(folder, f, 'mobility_relabel_gapartnet.usd') for f in subfolders]
    return usds



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

def get_cabinet_position(index, spacing_x, spacing_y, cabinets_per_row):
    x = index % cabinets_per_row
    y = index // cabinets_per_row
    return x * spacing_x, y * spacing_y

class FrankaMobileMultiTask(RLMultiTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 39 + 1 #37 + 3 + 7
        self._num_actions = 12 + 1 # 10 + 1

        RLMultiTask.__init__(self, name, env)

        self.annotations = load_annotation_file()
        self.usds = load_usd_paths()[:]
        self.mobility = load_joint_file()
        self.bbox = []
        self.bbox_object = []

        self.allForwardDirs = []

        self.allJointNames = []

        self.allJointIndices = []

        self.environment_positions = []
        self.all_cabinets = []

        self.cabinet_infos = []
        self.bbox_links = []

        self.original_upper_limit = []
   
        self.success_buf = torch.zeros((self._num_envs, 1), device=self._device)

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

    def reset_scene(self, scene):
        self.cabinet_scale = 0.5
        self.cabinet_orientation = torch.tensor([ 1.0, 0, 0, 0]).to(torch.float32)
        self._usd_context = omni.usd.get_context()
    
        cnt = 0
        
        for idx,  usd in enumerate(self.usds):
            if cnt >= 20:
                break
            print('setup: ', usd)

            # if idx != 2:
                # continue
            # 47570, 47565
            # if int(usd.split('/')[-2]) in [46130, 46893, 45235, 44853, 47570, 47565, 45238 ]:
            #     continue
            for _ in range(1):
                anno_id = int(usd.split('/')[-2])
                found_handle = False
                for anno in self.annotations[anno_id]:
                    if 'handle' in anno['category'] :
                        found_handle = True
                        break
            
                if found_handle: 
                    for anno in self.annotations[anno_id]:
                        if 'slider_drawer' in anno['category']:

                            status = self.get_assets(usd, env_id = cnt, annotation = self.annotations[anno_id], anno=anno, link_name = anno['link_name'])
                            
                            if status:
                                self.cabinet_infos.append( (self.annotations[anno_id], anno, anno['link_name']) )
                                cnt += 1

                            if cnt >= 20:
                                break
                                # if cnt == 258 or cnt == 259:
                                #     print('==========', usd)
                                    
        # print('cnt; ', cnt)
        # exit()
        self.bbox = torch.tensor(self.bbox)
        self.bbox_object = torch.tensor(self.bbox_object)
        self.bbox_orig = self.bbox.clone()
        # print(self.bbox)
        self.handle_out =   self.bbox[:,0] -  self.bbox[:,4]
        self.handle_long =  self.bbox[:,1] -  self.bbox[:,0]
        self.handle_short = self.bbox[:,3] -  self.bbox[:,0]

        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = []
        self.cnt = cnt
        for i in range(cnt):
            _cabinet = CabinetView(prim_paths_expr=f"/World/envs/env_{i}/cabinet", name=f"cabinet_view_{i}")
            
            # self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")
            self._cabinets.append(_cabinet)
            scene.add(_cabinet)

        self.allForwardDirs = torch.stack(self.allForwardDirs, dim=0).to(self._device)
        self.environment_positions = torch.tensor(self.environment_positions).to(self._device)
        # print('env pos: ', self.environment_positions.shape)
        # exit()

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)

        self.init_data()
     
    def get_assets(self, usd_path, env_id = 0, annotation = None, anno=None, link_name = ""):
            env_path = f"{self.default_base_env_path}/env_{env_id}"
            stage = get_current_stage()
            bbox_link = None
            
            def find_prims_by_type(prim_path, prim_type: Type[Usd.Typed]) -> List[Usd.Prim]:
                children = get_all_matching_child_prims(prim_path)
                found_prims = [ x for x in children if x.IsA(prim_type) and 'collisions' in x.GetPath().pathString ]
                return found_prims
          
           
            
            x, y = get_cabinet_position(env_id, 5, 5, 10)
            
            # print('x, y: ', x, y)
            cabinet = Cabinet(env_path + "/cabinet", name="cabinet", 
                            usd_path=usd_path, 
                            translation=[x,y, 0.0], orientation=[1,0,0,0], scale=[self.cabinet_scale, self.cabinet_scale, self.cabinet_scale])

            # move cabinet to the ground
            cabinet_prim_path = env_path + "/cabinet"
            bboxes = omni.usd.get_context().compute_path_world_bounding_box(cabinet_prim_path)
            min_box = np.array(bboxes[0])
            zmin = min_box[2]
            cabinet_offset = -zmin
            cabinet = Cabinet(env_path + "/cabinet", name="cabinet", 
                            usd_path=usd_path, 
                            translation=[x,y, cabinet_offset ], orientation=self.cabinet_orientation, scale=[self.cabinet_scale, self.cabinet_scale, self.cabinet_scale])
          

            

            self.all_cabinets.append(cabinet)
            
            children = get_all_matching_child_prims(env_path + "/cabinet")
            # print('children: ', children)
            prims: List[Usd.Prim] = [x for x in children if x.IsA(UsdPhysics.Joint)] 
            # print(prims)
            
            for joint in prims:
                body1 = joint.GetRelationship("physics:body1").GetTargets()[0]
                if bbox_link is None and len(joint.GetRelationship("physics:body0").GetTargets()) > 0:
                    body0 = joint.GetRelationship("physics:body0").GetTargets()[0]
                    if (body0.pathString).endswith(link_name):
                        bbox_link = body1

            if not bbox_link:

                # delete
                delete_prim(env_path + "/cabinet")
                return False
            
            upper_limits = []
            for prim in prims:
                joint = pxr.UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())	
                if joint:
                    upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
                    upper_limits.append(upper_limit)
                    # print(prim.GetPath(), "upper_limit", upper_limit)
                    mobility_prim = prim.GetParent().GetParent()
                    mobility_xform = pxr.UsdGeom.Xformable.Get(stage, mobility_prim.GetPath())
                    scale_factor = mobility_xform.GetOrderedXformOps()[2].Get()[0]
                    # print("scale_factor", scale_factor)
                    joint.CreateUpperLimitAttr(upper_limit * scale_factor)
            
            self.original_upper_limit.append(upper_limits)
            
            for joint in prims:
                
                body1 = joint.GetRelationship("physics:body1").GetTargets()[0]
                if (body1.pathString).endswith(link_name):
                    self.allJointNames.append(joint.GetPath().pathString)
         
            self.environment_positions.append([x, y, 0])

            franka = FrankaMobile(prim_path=env_path + "/franka", name="franka", translation=[-1.50+x, 0+y, 0.0])
            # add physics material
           
            # bbox_link_full = bbox_link
            bbox_link = bbox_link.pathString.split('/')[-1]
            self.bbox_links.append(bbox_link)
           
            prims: List[Usd.Prim] = find_prims_by_type(cabinet_prim_path, UsdGeom.Mesh)
            # self._sim_config.apply_articulation_settings(
            #     "cabinet", get_prim_at_path(cabinet_prim_path), self._sim_config.parse_actor_config("cabinet")
            # )

            # self._sim_config.apply_articulation_settings(
            #     "franka", get_prim_at_path(env_path + "/franka"), self._sim_config.parse_actor_config("franka")
            # )



            
            
            for anno in annotation:
                # print('bbox_link: ', bbox_link)
                # print(anno['link_name'])
                if anno['link_name'] == bbox_link:
                    bbox_to_use = anno['bbox']
                    break
            # print(bbox_to_use)
            # exit()
            bbox_to_use = self.cabinet_scale * np.array(bbox_to_use)
            self.bbox_object.append(bbox_to_use)
            
            bbox = np.array(bbox_to_use) + np.array([x, y, cabinet_offset ])
            self.bbox.append(bbox)
            # jointData['axis']['origin'] = np.array(jointData['axis']['origin']) + np.array([x, y, cabinet_offset ])
        
            prim = stage.GetPrimAtPath(env_path + f"/cabinet/{bbox_link}")
            matrix = inv(np.array(omni.usd.get_world_transform_matrix(prim)))
        
            forwardDir = matrix[0:3, 2]
            forwardDir = forwardDir/np.linalg.norm(forwardDir)
            forwardDir = torch.tensor(forwardDir).to(self._device)
            self.allForwardDirs.append(forwardDir)
            

            # print('cabinet: ', env_path)
            # for anno in annotation:
                # if 'handle' in anno['category'] :
                #     link_name = anno['link_name']
            prim = stage.GetPrimAtPath( env_path + f"/cabinet/{bbox_link}/collisions")
            # print('set collision for: ', prim)
            collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())


            if not collision_api:
                collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            
            collision_api.CreateApproximationAttr().Set("convexDecomposition")
            # mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            # mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
        
            prim = stage.GetPrimAtPath(env_path + f"/cabinet/{bbox_link}/collisions")
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
            subprims = get_all_matching_child_prims(env_path + f"/cabinet/{bbox_link}/collisions")
            for prim in subprims:
                physicsUtils.add_physics_material_to_prim(
                        stage,
                        prim,
                        _physicsMaterialPath,
                    )
            
            for prim in prims:
                if env_path + f"/cabinet/{bbox_link}/collisions" == prim.GetPath().pathString:
                    continue
                if  'franka' in prim.GetPath().pathString or prim.GetPath().pathString.endswith('collisions'):
                    continue
                # print(prim)
                collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
                if not collision_api:
                    collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                
                collision_api.CreateApproximationAttr().Set("boundingCube")

            return True
    
    def set_up_scene(self, scene) -> None:

      
       
       
        self.reset_scene(scene)


        super().set_up_scene(scene, filter_collisions=False)

       
        

        
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
        self._frankas = FrankaMobileView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        
        scene.add(self._cabinets)

        self.init_data()

   

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

        self.centers_orig = ((self.bbox[:, 0] + self.bbox[:, 6]) / 2).to(self._device)
        self.centers_obj = ((self.bbox_object[:, 0] +  self.bbox_object[: ,6] )/2.0).to(torch.float).to(self._device)

        device = self._device
        

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)
    
    def get_ee_pose(self):
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)
        return hand_position_w, hand_quat_w

    def get_ee_pose_o(self):
        def matrix_to_6d(rot_matrices):
            """
            Convert a batch of rotation matrices to their 6D representation.
            
            Parameters:
            rot_matrices (torch.Tensor): a batch of 3x3 rotation matrices of shape (batch_size, 3, 3).
            
            Returns:
            torch.Tensor: a batch of 6D representations of shape (batch_size, 6).
            """
            # Ensure the input is a proper batch of matrices
            assert rot_matrices.ndim == 3 and rot_matrices.shape[1:] == (3, 3), "Input must be a batch of 3x3 matrices."
            
            # Slice out the first two columns across all matrices in the batch
            return rot_matrices[:, :, :2].reshape(-1, 6)
        
        hand_position_w, hand_quat_w = self._frankas._hands.get_world_poses(clone=True)

        cabinet_world_poses = []
        for cabinet in self._cabinets:
            cabinet_world_poses.append(cabinet.get_world_poses(clone=True)[0])
        
        cabinet_world_poses = torch.stack(cabinet_world_poses, dim=0).squeeze(1).unsqueeze(-1)
        # print('cabinet_world poses: ', cabinet_world_poses.shape)
        position_w = hand_position_w - cabinet_world_poses.squeeze(-1)
        num_objects= len(cabinet_world_poses)
        
        rotation_matrix = quaternion_to_matrix( torch.tensor(self.cabinet_orientation).float() ).to(self._device)
        # print(rotation_matrix.shape)
        all_orientations = rotation_matrix.repeat((num_objects, 1,1)).to(torch.float).to(self._device)
       
        # print(all_orientations.shape)
        # print(position_w.shape)
        # exit()
        # position_w =  torch.matmul(all_orientations.mT, position_w.T).T
        position_w =  all_orientations.mT @ position_w.unsqueeze(-1)
        
        gripper_rot = quat_to_rot_matrices(hand_quat_w)
        hand_rot_w_object_space =  all_orientations.mT @ gripper_rot
        # hand_rot_w_object_space = matrix_to_6d(hand_rot_w_object_space)
        # hand_rot_w_object_space = hand_rot_w_object_space[:, :, :3].reshape(-1, 9)
        # print
        
        tmp  = matrices_to_euler_angles(hand_rot_w_object_space)
        
        roll, pitch, yaw = tmp[:, 0], tmp[:, 1], tmp[:, 2]
        # print(roll.shape)
        # print(pitch.shape)
        # print(yaw.shape)

        hand_rot_w_object_space =  quat_from_euler_xyz(roll, pitch, yaw)
        return position_w, hand_rot_w_object_space
    
    def get_root_pose(self):
        root_position_w, root_quat_w = self._frankas.get_world_poses(clone=True)
        root_position_w = root_position_w
        return root_position_w, root_quat_w
    

    def get_observations(self) -> dict:        
        # drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)

        cabinet_dof_pos = []
        cabinet_dof_vel = []
        for dof_index, cabinet in zip(self.allJointIndices, self._cabinets):
            cabinet_dof_pos.append(cabinet.get_joint_positions(clone=False)[:, dof_index])
            cabinet_dof_vel.append(cabinet.get_joint_velocities(clone=False)[:, dof_index])
        
        self.cabinet_dof_pos = torch.stack(cabinet_dof_pos, dim=1)
        self.cabinet_dof_vel = torch.stack(cabinet_dof_vel, dim=1)
        

        scale_tensor_t = self.cabinet_dof_pos.transpose(0, 1)
        # forwardDirs = torch.tensor([1, 0, 0]).to(self._device).repeat((self._num_envs, 1))
        num_envs = len(self._cabinets)
        forwardDirs = [torch.tensor([1, 0, 0]) for _ in range(num_envs)]
        forwardDirs = torch.stack(forwardDirs, dim=0).to(self._device)
        position_diff = torch.mul(forwardDirs, scale_tensor_t)
        centers = self.centers_obj.to(self._device)  +  position_diff 
        # centers = self.centers_orig.to(self._device)  +  position_diff
        

        hand_pos, hand_rot = self.get_ee_pose_o()
        hand_pos = hand_pos.squeeze(-1)
        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )

        
        # print(self.centers_obj.shape)
        # centers = (self.centers_obj.to(self._device) +  forwardDir * self.cabinet_dof_pos[:, 3].unsqueeze(-1)).to(torch.float32).to(self._device)
        # print('centers: ', centers.shape)
        # print('hand_pos: ', hand_pos.shape)
        # exit()
        tool_pos_diff = hand_pos  - centers

        corners = self.bbox_object.to(self._device) + position_diff.unsqueeze(1)
        handle_out = corners[:, 0] - corners[:, 4]
        handle_long = corners[:, 1] - corners[:, 0]
        handle_short = corners[:, 3] - corners[:, 0]

        normalized_dof_pos = (self.cabinet_dof_pos - self.cabinet_dof_lower_limits) / (self.cabinet_dof_upper_limits - self.cabinet_dof_lower_limits)
        # normalized_dof_pos *= 10
       
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,  # 12
                franka_dof_vel * self.dof_vel_scale, #12
                tool_pos_diff, # 3
                hand_pos, # 3
                hand_rot, # 4
                centers, # 3
                # handle_out.reshape(-1, 3),
                # handle_long.reshape(-1, 3),
                # handle_short.reshape(-1, 3),
                self.cabinet_dof_pos.transpose(0, 1),
                normalized_dof_pos.transpose(0, 1), # 1
                self.cabinet_dof_vel.transpose(0, 1), # 1 
            ),
            dim=-1,
        )
        
        self.obs_buf = self.obs_buf.to(torch.float32)
        observations = {self._frankas.name: {"obs_buf": self.obs_buf.to(torch.float32)}}
        

        
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
        mode_prob = (self.actions[:, 0]  + 1.0 )/2
        # sample 0 or 1 based on mode_prob
        # mode = torch.bernoulli(mode_prob).to(torch.int32)

        mode = (mode_prob > 0.5).long()
        base_indices = torch.nonzero(mode).long()
        arm_indices = torch.nonzero(1 - mode).long()


        # mode = self.actions[:, 0] <= 0
        # base_indices =  torch.nonzero(mode).long()

     
        # mode = self.actions[:, 0] > 0
        # arm_indices =  torch.nonzero(mode).long()

   
        
        self.actions[:, 1:] = (self.actions[:, 1:] + 1.0) / 2.0
        current_joint_positons = self._frankas.get_joint_positions(clone=False)
        base_positions = current_joint_positons[:, :3]
        arm_positions = current_joint_positons[:, 3:]

        # print(base_positions.shape)
        # print(arm_positions.shape)
        # print(self.franka_dof_targets.shape)
        # exit()

        targets = self.actions[:, 1:] *(self.franka_dof_upper_limits - self.franka_dof_lower_limits) + self.franka_dof_lower_limits

        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        if len(base_indices) > 0:
            self.franka_dof_targets[base_indices, :3 ] =  base_positions[base_indices]
        if len(arm_indices) > 0:
            self.franka_dof_targets[arm_indices, 3:] =  arm_positions[arm_indices]
        

        
        # self.franka_dof_targets[:,:3] = 0.0

        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        # print(self.franka_dof_targets)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def rotate_points_around_z(self, points, center, angle_degrees):
        """
        Rotate a set of points around the z-axis by a given angle around a center point.

        :param points: np.array, array of points to rotate.
        :param center: np.array, the center point around which to rotate.
        :param angle_degrees: float, rotation angle in degrees.
        :return: np.array, array of rotated points.
        """
        # Convert the angle to radians
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix for z-axis rotation
        cos_angle, sin_angle = np.cos(angle_radians), np.sin(angle_radians)
        rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                    [sin_angle,  cos_angle, 0],
                                    [0,          0,         1]])

        # Translate points to origin (center point becomes the origin)
        center = center.copy()
        center[2] = 0
        translated_points = points - center

        # Apply the rotation
        rotated_points = np.dot(translated_points, rotation_matrix.T)

        # Translate points back to original center
        rotated_points += center

        return rotated_points
    
    def reset_scale(self):
        import random
        self.bbox = []
        self.bbox_object = []
        
        stage = get_current_stage()
        
        # random_scale = random.uniform(0.4, 0.8)
        # self.cabinet_scale = random_scale
        for cabinet, cabinet_info, bbox_link, positions, upper_limits in zip(self._cabinets, 
                                                               self.cabinet_infos, self.bbox_links,
                                                                 self.environment_positions, self.original_upper_limit):
            random_scale = random.uniform(0.4, 0.8)
            cabinet.set_local_scales([[random_scale, random_scale, random_scale]])
            cabinet_prim_path = cabinet._prim_paths[0]
            position, orientation = cabinet.get_world_poses()
            
            # position[0][2] = 0
            # cabinet.set_world_poses(position, orientation)

            world = World()
            for _ in range(12):
                world.step(render=True)
            
            bboxes = omni.usd.get_context().compute_path_world_bounding_box(cabinet_prim_path)
            # cabinet_prim = stage.GetPrimAtPath(cabinet_prim_path)
            # cabinet_xform = UsdGeom.Xformable(cabinet_prim)
            # bboxes = cabinet_xform.ComputeWorldBound(0, UsdGeom.Tokens.default_)
            # bboxes = np.array([bboxes.ComputeAlignedRange().GetMin(), bboxes.ComputeAlignedRange().GetMax()])

            min_box = np.array(bboxes[0])
            zmin = min_box[2]
            cabinet_offset = -zmin + 0.01
            position, orientation = cabinet.get_world_poses()
            
            position[0][2] = position[0][2]  + cabinet_offset
            cabinet.set_world_poses(position, orientation)
            children = get_all_matching_child_prims(cabinet_prim_path)
            # print('children: ', children)
            prims: List[Usd.Prim] = [x for x in children if x.IsA(UsdPhysics.Joint)] 
            for prim, upper_limit in zip(prims, upper_limits):
                joint = pxr.UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())	
                if joint:
                    # upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
                    # print(prim.GetPath(), "upper_limit", upper_limit)
                    mobility_prim = prim.GetParent().GetParent()
                    mobility_xform = pxr.UsdGeom.Xformable.Get(stage, mobility_prim.GetPath())
                    # print(mobility_xform.GetOrderedXformOps()[2].Get())
                    scale_factor = mobility_xform.GetOrderedXformOps()[2].Get()[0]
                    # print("scale_factor", scale_factor)
                    joint.CreateUpperLimitAttr(upper_limit * scale_factor)

            annotation, _, _ = cabinet_info
          
            for anno in annotation:
                # print('bbox_link: ', bbox_link)
                # print(anno['link_name'])
                if anno['link_name'] == bbox_link:
                    bbox_to_use = anno['bbox']
                    break
            # print(bbox_to_use)
            # exit()
            bbox_to_use = random_scale * np.array(bbox_to_use)
            self.bbox_object.append(bbox_to_use)
            x, y, _= positions
            bbox = np.array(bbox_to_use) + np.array([x, y, cabinet_offset ])
            self.bbox.append(bbox)

        
        self.bbox = torch.tensor(self.bbox)
        self.bbox_object = torch.tensor(self.bbox_object)
        self.bbox_orig = self.bbox.clone()
        # print(self.bbox)
        self.handle_out =   self.bbox[:,0] -  self.bbox[:,4]
        self.handle_long =  self.bbox[:,1] -  self.bbox[:,0]
        self.handle_short = self.bbox[:,3] -  self.bbox[:,0]

        


    def reset_idx(self, env_ids):
        timeline = omni.timeline.get_timeline_interface()
        # timeline.pause()
       
        

        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
   

        # # reset cabinet
        for _cabinet in self._cabinets:
            _cabinet.set_joint_positions(
                torch.zeros_like(_cabinet.get_joint_positions(clone=False))
            )
            _cabinet.set_joint_velocities(
                torch.zeros_like(_cabinet.get_joint_velocities(clone=False))
            )

        
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)


        world = World()
        # for _ in range(4):
        #     world.step(render=False)

        # self.reset_scale()
        # while True:
        #     world.step(render=True)

        # for _cabinet in self._cabinets:
        #     _cabinet.set_joint_positions(
        #         torch.zeros_like(_cabinet.get_joint_positions(clone=False))
        #     )
        #     _cabinet.set_joint_velocities(
        #         torch.zeros_like(_cabinet.get_joint_velocities(clone=False))
        #     )

        
        # self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=indices)
        # self._frankas.set_joint_positions(dof_pos, indices=indices)
        # self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[:] = 0
        self.progress_buf[:] = 0
        # print success rate
       
        print(
            f"Success rate: {self.success_buf.sum().item() / self.success_buf.numel():.3f}"
        )

        # print success id
        print(self.success_buf.nonzero(as_tuple=False).squeeze(-1)[:,0].squeeze(-1).tolist())

        
        self.success_buf[env_ids] = 0
        cabinet_dof_limits = []
        for cabinet in self._cabinets:
            cabinet_dof_limits.append(cabinet.get_dof_limits())
        # cabinet_dof_limits = self._cabinets.get_dof_limits()

        self.cabinet_dof_lower_limits = []
        self.cabinet_dof_upper_limits = []
       
        for idx, joint_names in enumerate(self.allJointNames):
            # print('cabnet dof limits: ', cabinet_dof_limits[idx].shape)
            # print('joint names: ', joint_names)
            name = joint_names.split('/')[-1]
            # print('cabinet dof names: : ', self._cabinets[idx].dof_names)
            dof_index = self._cabinets[idx].dof_names.index(name)
            self.allJointIndices.append(dof_index)
            self.cabinet_dof_lower_limits.append(cabinet_dof_limits[idx][0, dof_index, 0].to(device=self._device))
            self.cabinet_dof_upper_limits.append(cabinet_dof_limits[idx][0, dof_index, 1].to(device=self._device))
        
        self.cabinet_dof_lower_limits = torch.stack(self.cabinet_dof_lower_limits)
        self.cabinet_dof_upper_limits = torch.stack(self.cabinet_dof_upper_limits)

        timeline.play()
     
        world = World()
        for _ in range(4):
            world.step(render=True)
        
        # world.

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
        print('post reset: ', self.franka_dof_targets.shape)
        
        cabinet_dof_limits = []
        for cabinet in self._cabinets:
            cabinet_dof_limits.append(cabinet.get_dof_limits())
        # cabinet_dof_limits = self._cabinets.get_dof_limits()

        self.cabinet_dof_lower_limits = []
        self.cabinet_dof_upper_limits = []
       
        for idx, joint_names in enumerate(self.allJointNames):
            # print('cabnet dof limits: ', cabinet_dof_limits[idx].shape)
            # print('joint names: ', joint_names)
            name = joint_names.split('/')[-1]
            # print('cabinet dof names: : ', self._cabinets[idx].dof_names)
            dof_index = self._cabinets[idx].dof_names.index(name)
            self.allJointIndices.append(dof_index)
            self.cabinet_dof_lower_limits.append(cabinet_dof_limits[idx][0, dof_index, 0].to(device=self._device))
            self.cabinet_dof_upper_limits.append(cabinet_dof_limits[idx][0, dof_index, 1].to(device=self._device))
        
        self.cabinet_dof_lower_limits = torch.stack(self.cabinet_dof_lower_limits)
        self.cabinet_dof_upper_limits = torch.stack(self.cabinet_dof_upper_limits)


  
        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    
    # def rotate_points_around_z(self, points, angle, axis_location):
    #     """
    #     Rotate multiple points around the z-axis at a specified axis location by a given angle.

    #     :param points: A tensor representing the points, each row is a point (x, y, z).
    #     :param angle: The angle of rotation in radians.
    #     :param axis_location: The location of the axis of rotation (x, y, z).
    #     :return: A tensor representing the rotated points.
    #     """
    #     # Rotation matrix for rotation around the z-axis
    #     # rotation_matrix = torch.tensor([
    #     #     [torch.cos(angle), -torch.sin(angle), 0],
    #     #     [torch.sin(angle), torch.cos(angle), 0],
    #     #     [0, 0, 1]
    #     # ]).to(self._device)
    #     cos_angle = torch.cos(angle)
    #     sin_angle = torch.sin(angle)
    #     rotation_matrices = torch.stack([
    #         torch.stack([cos_angle, -sin_angle, torch.zeros_like(angle)], dim=1),
    #         torch.stack([sin_angle, cos_angle, torch.zeros_like(angle)], dim=1),
    #         torch.stack([torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)], dim=1)
    #     ], dim=1)

    #     # Adjust the points based on the axis location
        
    #     adjusted_points = points - axis_location
    #     points_reshaped = adjusted_points.transpose(1, 2)
       
    #     rotated_points = torch.bmm(rotation_matrices, points_reshaped)

    #     # Transpose back after rotation
    #     rotated_points = rotated_points.transpose(1,2) + axis_location

    #     return rotated_points

    def calculate_metrics(self) -> None:
        
       
        cabinet_dof_pos = []
        cabinet_dof_vel = []
        for dof_index, cabinet in zip(self.allJointIndices, self._cabinets):
            cabinet_dof_pos.append(cabinet.get_joint_positions(clone=False)[:, dof_index])
            cabinet_dof_vel.append(cabinet.get_joint_velocities(clone=False)[:, dof_index])
        
        self.cabinet_dof_pos = torch.stack(cabinet_dof_pos, dim=1)
        self.cabinet_dof_vel = torch.stack(cabinet_dof_vel, dim=1)
        

        scale_tensor_t = self.cabinet_dof_pos.transpose(0, 1)
        position_diff = torch.mul(self.allForwardDirs, scale_tensor_t)
        self.centers = self.centers_orig +  position_diff 
        corners = self.bbox
      
        

        handle_out = corners[:, 0] - corners[:, 4]
        handle_long = corners[:, 1] - corners[:, 0]
        handle_short = corners[:, 3] - corners[:, 0]

        # Assigning the results to the corresponding class attributes
        self.handle_short = handle_short.to(self._device)
        self.handle_out = handle_out.to(self._device)
        self.handle_long = handle_long.to(self._device)
        
        # for idx in range(self._num_envs):

        #     handle_out = corners[idx][0] - corners[idx][4]
        #     handle_long = corners[idx][1] - corners[idx][0]
        #     handle_short = corners[idx][3] - corners[idx][0]

        #     self.handle_short[idx] = handle_short
        #     self.handle_out[idx] = handle_out
        #     self.handle_long[idx] = handle_long
        
        # self.handle_short = self.handle_short.to(self._device)
        # self.handle_out = self.handle_out.to(self._device)
        # self.handle_long = self.handle_long.to(self._device)

        # 
        
        # while True:
        #     color = 4283782485
        #     my_debugDraw = get_debug_draw_interface()
        #     cabinet_dof_pos = []
        #     cabinet_dof_vel = []
        #     for dof_index, cabinet in zip(self.allJointIndices, self._cabinets):
        #         cabinet_dof_pos.append(cabinet.get_joint_positions(clone=False)[:, dof_index])
        #         cabinet_dof_vel.append(cabinet.get_joint_velocities(clone=False)[:, dof_index])
        
        
        #     self.cabinet_dof_pos = torch.stack(cabinet_dof_pos, dim=1)
        #     self.cabinet_dof_vel = torch.stack(cabinet_dof_vel, dim=1)
        #     scale_tensor_t = self.cabinet_dof_pos.transpose(0, 1)
        #     position_diff = torch.mul(self.allForwardDirs, scale_tensor_t).to(self._device)
        #     self.centers = self.centers_orig +  position_diff 

        #     corners = self.bbox.to(self._device) + position_diff.unsqueeze(1) 

        #     # print('pre corners: ', corners)     
        #     # self.centers = corners.mean(dim=1)   
        #     for i, corner in enumerate(corners):
        #         corner = corner.cpu().numpy()
        #         my_debugDraw.draw_line(carb.Float3(corner[0]),color, carb.Float3(corner[4]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[1]),color, carb.Float3(corner[0]), color)
        #         my_debugDraw.draw_line(carb.Float3(corner[3]),color, carb.Float3(corner[0]), color)
        #         # my_debugDraw.draw_line(carb.Float3(corner[0]),color, carb.Float3(self.centers[i].cpu().numpy() ), color)

        #     world = World()
        #     world.step(render=True)
        # handle_out = corners[0] - corners[4]
        # handle_long = corners[1] - corners[0]
        # handle_short = corners[3] - corners[0]

        
        # print(self.centers)
        hand_pos, hand_rot = self.get_ee_pose()
        
        tcp_to_obj_delta = hand_pos - self.centers
        

        tcp_to_obj_dist = torch.norm(tcp_to_obj_delta, dim=-1)

        handle_out_length = torch.norm(self.handle_out, dim = -1).to(self._device)
        handle_long_length = torch.norm(self.handle_long, dim = -1).to(self._device)
        handle_short_length = torch.norm(self.handle_short, dim = -1).to(self._device)

    
        handle_out = self.handle_out / handle_out_length.unsqueeze(-1).to(self._device)
        handle_long = self.handle_long / handle_long_length.unsqueeze(-1).to(self._device)
        handle_short = self.handle_short / handle_short_length.unsqueeze(-1).to(self._device)


        self.franka_lfinger_pos = self._frankas._lfingers.get_world_poses(clone=False)[0] #- self._env_pos
        self.franka_rfinger_pos = self._frankas._rfingers.get_world_poses(clone=False)[0] #- self._env_pos
        
        gripper_length = torch.norm(self.franka_lfinger_pos - self.franka_rfinger_pos, dim=-1)
       
        short_ltip = ((self.franka_lfinger_pos - self.centers) * handle_short).sum(dim=-1) 
        short_rtip = ((self.franka_rfinger_pos - self.centers) *handle_short).sum(dim=-1)
        is_reached_short = (short_ltip * short_rtip) < 0

        is_reached_long = (tcp_to_obj_delta * handle_long).sum(dim=-1).abs() < (handle_long_length / 2.0)
        is_reached_out = (tcp_to_obj_delta * handle_out).sum(dim=-1).abs() < (handle_out_length / 2.0 )

        # print('handle out: ', handle_out)
        # exit()
        ############################

        # hand_grip_dir = quat_axis(hand_rot, 1).cuda()
        # hand_sep_dir =  quat_axis(hand_rot, 0).cuda()
        # hand_down_dir = quat_axis(hand_rot, 2).cuda()
        # dot1 = torch.max((hand_grip_dir * handle_out).sum(dim=-1), (-hand_grip_dir * handle_out).sum(dim=-1))
        # dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
        # dot2 = (-hand_sep_dir * handle_short).sum(dim=-1)
        # dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))

        ############################

        hand_grip_dir = quat_axis(hand_rot, 1).to(self._device)
        hand_sep_dir = quat_axis(hand_rot, 0). to(self._device)
        hand_down_dir = quat_axis(hand_rot, 2).to(self._device)
     
        dot1 = torch.max((hand_grip_dir * handle_out).sum(dim=-1), (-hand_grip_dir * handle_out).sum(dim=-1))
        # dot2 = torch.max((hand_sep_dir * handle_short).sum(dim=-1), (-hand_sep_dir * handle_short).sum(dim=-1)) 
        dot2 = (-hand_sep_dir * handle_short).sum(dim=-1)
        dot3 = torch.max((hand_down_dir * handle_long).sum(dim=-1), (-hand_down_dir * handle_long).sum(dim=-1))

        rot_reward = dot1 + dot2 + dot3 - 3     
        reaching_reward = - tcp_to_obj_dist +  0.1 * (is_reached_short + is_reached_long + is_reached_out) 

        is_reached =  is_reached_out & is_reached_long & is_reached_short #& (tcp_to_obj_dist < 0.03) 

        # if torch.any(is_reached_out):
        #     print('is reached out')
        
        # if torch.any(is_reached_long):
        #     print('is reached long')
        
        # if torch.any(is_reached_short):
        #     print('is reached short')

        # if is_reached.sum() > 10:
        #     print('is reached: ')
        #     print(torch.nonzero(is_reached).squeeze())
        #     print()
        #     timeline = omni.timeline.get_timeline_interface()
        #     print()
        #     from omni.isaac.core import World
        #     world = World()
            
        #     while True:
        #         world.step(render=True)

        # close_reward = is_reached * (gripper_length < 0.02) * 0.1 + 0.1 * (gripper_length > 0.08) * (~is_reached)
        close_reward =  (0.1 - gripper_length ) * is_reached + 0.1 * ( gripper_length -0.1) * (~is_reached)
        # print('close reward: ', close_reward)

        grasp_success = is_reached & (gripper_length < handle_short_length + 0.01) & (rot_reward > -0.2)

        # if torch.any(grasp_success):
        #     if grasp_success.sum() > 0.5 * self._num_envs:
        #         print('grasp half success')
        #     num_true = grasp_success.sum()
        #     if num_true >= 10:

        #         print(torch.nonzero(grasp_success).squeeze())
        #         print('grasp 10 success')
        #         print()
        #         timeline = omni.timeline.get_timeline_interface()
        #         timeline.pause()
        #         from omni.isaac.core import World
        #         world = World()
        #         while True:
        #             world.step(render=True)


        normalized_dof_pos = (self.cabinet_dof_pos - self.cabinet_dof_lower_limits) / (self.cabinet_dof_upper_limits - self.cabinet_dof_lower_limits)
        condition_mask = (normalized_dof_pos >= 0.95) & grasp_success

        # if torch.any(normalized_dof_pos > 0.65):
        #     print('open 65%')
        # print(normalized_dof_pos)

        self.rew_buf[:] = reaching_reward +  rot_reward * 0.5 + 5 * close_reward + grasp_success * 10 * ( 0.1 + normalized_dof_pos) 

        # self.rew_buf = self.rew_buf + self.rew_buf.abs() * rot_reward
        condition_mask = condition_mask.squeeze(0)
        # self.rew_buf[condition_mask] += 2.0
        
        self.rew_buf[:] = self.rew_buf[:].to(torch.float32)

        self.success_buf = torch.zeros((self._num_envs, 1), device=self._device)
        
        success_mask = (normalized_dof_pos > 0.90) #& grasp_success
        fail_mask = ~ success_mask

        success_mask = success_mask.squeeze(0)
        fail_mask = fail_mask.squeeze(0)
        self.success_buf[success_mask] = 1.0
        self.success_buf[fail_mask] = 0.0
        

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