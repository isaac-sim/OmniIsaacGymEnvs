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
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom


class FrankaCabinetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 23
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
        self.get_franka()
        self.get_cabinet()
        if self.num_props > 0:
            self.get_props()

        super().set_up_scene(scene, filter_collisions=False)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

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
        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka")
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path), self._sim_config.parse_actor_config("franka")
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", name="cabinet")
        self._sim_config.apply_articulation_settings(
            "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        )

    def get_props(self):
        prop_cloner = Cloner()
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.08
        prop_spacing = 0.09
        xmin = -0.5 * prop_spacing * (props_per_row - 1)
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])

                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

        prop_paths = [f"{self.default_zero_env_path}/prop/prop_{j}" for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + drawer_pos.numpy(),
            replicate_physics=False,
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

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        (
            self.franka_grasp_rot,
            self.franka_grasp_pos,
            self.drawer_grasp_rot,
            self.drawer_grasp_pos,
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,
                self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                self.cabinet_dof_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )

        observations = {self._frankas.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=indices
        )

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()],
                self.default_prop_rot[self.prop_indices[env_ids].flatten()],
                self.prop_indices[env_ids].flatten().to(torch.int32),
            )

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

        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.cabinet_dof_pos,
            self.franka_grasp_pos,
            self.drawer_grasp_pos,
            self.franka_grasp_rot,
            self.drawer_grasp_rot,
            self.franka_lfinger_pos,
            self.franka_rfinger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.franka_dof_pos,
            self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
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

    def compute_franka_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
            torch.where(
                franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]), finger_close_reward
        )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward * finger_close_reward_scale
        )

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
