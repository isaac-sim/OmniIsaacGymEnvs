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

import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.maths import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.balance_bot import BalanceBot
from pxr import PhysxSchema


class BallBalanceTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)

        self._num_observations = 12 + 12
        self._num_actions = 3

        self.anchored = False

        RLTask.__init__(self, name, env)

        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._dt = self._task_cfg["sim"]["dt"]
        self._table_position = torch.tensor([0, 0, 0.56])
        self._ball_position = torch.tensor([0.0, 0.0, 1.0])
        self._ball_radius = 0.1

        self._action_speed_scale = self._task_cfg["env"]["actionSpeedScale"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

    def set_up_scene(self, scene) -> None:
        self.get_balance_table()
        self.add_ball()
        super().set_up_scene(scene, replicate_physics=False)
        self.set_up_table_anchors()
        self._balance_bots = ArticulationView(
            prim_paths_expr="/World/envs/.*/BalanceBot/tray", name="balance_bot_view", reset_xform_properties=False
        )
        scene.add(self._balance_bots)
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False
        )
        scene.add(self._balls)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("balance_bot_view"):
            scene.remove_object("balance_bot_view", registry_only=True)
        if scene.object_exists("ball_view"):
            scene.remove_object("ball_view", registry_only=True)
        self._balance_bots = ArticulationView(
            prim_paths_expr="/World/envs/.*/BalanceBot/tray", name="balance_bot_view", reset_xform_properties=False
        )
        scene.add(self._balance_bots)
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Ball/ball", name="ball_view", reset_xform_properties=False
        )
        scene.add(self._balls)

    def get_balance_table(self):
        balance_table = BalanceBot(
            prim_path=self.default_zero_env_path + "/BalanceBot", name="BalanceBot", translation=self._table_position
        )
        self._sim_config.apply_articulation_settings(
            "table", get_prim_at_path(balance_table.prim_path), self._sim_config.parse_actor_config("table")
        )

    def add_ball(self):
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/Ball/ball",
            translation=self._ball_position,
            name="ball_0",
            radius=self._ball_radius,
            color=torch.tensor([0.9, 0.6, 0.2]),
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )

    def set_up_table_anchors(self):
        from pxr import Gf
        height = 0.08
        stage = get_current_stage()
        for i in range(self._num_envs):
            base_path = f"{self.default_base_env_path}/env_{i}/BalanceBot"
            for j, leg_offset in enumerate([(0.4, 0, height), (-0.2, 0.34641, 0), (-0.2, -0.34641, 0)]):
                # fix the legs to ground
                leg_path = f"{base_path}/lower_leg{j}"
                ground_joint_path = leg_path + "_ground"
                env_pos = stage.GetPrimAtPath(f"{self.default_base_env_path}/env_{i}").GetAttribute("xformOp:translate").Get()
                anchor_pos = env_pos + Gf.Vec3d(*leg_offset)
                self.fix_to_ground(stage, ground_joint_path, leg_path, anchor_pos)

    def fix_to_ground(self, stage, joint_path, prim_path, anchor_pos):
        from pxr import UsdPhysics, Gf
        # D6 fixed joint
        d6FixedJoint = UsdPhysics.Joint.Define(stage, joint_path)
        d6FixedJoint.CreateBody0Rel().SetTargets(["/World/defaultGroundPlane"])
        d6FixedJoint.CreateBody1Rel().SetTargets([prim_path])
        d6FixedJoint.CreateLocalPos0Attr().Set(anchor_pos)
        d6FixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        d6FixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.18))
        d6FixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        # lock all DOF (lock - low is greater than high)
        d6Prim = stage.GetPrimAtPath(joint_path)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)

    def get_observations(self) -> dict:
        ball_positions, ball_orientations = self._balls.get_world_poses(clone=False)
        ball_positions = ball_positions[:, 0:3] - self._env_pos

        ball_velocities = self._balls.get_velocities(clone=False)
        ball_linvels = ball_velocities[:, 0:3]
        ball_angvels = ball_velocities[:, 3:6]

        dof_pos = self._balance_bots.get_joint_positions(clone=False)
        dof_vel = self._balance_bots.get_joint_velocities(clone=False)

        sensor_force_torques = self._balance_bots.get_measured_joint_forces(joint_indices=self._sensor_indices) # (num_envs, num_sensors, 6)

        self.obs_buf[..., 0:3] = dof_pos[..., self.actuated_dof_indices]
        self.obs_buf[..., 3:6] = dof_vel[..., self.actuated_dof_indices]
        self.obs_buf[..., 6:9] = ball_positions
        self.obs_buf[..., 9:12] = ball_linvels
        self.obs_buf[..., 12:15] = sensor_force_torques[..., 0] / 20.0
        self.obs_buf[..., 15:18] = sensor_force_torques[..., 3] / 20.0
        self.obs_buf[..., 18:21] = sensor_force_torques[..., 4] / 20.0
        self.obs_buf[..., 21:24] = sensor_force_torques[..., 5] / 20.0

        self.ball_positions = ball_positions
        self.ball_linvels = ball_linvels

        observations = {"ball_balance": {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # update position targets from actions
        self.dof_position_targets[..., self.actuated_dof_indices] += (
            self._dt * self._action_speed_scale * actions.to(self.device)
        )
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits
        )

        # reset position targets for reset envs
        self.dof_position_targets[reset_env_ids] = 0

        self._balance_bots.set_joint_position_targets(self.dof_position_targets)  # .clone())

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        env_ids_32 = env_ids.type(torch.int32)
        env_ids_64 = env_ids.type(torch.int64)

        min_d = 0.001  # min horizontal dist from origin
        max_d = 0.4  # max horizontal dist from origin
        min_height = 1.0
        max_height = 2.0
        min_horizontal_speed = 0
        max_horizontal_speed = 2

        dists = torch_rand_float(min_d, max_d, (num_resets, 1), self._device)
        dirs = torch_random_dir_2((num_resets, 1), self._device)
        hpos = dists * dirs

        speedscales = (dists - min_d) / (max_d - min_d)
        hspeeds = torch_rand_float(min_horizontal_speed, max_horizontal_speed, (num_resets, 1), self._device)
        hvels = -speedscales * hspeeds * dirs
        vspeeds = -torch_rand_float(5.0, 5.0, (num_resets, 1), self._device).squeeze()

        ball_pos = self.initial_ball_pos.clone()
        ball_rot = self.initial_ball_rot.clone()
        # position
        ball_pos[env_ids_64, 0:2] += hpos[..., 0:2]
        ball_pos[env_ids_64, 2] += torch_rand_float(min_height, max_height, (num_resets, 1), self._device).squeeze()
        # rotation
        ball_rot[env_ids_64, 0] = 1
        ball_rot[env_ids_64, 1:] = 0
        ball_velocities = self.initial_ball_velocities.clone()
        # linear
        ball_velocities[env_ids_64, 0:2] = hvels[..., 0:2]
        ball_velocities[env_ids_64, 2] = vspeeds
        # angular
        ball_velocities[env_ids_64, 3:6] = 0

        # reset root state for bbots and balls in selected envs
        self._balls.set_world_poses(ball_pos[env_ids_64], ball_rot[env_ids_64], indices=env_ids_32)
        self._balls.set_velocities(ball_velocities[env_ids_64], indices=env_ids_32)

        # reset root pose and velocity
        self._balance_bots.set_world_poses(
            self.initial_bot_pos[env_ids_64].clone(), self.initial_bot_rot[env_ids_64].clone(), indices=env_ids_32
        )
        self._balance_bots.set_velocities(self.initial_bot_velocities[env_ids_64].clone(), indices=env_ids_32)

        # reset DOF states for bbots in selected envs
        self._balance_bots.set_joint_positions(self.initial_dof_positions[env_ids_64].clone(), indices=env_ids_32)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        dof_limits = self._balance_bots.get_dof_limits()
        self.bbot_dof_lower_limits, self.bbot_dof_upper_limits = torch.t(dof_limits[0].to(device=self._device))

        self.initial_dof_positions = self._balance_bots.get_joint_positions()
        self.initial_bot_pos, self.initial_bot_rot = self._balance_bots.get_world_poses()
        # self.initial_bot_pos[..., 2] = 0.559  # tray_height
        self.initial_bot_velocities = self._balance_bots.get_velocities()
        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses()
        self.initial_ball_velocities = self._balls.get_velocities()

        self.dof_position_targets = torch.zeros(
            (self.num_envs, self._balance_bots.num_dof), dtype=torch.float32, device=self._device, requires_grad=False
        )

        actuated_joints = ["lower_leg0", "lower_leg1", "lower_leg2"]
        self.actuated_dof_indices = torch.tensor(
            [self._balance_bots._dof_indices[j] for j in actuated_joints], device=self._device, dtype=torch.long
        )
        force_links = ["upper_leg0", "upper_leg1", "upper_leg2"]
        self._sensor_indices = torch.tensor(
            [self._balance_bots._body_indices[j] for j in force_links], device=self._device, dtype=torch.long
        )

    def calculate_metrics(self) -> None:
        ball_dist = torch.sqrt(
            self.ball_positions[..., 0] * self.ball_positions[..., 0]
            + (self.ball_positions[..., 2] - 0.7) * (self.ball_positions[..., 2] - 0.7)
            + (self.ball_positions[..., 1]) * self.ball_positions[..., 1]
        )
        ball_speed = torch.sqrt(
            self.ball_linvels[..., 0] * self.ball_linvels[..., 0]
            + self.ball_linvels[..., 1] * self.ball_linvels[..., 1]
            + self.ball_linvels[..., 2] * self.ball_linvels[..., 2]
        )
        pos_reward = 1.0 / (1.0 + ball_dist)
        speed_reward = 1.0 / (1.0 + ball_speed)
        self.rew_buf[:] = pos_reward * speed_reward

    def is_done(self) -> None:
        reset = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )
        reset = torch.where(
            self.ball_positions[..., 2] < self._ball_radius * 1.5, torch.ones_like(self.reset_buf), reset
        )
        self.reset_buf[:] = reset
