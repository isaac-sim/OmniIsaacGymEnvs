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

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.warp as warp_utils
from omniisaacgymenvs.tasks.base.rl_task import RLTaskWarp

import numpy as np
import torch
import warp as wp


class LocomotionTask(RLTaskWarp):
    def __init__(
        self,
        name,
        env,
        offset=None
    ) -> None:

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self._task_cfg["env"]["angularVelocityScale"]
        self.contact_force_scale = self._task_cfg["env"]["contactForceScale"]
        self.power_scale = self._task_cfg["env"]["powerScale"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]
        self._num_sensors = 2

        RLTaskWarp.__init__(self, name, env)
        return

    @abstractmethod
    def set_up_scene(self, scene) -> None:
        pass

    @abstractmethod
    def get_robot(self):
        pass

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        velocities = self._robots.get_velocities(clone=False)
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        # force sensors attached to the feet
        sensor_force_torques = self._robots.get_measured_joint_forces()
        
        wp.launch(get_observations, dim=self._num_envs,
            inputs=[self.obs_buf, torso_position, torso_rotation, self._env_pos, velocities, dof_pos, dof_vel, 
                self.prev_potentials, self.potentials, self.dt, self.target, 
                self.basis_vec0, self.basis_vec1, self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
                sensor_force_torques, self.contact_force_scale, self.actions, self.angular_velocity_scale,
                self._robots._num_dof, self._num_sensors, self._sensor_indices], device=self._device
        )

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        self.reset_idx()

        actions_wp = wp.from_torch(actions)
        self.actions = actions_wp
        wp.launch(compute_forces, dim=(self._num_envs, self._robots._num_dof),
            inputs=[self.forces, self.actions, self.joint_gears, self.power_scale], device=self._device)

        # applies joint torques
        self._robots.set_joint_efforts(self.forces)

    def reset_idx(self):
        reset_env_ids = wp.to_torch(self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        num_resets = len(reset_env_ids)
        indices = wp.from_torch(reset_env_ids.to(dtype=torch.int32), dtype=wp.int32)

        if num_resets > 0:
            wp.launch(reset_dofs, dim=(num_resets, self._robots._num_dof), 
                inputs=[self.dof_pos, self.dof_vel, self.initial_dof_pos, self.dof_limits_lower, self.dof_limits_upper, indices, self._rand_seed], 
                device=self._device)

            wp.launch(reset_idx, dim=num_resets, 
                inputs=[self.root_pos, self.root_rot, self.initial_root_pos, self.initial_root_rot, self._env_pos,
                self.target, self.prev_potentials, self.potentials, self.dt,
                self.reset_buf, self.progress_buf, indices, self._rand_seed], 
                device=self._device)

            # apply resets
            self._robots.set_joint_positions(self.dof_pos[indices], indices=indices)
            self._robots.set_joint_velocities(self.dof_vel[indices], indices=indices)

            self._robots.set_world_poses(self.root_pos[indices], self.root_rot[indices], indices=indices)
            self._robots.set_velocities(self.root_vel[indices], indices=indices)

    def post_reset(self):
        self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()
        self.initial_dof_pos = self._robots.get_joint_positions()

        # initialize some data used later on
        self.basis_vec0 = wp.vec3(1, 0, 0)
        self.basis_vec1 = wp.vec3(0, 0, 1)
        self.target = wp.vec3(1000, 0, 0)
        self.dt = 1.0 / 60.0

        # initialize potentials
        self.potentials = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        self.prev_potentials = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
        wp.launch(init_potentials, dim=self._num_envs, 
            inputs=[self.potentials, self.prev_potentials, self.dt], device=self._device)

        self.actions = wp.zeros((self.num_envs, self.num_actions), device=self._device, dtype=wp.float32)
        self.forces = wp.zeros((self._num_envs, self._robots._num_dof), dtype=wp.float32, device=self._device)

        self.dof_pos = wp.zeros((self.num_envs, self._robots._num_dof), device=self._device, dtype=wp.float32)
        self.dof_vel = wp.zeros((self.num_envs, self._robots._num_dof), device=self._device, dtype=wp.float32)

        self.root_pos = wp.zeros((self.num_envs, 3), device=self._device, dtype=wp.float32)
        self.root_rot = wp.zeros((self.num_envs, 4), device=self._device, dtype=wp.float32)
        self.root_vel = wp.zeros((self.num_envs, 6), device=self._device, dtype=wp.float32)

        # randomize all env
        self.reset_idx()

    def calculate_metrics(self) -> None:
        dof_at_limit_cost = self.get_dof_at_limit_cost()        
        wp.launch(calculate_metrics, dim=self._num_envs,
            inputs=[self.rew_buf, self.obs_buf, self.actions, self.up_weight, self.heading_weight, self.potentials, self.prev_potentials,
            self.actions_cost_scale, self.energy_cost_scale, self.termination_height,
            self.death_cost, self._robots.num_dof, dof_at_limit_cost, self.alive_reward_scale, self.motor_effort_ratio],
            device=self._device
        )

    def is_done(self) -> None:
        wp.launch(is_done, dim=self._num_envs,
            inputs=[self.obs_buf, self.termination_height, self.reset_buf, self.progress_buf, self._max_episode_length],
            device=self._device
        )


#####################################################################
###==========================warp kernels=========================###
#####################################################################

@wp.kernel
def init_potentials(potentials: wp.array(dtype=wp.float32),
                    prev_potentials: wp.array(dtype=wp.float32),
                    dt: float):
    i = wp.tid()
    potentials[i] = -1000.0 / dt
    prev_potentials[i] = -1000.0 / dt

@wp.kernel
def reset_idx(root_pos: wp.array(dtype=wp.float32, ndim=2),
              root_rot: wp.array(dtype=wp.float32, ndim=2),
              initial_root_pos: wp.indexedarray(dtype=wp.float32, ndim=2),
              initial_root_rot: wp.indexedarray(dtype=wp.float32, ndim=2), 
              env_pos: wp.array(dtype=wp.float32, ndim=2),
              target: wp.vec3, 
              prev_potentials: wp.array(dtype=wp.float32), 
              potentials: wp.array(dtype=wp.float32),
              dt: float,
              reset_buf: wp.array(dtype=wp.int32), 
              progress_buf: wp.array(dtype=wp.int32),
              indices: wp.array(dtype=wp.int32),
              rand_seed: int):
    i = wp.tid()
    idx = indices[i]

    # reset root states
    for j in range(3):
        root_pos[idx, j] = initial_root_pos[idx, j]
    for j in range(4):
        root_rot[idx, j] = initial_root_rot[idx, j]

    # reset potentials
    to_target = target - wp.vec3(initial_root_pos[idx, 0] - env_pos[idx, 0], initial_root_pos[idx, 1] - env_pos[idx, 1], target[2])
    prev_potentials[idx] = -wp.length(to_target) / dt
    potentials[idx] = -wp.length(to_target) / dt
    temp = potentials[idx] - prev_potentials[idx]

    # bookkeeping
    reset_buf[idx] = 0
    progress_buf[idx] = 0

@wp.kernel
def reset_dofs(dof_pos: wp.array(dtype=wp.float32, ndim=2), 
              dof_vel: wp.array(dtype=wp.float32, ndim=2), 
              initial_dof_pos: wp.indexedarray(dtype=wp.float32, ndim=2), 
              dof_limits_lower: wp.array(dtype=wp.float32),
              dof_limits_upper: wp.array(dtype=wp.float32),
              indices: wp.array(dtype=wp.int32),
              rand_seed: int):

    i, j = wp.tid()
    idx = indices[i]
    rand_state = wp.rand_init(rand_seed, i * j + j)

    # randomize DOF positions and velocities
    dof_pos[idx, j] = wp.clamp(wp.randf(rand_state, -0.2, 0.2) + initial_dof_pos[idx, j], dof_limits_lower[j], dof_limits_upper[j])
    dof_vel[idx, j] = wp.randf(rand_state, -0.1, 0.1)

@wp.kernel
def compute_forces(forces: wp.array(dtype=wp.float32, ndim=2), 
                   actions: wp.array(dtype=wp.float32, ndim=2), 
                   joint_gears: wp.array(dtype=wp.float32),
                   power_scale: float):
    i, j = wp.tid()
    forces[i, j] = actions[i, j] * joint_gears[j] * power_scale

@wp.func
def get_euler_xyz(q: wp.quat):
    qx = 0
    qy = 1
    qz = 2
    qw = 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[qw] * q[qx] + q[qy] * q[qz])
    cosr_cosp = q[qw] * q[qw] - q[qx] * q[qx] - q[qy] * q[qy] + q[qz] * q[qz]
    roll = wp.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[qw] * q[qy] - q[qz] * q[qx])
    if wp.abs(sinp) >= 1:
        pitch = warp_utils.PI / 2.0 * (wp.abs(sinp)/sinp)
    else:
        pitch = wp.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[qw] * q[qz] + q[qx] * q[qy])
    cosy_cosp = q[qw] * q[qw] + q[qx] * q[qx] - q[qy] * q[qy] - q[qz] * q[qz]
    yaw = wp.atan2(siny_cosp, cosy_cosp)

    rpy = wp.vec3(roll % (2.0 * warp_utils.PI), pitch % (2.0 * warp_utils.PI), yaw % (2.0 * warp_utils.PI))
    return rpy

@wp.func
def compute_up_vec(torso_rotation: wp.quat, vec1: wp.vec3):
    up_vec = wp.quat_rotate(torso_rotation, vec1)
    return up_vec

@wp.func
def compute_heading_vec(torso_rotation: wp.quat, vec0: wp.vec3):
    heading_vec = wp.quat_rotate(torso_rotation, vec0)
    return heading_vec

@wp.func
def unscale(x:float, lower:float, upper:float):
    return (2.0 * x - upper - lower) / (upper - lower)

@wp.func
def normalize_angle(x: float):
    return wp.atan2(wp.sin(x), wp.cos(x))

@wp.kernel
def get_observations(
    obs_buf: wp.array(dtype=wp.float32, ndim=2),
    torso_pos: wp.indexedarray(dtype=wp.float32, ndim=2),
    torso_rot: wp.indexedarray(dtype=wp.float32, ndim=2),
    env_pos: wp.array(dtype=wp.float32, ndim=2),
    velocity: wp.indexedarray(dtype=wp.float32, ndim=2),
    dof_pos: wp.indexedarray(dtype=wp.float32, ndim=2),
    dof_vel: wp.indexedarray(dtype=wp.float32, ndim=2),
    prev_potentials: wp.array(dtype=wp.float32), 
    potentials: wp.array(dtype=wp.float32),
    dt: float,
    target: wp.vec3,
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_limits_lower: wp.array(dtype=wp.float32),
    dof_limits_upper: wp.array(dtype=wp.float32),
    dof_vel_scale: float,
    sensor_force_torques: wp.indexedarray(dtype=wp.float32, ndim=3),
    contact_force_scale: float,
    actions: wp.array(dtype=wp.float32, ndim=2),
    angular_velocity_scale: float,
    num_dofs: int,
    num_sensors: int,
    sensor_indices: wp.array(dtype=wp.int32)
):
    i = wp.tid()

    torso_position_x = torso_pos[i, 0] - env_pos[i, 0]
    torso_position_y = torso_pos[i, 1] - env_pos[i, 1]
    torso_position_z = torso_pos[i, 2] - env_pos[i, 2]
    to_target = target - wp.vec3(torso_position_x, torso_position_y, target[2])

    prev_potentials[i] = potentials[i]
    potentials[i] = -wp.length(to_target) / dt
    temp = potentials[i] - prev_potentials[i]

    torso_quat = wp.quat(torso_rot[i, 1], torso_rot[i, 2], torso_rot[i, 3], torso_rot[i, 0])

    up_vec = compute_up_vec(torso_quat, basis_vec1)
    up_proj = up_vec[2]
    heading_vec = compute_heading_vec(torso_quat, basis_vec0)
    target_dir = wp.normalize(to_target)
    heading_proj = wp.dot(heading_vec, target_dir)

    lin_velocity = wp.vec3(velocity[i, 0], velocity[i, 1], velocity[i, 2])
    ang_velocity = wp.vec3(velocity[i, 3], velocity[i, 4], velocity[i, 5])

    rpy = get_euler_xyz(torso_quat)
    vel_loc = wp.quat_rotate_inv(torso_quat, lin_velocity)
    angvel_loc = wp.quat_rotate_inv(torso_quat, ang_velocity)
    walk_target_angle = wp.atan2(target[2] - torso_position_z, target[0] - torso_position_x)
    angle_to_target = walk_target_angle - rpy[2] # yaw

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, num_sensors * 6, num_dofs
    obs_offset = 0
    obs_buf[i, 0] = torso_position_z
    obs_offset = obs_offset + 1
    for j in range(3):
        obs_buf[i, j+obs_offset] = vel_loc[j]
    obs_offset = obs_offset + 3
    for j in range(3):
        obs_buf[i, j+obs_offset] = angvel_loc[j] * angular_velocity_scale
    obs_offset = obs_offset + 3
    obs_buf[i, obs_offset+0] = normalize_angle(rpy[2])
    obs_buf[i, obs_offset+1] = normalize_angle(rpy[0])
    obs_buf[i, obs_offset+2] = normalize_angle(angle_to_target)
    obs_buf[i, obs_offset+3] = up_proj
    obs_buf[i, obs_offset+4] = heading_proj
    obs_offset = obs_offset + 5
    for j in range(num_dofs):
        obs_buf[i, obs_offset+j] = unscale(dof_pos[i, j], dof_limits_lower[j], dof_limits_upper[j])
    obs_offset = obs_offset + num_dofs
    for j in range(num_dofs):
        obs_buf[i, obs_offset+j] = dof_vel[i, j] * dof_vel_scale
    obs_offset = obs_offset + num_dofs
    for j in range(num_sensors):
        sensor_idx = sensor_indices[j]
        for k in range(6):
            obs_buf[i, obs_offset+j*6+k] = sensor_force_torques[i, sensor_idx, k] * contact_force_scale
    obs_offset = obs_offset + (num_sensors * 6)
    for j in range(num_dofs):
        obs_buf[i, obs_offset+j] = actions[i, j]

@wp.kernel
def is_done(
    obs_buf: wp.array(dtype=wp.float32, ndim=2),
    termination_height: float,
    reset_buf: wp.array(dtype=wp.int32),
    progress_buf: wp.array(dtype=wp.int32),
    max_episode_length: int
):
    i = wp.tid()
    if obs_buf[i, 0] < termination_height or progress_buf[i] >= max_episode_length - 1:
        reset_buf[i] = 1
    else:
        reset_buf[i] = 0

@wp.kernel
def calculate_metrics(
    rew_buf: wp.array(dtype=wp.float32),
    obs_buf: wp.array(dtype=wp.float32, ndim=2),
    actions: wp.array(dtype=wp.float32, ndim=2),
    up_weight: float,
    heading_weight: float,
    potentials: wp.array(dtype=wp.float32),
    prev_potentials: wp.array(dtype=wp.float32),
    actions_cost_scale: float,
    energy_cost_scale: float,
    termination_height: float,
    death_cost: float,
    num_dof: int,
    dof_at_limit_cost: wp.array(dtype=wp.float32),
    alive_reward_scale: float,
    motor_effort_ratio: wp.array(dtype=wp.float32)
):
    i = wp.tid()

    # heading reward
    if obs_buf[i, 11] > 0.8:
        heading_reward = heading_weight
    else:
        heading_reward = heading_weight * obs_buf[i, 11] / 0.8

    # aligning up axis of robot and environment
    up_reward = 0.0
    if obs_buf[i, 10] > 0.93:
        up_reward = up_weight

    # energy penalty for movement
    actions_cost = float(0.0)
    electricity_cost = float(0.0)
    for j in range(num_dof):
        actions_cost = actions_cost + (actions[i, j] *  actions[i, j])
        electricity_cost = electricity_cost + (wp.abs(actions[i, j] * obs_buf[i, 12+num_dof+j]) * motor_effort_ratio[j])

    # reward for duration of staying alive
    progress_reward = potentials[i] - prev_potentials[i]

    total_reward = (
        progress_reward
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost[i]
    )

    # adjust reward for fallen agents
    if obs_buf[i, 0] < termination_height:
        total_reward = death_cost

    rew_buf[i] = total_reward
