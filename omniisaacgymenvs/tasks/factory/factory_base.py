# Copyright (c) 2018-2023, NVIDIA Corporation
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

"""Factory: base class.

Inherits Gym's RLTask class and abstract base class. Inherited by environment classes. Not directly executed.

Configuration defined in FactoryBase.yaml. Asset info defined in factory_asset_info_franka_table.yaml.
"""


import carb
import hydra
import math
import numpy as np
import torch

from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.factory_franka import FactoryFranka
from pxr import PhysxSchema, UsdPhysics
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_schema_class_base import FactoryABCBase
from omniisaacgymenvs.tasks.factory.factory_schema_config_base import (
    FactorySchemaConfigBase,
)


class FactoryBase(RLTask, FactoryABCBase):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize instance variables. Initialize RLTask superclass."""

        # Set instance variables from base YAML
        self._get_base_yaml_params()
        self._env_spacing = self.cfg_base.env.env_spacing

        # Set instance variables from task and train YAMLs
        self._sim_config = sim_config
        self._cfg = sim_config.config  # CL args, task config, and train config
        self._task_cfg = sim_config.task_config  # just task config
        self._num_envs = sim_config.task_config["env"]["numEnvs"]
        self._num_observations = sim_config.task_config["env"]["numObservations"]
        self._num_actions = sim_config.task_config["env"]["numActions"]

        super().__init__(name, env)

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        config_path = (
            "task/FactoryBase.yaml"  # relative to Gym's Hydra search path (cfg dir)
        )
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_franka_table.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table[""][""][""][
            "tasks"
        ]["factory"][
            "yaml"
        ]  # strip superfluous nesting

    def import_franka_assets(self, add_to_stage=True):
        """Set Franka and table asset options. Import assets."""

        self._stage = get_current_stage()

        if add_to_stage:
            franka_translation = np.array([self.cfg_base.env.franka_depth, 0.0, 0.0])
            franka_orientation = np.array([0.0, 0.0, 0.0, 1.0])

            franka = FactoryFranka(
                prim_path=self.default_zero_env_path + "/franka",
                name="franka",
                translation=franka_translation,
                orientation=franka_orientation,
            )
            self._sim_config.apply_articulation_settings(
                "franka",
                get_prim_at_path(franka.prim_path),
                self._sim_config.parse_actor_config("franka"),
            )

            for link_prim in franka.prim.GetChildren():
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(
                        self._stage, link_prim.GetPrimPath()
                    )
                    rb.GetDisableGravityAttr().Set(True)
                    rb.GetRetainAccelerationsAttr().Set(False)
                    if self.cfg_base.sim.add_damping:
                        rb.GetLinearDampingAttr().Set(
                            1.0
                        )  # default = 0.0; increased to improve stability
                        rb.GetMaxLinearVelocityAttr().Set(
                            1.0
                        )  # default = 1000.0; reduced to prevent CUDA errors
                        rb.GetAngularDampingAttr().Set(
                            5.0
                        )  # default = 0.5; increased to improve stability
                        rb.GetMaxAngularVelocityAttr().Set(
                            2 / math.pi * 180
                        )  # default = 64.0; reduced to prevent CUDA errors
                    else:
                        rb.GetLinearDampingAttr().Set(0.0)
                        rb.GetMaxLinearVelocityAttr().Set(1000.0)
                        rb.GetAngularDampingAttr().Set(0.5)
                        rb.GetMaxAngularVelocityAttr().Set(64 / math.pi * 180)

            table_translation = np.array(
                [0.0, 0.0, self.cfg_base.env.table_height * 0.5]
            )
            table_orientation = np.array([1.0, 0.0, 0.0, 0.0])

            table = FixedCuboid(
                prim_path=self.default_zero_env_path + "/table",
                name="table",
                translation=table_translation,
                orientation=table_orientation,
                scale=np.array(
                    [
                        self.asset_info_franka_table.table_depth,
                        self.asset_info_franka_table.table_width,
                        self.cfg_base.env.table_height,
                    ]
                ),
                size=1.0,
                color=np.array([0, 0, 0]),
            )

        self.parse_controller_spec(add_to_stage=add_to_stage)

    def acquire_base_tensors(self):
        """Acquire tensors."""

        self.num_dofs = 9
        self.env_pos = self._env_pos

        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.dof_torque = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.ctrl_target_dof_pos = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device
        )
        self.ctrl_target_gripper_dof_pos = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self.ctrl_target_fingertip_contact_wrench = torch.zeros(
            (self.num_envs, 6), device=self.device
        )

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

    def refresh_base_tensors(self):
        """Refresh tensors."""

        if not self._env._world.is_playing():
            return

        self.dof_pos = self.frankas.get_joint_positions(clone=False)
        self.dof_vel = self.frankas.get_joint_velocities(clone=False)

        # Jacobian shape: [4, 11, 6, 9] (root has no Jacobian)
        self.franka_jacobian = self.frankas.get_jacobians()
        self.franka_mass_matrix = self.frankas.get_mass_matrices(clone=False)

        self.arm_dof_pos = self.dof_pos[:, 0:7]
        self.arm_mass_matrix = self.franka_mass_matrix[
            :, 0:7, 0:7
        ]  # for Franka arm (not gripper)

        self.hand_pos, self.hand_quat = self.frankas._hands.get_world_poses(clone=False)
        self.hand_pos -= self.env_pos
        hand_velocities = self.frankas._hands.get_velocities(clone=False)
        self.hand_linvel = hand_velocities[:, 0:3]
        self.hand_angvel = hand_velocities[:, 3:6]

        (
            self.left_finger_pos,
            self.left_finger_quat,
        ) = self.frankas._lfingers.get_world_poses(clone=False)
        self.left_finger_pos -= self.env_pos
        left_finger_velocities = self.frankas._lfingers.get_velocities(clone=False)
        self.left_finger_linvel = left_finger_velocities[:, 0:3]
        self.left_finger_angvel = left_finger_velocities[:, 3:6]
        self.left_finger_jacobian = self.franka_jacobian[:, 8, 0:6, 0:7]
        left_finger_forces = self.frankas._lfingers.get_net_contact_forces(clone=False)
        self.left_finger_force = left_finger_forces[:, 0:3]

        (
            self.right_finger_pos,
            self.right_finger_quat,
        ) = self.frankas._rfingers.get_world_poses(clone=False)
        self.right_finger_pos -= self.env_pos
        right_finger_velocities = self.frankas._rfingers.get_velocities(clone=False)
        self.right_finger_linvel = right_finger_velocities[:, 0:3]
        self.right_finger_angvel = right_finger_velocities[:, 3:6]
        self.right_finger_jacobian = self.franka_jacobian[:, 9, 0:6, 0:7]
        right_finger_forces = self.frankas._rfingers.get_net_contact_forces(clone=False)
        self.right_finger_force = right_finger_forces[:, 0:3]

        self.gripper_dof_pos = self.dof_pos[:, 7:9]

        (
            self.fingertip_centered_pos,
            self.fingertip_centered_quat,
        ) = self.frankas._fingertip_centered.get_world_poses(clone=False)
        self.fingertip_centered_pos -= self.env_pos
        fingertip_centered_velocities = self.frankas._fingertip_centered.get_velocities(
            clone=False
        )
        self.fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        self.fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]
        self.fingertip_centered_jacobian = self.franka_jacobian[:, 10, 0:6, 0:7]

        self.finger_midpoint_pos = (self.left_finger_pos + self.right_finger_pos) / 2
        self.fingertip_midpoint_pos = fc.translate_along_local_z(
            pos=self.finger_midpoint_pos,
            quat=self.hand_quat,
            offset=self.asset_info_franka_table.franka_finger_length,
            device=self.device,
        )
        self.fingertip_midpoint_quat = self.fingertip_centered_quat  # always equal

        # TODO: Add relative velocity term (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        self.fingertip_midpoint_linvel = self.fingertip_centered_linvel + torch.cross(
            self.fingertip_centered_angvel,
            (self.fingertip_midpoint_pos - self.fingertip_centered_pos),
            dim=1,
        )

        # From sum of angular velocities (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand; angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.
        self.fingertip_midpoint_angvel = self.fingertip_centered_angvel  # always equal

        self.fingertip_midpoint_jacobian = (
            self.left_finger_jacobian + self.right_finger_jacobian
        ) * 0.5

    def parse_controller_spec(self, add_to_stage):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {
            "num_envs",
            "jacobian_type",
            "gripper_prop_gains",
            "gripper_deriv_gains",
            "motor_ctrl_mode",
            "gain_space",
            "ik_method",
            "joint_prop_gains",
            "joint_deriv_gains",
            "do_motion_ctrl",
            "task_prop_gains",
            "task_deriv_gains",
            "do_inertial_comp",
            "motion_ctrl_axes",
            "do_force_ctrl",
            "force_ctrl_method",
            "wrench_prop_gains",
            "force_ctrl_axes",
        }
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl["num_envs"] = self.num_envs
        self.cfg_ctrl["jacobian_type"] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_prop_gains, device=self.device
        ).repeat((self.num_envs, 1))
        self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
            self.cfg_task.ctrl.all.gripper_deriv_gains, device=self.device
        ).repeat((self.num_envs, 1))

        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == "gym_default":
            self.cfg_ctrl["motor_ctrl_mode"] = "gym"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["gripper_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.gripper_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["gripper_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.gym_default.gripper_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "joint_space_ik":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = False
        elif ctrl_type == "joint_space_id":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "joint"
            self.cfg_ctrl["ik_method"] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl["joint_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_prop_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["joint_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_deriv_gains, device=self.device
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
        elif ctrl_type == "task_space_impedance":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = False
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = False
        elif ctrl_type == "operational_space_motion":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = False
        elif ctrl_type == "open_loop_force":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = False
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "open"
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.open_loop_force.force_ctrl_axes, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "closed_loop_force":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = False
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "closed"
            self.cfg_ctrl["wrench_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes, device=self.device
            ).repeat((self.num_envs, 1))
        elif ctrl_type == "hybrid_force_motion":
            self.cfg_ctrl["motor_ctrl_mode"] = "manual"
            self.cfg_ctrl["gain_space"] = "task"
            self.cfg_ctrl["do_motion_ctrl"] = True
            self.cfg_ctrl["task_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["task_deriv_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_inertial_comp"] = True
            self.cfg_ctrl["motion_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["do_force_ctrl"] = True
            self.cfg_ctrl["force_ctrl_method"] = "closed"
            self.cfg_ctrl["wrench_prop_gains"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                device=self.device,
            ).repeat((self.num_envs, 1))
            self.cfg_ctrl["force_ctrl_axes"] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                device=self.device,
            ).repeat((self.num_envs, 1))

        if add_to_stage:
            if self.cfg_ctrl["motor_ctrl_mode"] == "gym":
                for i in range(7):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_link{i}/panda_joint{i+1}"
                    )
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                    drive.GetStiffnessAttr().Set(
                        self.cfg_ctrl["joint_prop_gains"][0, i].item() * np.pi / 180
                    )
                    drive.GetDampingAttr().Set(
                        self.cfg_ctrl["joint_deriv_gains"][0, i].item() * np.pi / 180
                    )

                for i in range(2):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_hand/panda_finger_joint{i+1}"
                    )
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
                    drive.GetStiffnessAttr().Set(
                        self.cfg_ctrl["gripper_deriv_gains"][0, i].item()
                    )
                    drive.GetDampingAttr().Set(
                        self.cfg_ctrl["gripper_deriv_gains"][0, i].item()
                    )

            elif self.cfg_ctrl["motor_ctrl_mode"] == "manual":
                for i in range(7):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_link{i}/panda_joint{i+1}"
                    )
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

                for i in range(2):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path
                        + f"/franka/panda_hand/panda_finger_joint{i+1}"
                    )
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "None")
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets or DOF torques."""

        # Get desired Jacobian
        if self.cfg_ctrl["jacobian_type"] == "geometric":
            self.fingertip_midpoint_jacobian_tf = self.fingertip_midpoint_jacobian
        elif self.cfg_ctrl["jacobian_type"] == "analytic":
            self.fingertip_midpoint_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_quat,
                fingertip_jacobian=self.fingertip_midpoint_jacobian,
                num_envs=self.num_envs,
                device=self.device,
            )

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl["motor_ctrl_mode"] == "gym":
            self._set_dof_pos_target()
        elif self.cfg_ctrl["motor_ctrl_mode"] == "manual":
            self._set_dof_torque()

    def _set_dof_pos_target(self):
        """Set Franka DOF position target to move fingertips towards target pose."""

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            device=self.device,
        )

        self.frankas.set_joint_position_targets(positions=self.ctrl_target_dof_pos)

    def _set_dof_torque(self):
        """Set Franka DOF torque to move fingertips towards target pose."""

        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device,
        )

        self.frankas.set_joint_efforts(efforts=self.dof_torque)

    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        gravity = [0.0, 0.0, -gravity_mag]
        self._env._world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )

    def disable_gravity(self):
        """Disable gravity."""

        gravity = [0.0, 0.0, 0.0]
        self._env._world._physics_sim_view.set_gravity(
            carb.Float3(gravity[0], gravity[1], gravity[2])
        )
