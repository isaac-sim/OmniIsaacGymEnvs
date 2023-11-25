# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import PhysxSchema

class FrankaMobile(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "franka",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
    
        self._usd_path = f"/home/nikepupu/Downloads/franka/ridgeback_franka.usd"
        # self._usd_path = f"omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.0/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths=[

            # base
            "world/dummy_base_prismatic_x_joint",
            "dummy_base_x/dummy_base_prismatic_y_joint",
            "dummy_base_y/dummy_base_revolute_z_joint",
            #arm
            "panda_link0/panda_joint1",
            "panda_link1/panda_joint2",
            "panda_link2/panda_joint3",
            "panda_link3/panda_joint4",
            "panda_link4/panda_joint5",
            "panda_link5/panda_joint6",
            "panda_link6/panda_joint7",
            "panda_hand/panda_finger_joint1",
            "panda_hand/panda_finger_joint2",
        ]

        #  actuator_groups={
        #  "base": ActuatorGroupCfg(
        #     dof_names=["base_y_base_x", "base_theta_base_y", "base_link_base_theta"],
        #     model_cfg=ImplicitActuatorCfg(velocity_limit=500.0, torque_limit=1000.0),
        #     control_cfg=ActuatorControlCfg(command_types=["v_abs"], stiffness={".*": 0.0}, damping={".*": 1e5}),
        # ),

        # "shoulder": ActuatorGroupCfg(
        #     dof_names=["Actuator[1-4]"],
        #     model_cfg=ImplicitActuatorCfg(velocity_limit=500.0, torque_limit=1000.0),
        #     control_cfg=ActuatorControlCfg(
        #         command_types=["p_abs"],
        #         stiffness={".*": 800.0},
        #         damping={".*": 40.0},
        #         dof_pos_offset={
        #             "Actuator1": 0.0,
        #             "Actuator2": 0.0,
        #             "Actuator3": 0.0,
        #             "Actuator4": 0.0,
        #         },
        #     ),
        # ),
        # "forearm": ActuatorGroupCfg(
        #     dof_names=["Actuator[5-7]"],
        #     model_cfg=ImplicitActuatorCfg(velocity_limit=500.0, torque_limit=1000.0),
        #     control_cfg=ActuatorControlCfg(
        #         command_types=["p_abs"],
        #         stiffness={".*": 800.0},
        #         damping={".*": 40.0},
        #         dof_pos_offset={"Actuator5": 0.0, "Actuator6": 0, "Actuator7": 0},
        #     ),
        # ),

        # drive_type = ["linear"] * 2 + ["angular"] * 10  
        # default_dof_pos = [math.degrees(x) for x in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02]]
        # stiffness = [800] * 3 +  [800] * 9
        # damping = [1500] * 3 + [600] * 7 + [100] * 2
        # max_force = [500.0, 500.0, 500] + [ 1000 ] * 9 #[100, 100, 87, 87, 87, 87, 12, 12, 12, 200, 200]
        # max_velocity = [20.0, 20.0, 20.0] + [200]*7 + [20]*2

        drive_type = ['linear'] * 2  + ['angular'] +   ["angular"] * 7 + ["linear"] * 2
        default_dof_pos = [math.degrees(x) for x in [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02]
        stiffness = [800]*3 + [400 * np.pi / 180] * 7 + [10000] * 2
        damping =  [200]*3 + [80 * np.pi / 180] * 7 + [100] * 2
        max_force = [200, 200, 100, 87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_velocity = [100 ] * 3 +  [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]

        for i, dof in enumerate(dof_paths):
            print(f"{self.prim_path}/{dof}")
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )
            print('Done')

    def set_kinova_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)

