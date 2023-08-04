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


from typing import Optional

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class ShadowHand(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "shadow_hand",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/ShadowHand/shadow_hand_instanceable.usd"

        self._position = torch.tensor([0.0, 0.0, 0.5]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_hand_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_hand_path):
        joints_config = {
            "robot0_WRJ1": {"stiffness": 5, "damping": 0.5, "max_force": 4.785},
            "robot0_WRJ0": {"stiffness": 5, "damping": 0.5, "max_force": 2.175},
            "robot0_FFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_RFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_LFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_THJ4": {"stiffness": 1, "damping": 0.1, "max_force": 2.3722},
            "robot0_THJ3": {"stiffness": 1, "damping": 0.1, "max_force": 1.45},
            "robot0_THJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ0": {"stiffness": 1, "damping": 0.1, "max_force": 0.81},
        }

        for joint_name, config in joints_config.items():
            set_drive(
                f"{self.prim_path}/joints/{joint_name}",
                "angular",
                "position",
                0.0,
                config["stiffness"] * np.pi / 180,
                config["damping"] * np.pi / 180,
                config["max_force"],
            )
