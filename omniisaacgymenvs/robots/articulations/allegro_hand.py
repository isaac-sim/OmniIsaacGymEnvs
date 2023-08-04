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
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class AllegroHand(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "allegro_hand",
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
            self._usd_path = assets_root_path + "/Isaac/Robots/AllegroHand/allegro_hand_instanceable.usd"

        self._position = torch.tensor([0.0, 0.0, 0.5]) if translation is None else translation
        self._orientation = (
            torch.tensor([0.257551, 0.283045, 0.683330, -0.621782]) if orientation is None else orientation
        )

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_allegro_hand_properties(self, stage, allegro_hand_prim):
        for link_prim in allegro_hand_prim.GetChildren():
            if not (
                link_prim == stage.GetPrimAtPath("/allegro/Looks")
                or link_prim == stage.GetPrimAtPath("/allegro/root_joint")
            ):
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(link_prim)
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetEnableGyroscopicForcesAttr().Set(False)
                rb.GetAngularDampingAttr().Set(0.01)
                rb.GetMaxLinearVelocityAttr().Set(1000)
                rb.GetMaxAngularVelocityAttr().Set(64 / np.pi * 180)
                rb.GetMaxDepenetrationVelocityAttr().Set(1000)
                rb.GetMaxContactImpulseAttr().Set(1e32)

    def set_motor_control_mode(self, stage, allegro_hand_path):
        prim = stage.GetPrimAtPath(allegro_hand_path)
        self._set_joint_properties(stage, prim)

    def _set_joint_properties(self, stage, prim):
        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetStiffnessAttr().Set(3 * np.pi / 180)
            drive.GetDampingAttr().Set(0.1 * np.pi / 180)
            drive.GetMaxForceAttr().Set(0.5)
            revolute_joint = PhysxSchema.PhysxJointAPI.Get(stage, prim.GetPath())
            revolute_joint.GetJointFrictionAttr().Set(0.01)
        for child_prim in prim.GetChildren():
            self._set_joint_properties(stage, child_prim)
