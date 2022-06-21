
from typing import Optional
import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch

from pxr import PhysxSchema

class Anymal(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Anymal",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]
        """
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find nucleus server with /Isaac folder")
            self._usd_path = assets_root_path + "/Users/kellyg/Anymal/Anymal_Instanceable.usda"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = ["LF_HAA",
                           "LH_HAA",
                           "RF_HAA",
                           "RH_HAA",
                           "LF_HFE",
                           "LH_HFE",
                           "RF_HFE",
                           "RH_HFE",
                           "LF_KFE",
                           "LH_KFE",
                           "RF_KFE",
                           "RH_KFE"]

    @property
    def dof_names(self):
        return self._dof_names

    def initialize(self):
        self._knees = RigidPrimView(prim_paths_expr="/World/envs/*/anymal/*_SHANK", name="knees_view")

    def get_knee_transforms(self):
        return self._knees.get_world_poses()

    def set_anymal_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
