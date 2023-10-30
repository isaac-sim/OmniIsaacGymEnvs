from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class KinovaMobileView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "KinovaMobileView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        # /mec_arm/right_inner_finger
        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/kinova/robotiq_85_base_link", name="hands_view", reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/kinova/left_inner_finger", name="lfingers_view", reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/kinova/right_inner_finger",
            name="rfingers_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("finger_joint"), 
                                 self.get_dof_index("left_inner_knuckle_joint"),
                                 self.get_dof_index("right_inner_knuckle_joint"),
                                 self.get_dof_index("left_inner_finger_joint"),
                                 self.get_dof_index("right_outer_knuckle_joint"),
                                 self.get_dof_index("right_inner_finger_joint")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
