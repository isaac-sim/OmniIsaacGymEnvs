from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class FrankaMobileView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FrankaMobileView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        # "_f85_instanceable/robotiq_arg2f_base_link/finger_joint",
        #     "_f85_instanceable/robotiq_arg2f_base_link/left_inner_knuckle_joint",
        #     "_f85_instanceable/robotiq_arg2f_base_link/right_inner_knuckle_joint",
        #     "_f85_instanceable/left_outer_finger/left_inner_finger_joint",
        #     "_f85_instanceable/robotiq_arg2f_base_link/right_outer_knuckle_joint",
        #     "_f85_instanceable/right_outer_finger/right_inner_finger_joint"
        # /mec_arm/right_inner_finger
        # self._hands = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/kinova/robotiq_arg2f_base_link", name="hands_view", reset_xform_properties=False
        # )
        # self._lfingers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/kinova/left_inner_finger_pad", name="lfingers_view", reset_xform_properties=False
        # )
        # self._rfingers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/kinova/right_inner_finger_pad",
        #     name="rfingers_view",
        #     reset_xform_properties=False,
        # )

        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/endeffector", name="hands_view", reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_leftfinger", name="lfingers_view", reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_rightfinger",
            name="rfingers_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]
        # self.gripper_indices = [10, 11, 12,13,14, 15]

    @property
    def gripper_indices(self):
        return self._gripper_indices
