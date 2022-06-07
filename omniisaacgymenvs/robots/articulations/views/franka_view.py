
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class FrankaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FrankaView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        # create these views in the init and add them to the view
        self._grippers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/.*finger")
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_link7")
        self._lfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_leftfinger")
        self._rfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_rightfinger")

        self._grippers.initialize(physics_sim_view)
        self._hands.initialize(physics_sim_view)
        self._lfingers.initialize(physics_sim_view)
        self._rfingers.initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
    
