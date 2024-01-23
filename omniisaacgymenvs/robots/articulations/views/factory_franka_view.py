from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class FactoryFrankaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FactoryFrankaView",
    ) -> None:
        """Initialize articulation view."""

        super().__init__(
            prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
        )

        self._hands = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_hand",
            name="hands_view",
            reset_xform_properties=False,
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_leftfinger",
            name="lfingers_view",
            reset_xform_properties=False,
            track_contact_forces=True,
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_rightfinger",
            name="rfingers_view",
            reset_xform_properties=False,
            track_contact_forces=True,
        )
        self._fingertip_centered = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/panda_fingertip_centered",
            name="fingertips_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        """Initialize physics simulation view."""

        super().initialize(physics_sim_view)
