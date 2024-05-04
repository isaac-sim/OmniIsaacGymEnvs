from typing import List, Optional, Sequence, Tuple, Union

# omniverse
import carb
import numpy as np
import omni.kit.app
import torch
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView as DeformablePrimViewBase
from omniisaacgymenvs.views.xform_prim_view import XFormPrimView

# isaac-core
from omni.isaac.core.simulation_context.simulation_context import SimulationContext
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade, Vt


class DeformablePrimView(DeformablePrimViewBase):
    """The view class for deformable prims."""

    def __init__(
        self,
        prim_paths_expr: str,
        deformable_materials: Optional[Union[np.ndarray, torch.Tensor]] = None,
        name: str = "deformable_prim_view",
        reset_xform_properties: bool = True,
        positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
        visibilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
        vertex_velocity_dampings: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sleep_dampings: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sleep_thresholds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        settling_thresholds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        self_collisions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        self_collision_filter_distances: Optional[Union[np.ndarray, torch.Tensor]] = None,
        solver_position_iteration_counts: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Provides high level functions to deal with deformable bodies (1 or more deformable bodies)
        as well as its attributes/ properties. This object wraps all matching deformable bodies found at the regex provided at the prim_paths_expr.

        Note: - if the underlying UsdGeom.Mesh.Get does not already have appropriate USD deformable body apis applied to it before init, this class will apply it.
        Args:
            prim_paths_expr (str): Prim paths regex to encapsulate all prims that match it.
            name (str): Shortname to be used as a key by Scene class.
            positions (Union[np.ndarray, torch.Tensor], optional): Default positions in the world frame of the prim. shape is (N, 3).
            translations (Union[np.ndarray, torch.Tensor], optional): Default translations in the local frame of the
                                                                        prims (with respect to its parent prims). shape is (N, 3).
            orientations (Union[np.ndarray, torch.Tensor], optional): Default quaternion orientations in the world/
                                                                        local frame of the prim (depends if translation or position is specified).
                                                                        quaternion is scalar-first (w, x, y, z). shape is (N, 4).
            scales (Union[np.ndarray, torch.Tensor], optional): Local scales to be applied to the prim's dimensions. shape is (N, 3).
            visibilities (Union[np.ndarray, torch.Tensor], optional): Set to false for an invisible prim in the stage while rendering. shape is (N,).
            vertex_velocity_dampings (Union[np.ndarray, torch.Tensor], optional): Velocity damping parameter controlling how much after every time step the nodal velocity is reduced
            sleep_dampings (Union[np.ndarray, torch.Tensor], optional): Damping value that damps the motion of bodies that move slow enough to be candidates for sleeping (see sleep_threshold)
            sleep_thresholds (Union[np.ndarray, torch.Tensor], optional): Threshold that defines the maximal magnitude of the linear motion a soft body can move in one second such that it can go to sleep in the next frame
            settling_thresholds (Union[np.ndarray, torch.Tensor], optional): Threshold that defines the maximal magnitude of the linear motion a fem body can move in one second before it becomes a candidate for sleeping
            self_collisions (Union[np.ndarray, torch.Tensor], optional): Enables the self collision for the deformable body based on the rest position distances.
            self_collision_filter_distances (Union[np.ndarray, torch.Tensor], optional): Penetration value that needs to get exceeded before contacts for self collision are generated. Will only have an effect if self collisions are enabled based on the rest position distances.
            solver_position_iteration_counts (Union[np.ndarray, torch.Tensor], optional): Number of the solver's positional iteration counts
        """

        self._physics_view = None
        self._device = None
        self._name = name
        XFormPrimView.__init__(
            self,
            prim_paths_expr=prim_paths_expr,
            name=name,
            positions=positions,
            translations=translations,
            orientations=orientations,
            scales=scales,
            visibilities=visibilities,
            reset_xform_properties=reset_xform_properties,
        )
        self._deformable_body_apis = [None] * self._count
        self._deformable_apis = [None] * self._count
        self._mass_apis = [None] * self._count
        self._applied_deformable_materials = [None] * self._count
        self._binding_apis = [None] * self._count

        if vertex_velocity_dampings is not None:
            self.set_vertex_velocity_dampings(vertex_velocity_dampings)
        if sleep_dampings is not None:
            self.set_sleep_dampings(sleep_dampings)
        if sleep_thresholds is not None:
            self.set_sleep_thresholds(sleep_thresholds)
        if settling_thresholds is not None:
            self.set_settling_thresholds(settling_thresholds)
        if self_collisions is not None:
            self.set_self_collisions(self_collisions)
        if self_collision_filter_distances is not None:
            self.set_self_collision_filter_distances(self_collision_filter_distances)
        if solver_position_iteration_counts is not None:
            self.set_solver_position_iteration_counts(solver_position_iteration_counts)

        timeline = omni.timeline.get_timeline_interface()
        self._invalidate_physics_handle_event = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._invalidate_physics_handle_callback
        )