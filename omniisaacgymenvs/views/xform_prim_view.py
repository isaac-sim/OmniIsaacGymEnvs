from typing import List, Optional, Tuple, Union

import carb
import numpy as np
import omni.kit.app
import torch
import warp as wp
from omni.isaac.core.prims import XFormPrimView as XFormPrimViewBase
from omni.isaac.core.materials.omni_glass import OmniGlass
from omni.isaac.core.materials.omni_pbr import OmniPBR
from omni.isaac.core.materials.preview_surface import PreviewSurface
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.simulation_context.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import (
    find_matching_prim_paths,
    get_prim_at_path,
    get_prim_parent,
    is_prim_non_root_articulation_link,
    is_prim_path_valid,
)
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.types import XFormPrimViewState
from omni.isaac.core.utils.xforms import get_local_pose, get_world_pose
from pxr import Gf, Usd, UsdGeom, UsdShade


class XFormPrimView(XFormPrimViewBase):
    """Provides high level functions to deal with a Xform prim view (one or many) and its descendants
    as well as its attributes/properties.

    This class wraps all matching Xforms found at the regex provided at the ``prim_paths_expr`` argument

    .. note::

        This class is inherited from the original XFormPrimView in omni.isaac.core. 
        The purpose for this class is to bypass some of the default state processing in the original
        class, since for RL purposes, we do not need to do them.

    Args:
        prim_paths_expr (str): prim paths regex to encapsulate all prims that match it.
                                example: "/World/Env[1-5]/Franka" will match /World/Env1/Franka,
                                /World/Env2/Franka..etc.
                                (a non regex prim path can also be used to encapsulate one Xform).
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "xform_prim_view".
        positions (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default positions in the world frame of the prim.
                                                        shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        translations (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default translations in the local frame of the prims
                                                        (with respect to its parent prims). shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        orientations (Optional[Union[np.ndarray, torch.Tensor]], optional):
                                                        default quaternion orientations in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (N, 4).
                                                        Defaults to None, which means left unchanged.
        scales (Optional[Union[np.ndarray, torch.Tensor]], optional): local scales to be applied to
                                                        the prim's dimensions. shape is (N, 3).
                                                        Defaults to None, which means left unchanged.
        visibilities (Optional[Union[np.ndarray, torch.Tensor]], optional): set to false for an invisible prim in
                                                                            the stage while rendering. shape is (N,).
                                                                            Defaults to None.
        reset_xform_properties (bool, optional): True if the prims don't have the right set of xform properties
                                                (i.e: translate, orient and scale) ONLY and in that order.
                                                Set this parameter to False if the object were cloned using using
                                                the cloner api in omni.isaac.cloner. Defaults to True.

    Raises:
        Exception: if translations and positions defined at the same time.
        Exception: No prim was matched using the prim_paths_expr provided.

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.stage as stage_utils
        >>> from omni.isaac.cloner import GridCloner
        >>> from omni.isaac.core.prims import XFormPrimView
        >>> from pxr import UsdGeom
        >>>
        >>> env_zero_path = "/World/envs/env_0"
        >>> num_envs = 5
        >>>
        >>> # load the Franka Panda robot USD file
        >>> stage_utils.add_reference_to_stage(usd_path, prim_path=f"{env_zero_path}/panda")  # /World/envs/env_0/panda
        >>>
        >>> # clone the environment (num_envs)
        >>> cloner = GridCloner(spacing=1.5)
        >>> cloner.define_base_env(env_zero_path)
        >>> UsdGeom.Xform.Define(stage_utils.get_current_stage(), env_zero_path)
        >>> env_pos = cloner.clone(
        ...     source_prim_path=env_zero_path,
        ...     prim_paths=cloner.generate_paths("/World/envs/env", num_envs),
        ...     copy_from_source=True
        ... )
        >>>
        >>> # wrap all Xforms
        >>> prims = XFormPrimView(prim_paths_expr="/World/envs/env.*", name="xform_view")
        >>> prims
        <omni.isaac.core.prims.xform_prim_view.XFormPrimView object at 0x7f8ffd22ebc0>
    """

    def __init__(
        self,
        prim_paths_expr: str,
        name: str = "xform_prim_view",
        positions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        scales: Optional[Union[np.ndarray, torch.Tensor]] = None,
        visibilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
        reset_xform_properties: bool = True,
    ) -> None:
        self._non_root_link = False
        self._prim_paths = find_matching_prim_paths(prim_paths_expr)
        if len(self._prim_paths) == 0:
            raise Exception(
                "Prim path expression {} is invalid, a prim matching the expression needs to created before wrapping it as view".format(
                    prim_paths_expr
                )
            )
        self._name = name
        self._count = len(self._prim_paths)
        self._prims = []
        self._regex_prim_paths = prim_paths_expr
        for prim_path in self._prim_paths:
            self._prims.append(get_prim_at_path(prim_path))

        if SimulationContext.instance() is not None:
            self._backend = SimulationContext.instance().backend
            self._device = SimulationContext.instance().device
            self._backend_utils = SimulationContext.instance().backend_utils
        else:
            import omni.isaac.core.utils.numpy as np_utils

            self._backend = "numpy"
            self._device = None
            self._backend_utils = np_utils

        self._default_state = None
        self._applied_visual_materials = [None] * self._count
        self._binding_apis = [None] * self._count
        self._non_root_link = is_prim_non_root_articulation_link(prim_path=self._prim_paths[0])

        return

    def post_reset(self) -> None:
        """Reset the prims to its default state (positions and orientations)

        Example:

        .. code-block:: python

            >>> prims.post_reset()
        """
        return