# Copyright (c) 2018-2023, NVIDIA Corporation
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

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_nut_bolt.yaml.
"""


import hydra
import numpy as np
import torch

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.physx.scripts import physicsUtils, utils

from omniisaacgymenvs.robots.articulations.views.factory_franka_view import (
    FactoryFrankaView,
)
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)


class FactoryEnvNutBolt(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/FactoryEnvNutBolt.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_nut_bolt.yaml"
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]

        self._get_env_yaml_params()

    def set_up_scene(self, scene) -> None:
        """Import assets. Add to scene."""

        # Increase buffer size to prevent overflow for Place and Screw tasks
        physxSceneAPI = self.world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)

        self.import_franka_assets(add_to_stage=True)
        self.create_nut_bolt_material()
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        self._import_env_assets(add_to_stage=True)

        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.nuts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/nut/factory_nut.*",
            name="nuts_view",
            track_contact_forces=True,
        )
        self.bolts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/bolt/factory_bolt.*",
            name="bolts_view",
            track_contact_forces=True,
        )

        scene.add(self.nuts)
        scene.add(self.bolts)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

        return

    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""

        super().initialize_views(scene)

        self.import_franka_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)

        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        if scene.object_exists("nuts_view"):
            scene.remove_object("nuts_view", registry_only=True)
        if scene.object_exists("bolts_view"):
            scene.remove_object("bolts_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("fingertips_view"):
            scene.remove_object("fingertips_view", registry_only=True)

        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.nuts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/nut/factory_nut.*", name="nuts_view"
        )
        self.bolts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/bolt/factory_bolt.*", name="bolts_view"
        )

        scene.add(self.nuts)
        scene.add(self.bolts)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

    def create_nut_bolt_material(self):
        """Define nut and bolt material."""

        self.nutboltPhysicsMaterialPath = "/World/Physics_Materials/NutBoltMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.nutboltPhysicsMaterialPath,
            density=self.cfg_env.env.nut_bolt_density,
            staticFriction=self.cfg_env.env.nut_bolt_friction,
            dynamicFriction=self.cfg_env.env.nut_bolt_friction,
            restitution=0.0,
        )

    def _import_env_assets(self, add_to_stage=True):
        """Set nut and bolt asset options. Import assets."""

        self.nut_heights = []
        self.nut_widths_max = []
        self.bolt_widths = []
        self.bolt_head_heights = []
        self.bolt_shank_lengths = []
        self.thread_pitches = []

        assets_root_path = get_assets_root_path()

        for i in range(0, self._num_envs):
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_nut_bolt[subassembly])

            nut_translation = torch.tensor(
                [
                    0.0,
                    self.cfg_env.env.nut_lateral_offset,
                    self.cfg_base.env.table_height,
                ],
                device=self._device,
            )
            nut_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            nut_height = self.asset_info_nut_bolt[subassembly][components[0]]["height"]
            nut_width_max = self.asset_info_nut_bolt[subassembly][components[0]][
                "width_max"
            ]
            self.nut_heights.append(nut_height)
            self.nut_widths_max.append(nut_width_max)

            nut_file = (
                assets_root_path
                + self.asset_info_nut_bolt[subassembly][components[0]]["usd_path"]
            )

            if add_to_stage:
                add_reference_to_stage(nut_file, f"/World/envs/env_{i}" + "/nut")
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/nut",
                    translation=nut_translation,
                    orientation=nut_orientation,
                )

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/nut/factory_{components[0]}/collisions"
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/nut/factory_{components[0]}/collisions/mesh_0"
                    ),
                    self.nutboltPhysicsMaterialPath,
                )

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "nut",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/nut"),
                    self._sim_config.parse_actor_config("nut"),
                )

            bolt_translation = torch.tensor(
                [0.0, 0.0, self.cfg_base.env.table_height], device=self._device
            )
            bolt_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

            bolt_width = self.asset_info_nut_bolt[subassembly][components[1]]["width"]

            bolt_head_height = self.asset_info_nut_bolt[subassembly][components[1]][
                "head_height"
            ]
            bolt_shank_length = self.asset_info_nut_bolt[subassembly][components[1]][
                "shank_length"
            ]
            self.bolt_widths.append(bolt_width)
            self.bolt_head_heights.append(bolt_head_height)
            self.bolt_shank_lengths.append(bolt_shank_length)

            if add_to_stage:
                bolt_file = (
                    assets_root_path
                    + self.asset_info_nut_bolt[subassembly][components[1]]["usd_path"]
                )
                add_reference_to_stage(bolt_file, f"/World/envs/env_{i}" + "/bolt")
                XFormPrim(
                    prim_path=f"/World/envs/env_{i}" + "/bolt",
                    translation=bolt_translation,
                    orientation=bolt_orientation,
                )

                self._stage.GetPrimAtPath(
                    f"/World/envs/env_{i}" + f"/bolt/factory_{components[1]}/collisions"
                ).SetInstanceable(
                    False
                )  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f"/World/envs/env_{i}"
                        + f"/bolt/factory_{components[1]}/collisions/mesh_0"
                    ),
                    self.nutboltPhysicsMaterialPath,
                )

                # applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    "bolt",
                    self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/bolt"),
                    self._sim_config.parse_actor_config("bolt"),
                )

            thread_pitch = self.asset_info_nut_bolt[subassembly]["thread_pitch"]
            self.thread_pitches.append(thread_pitch)

        # For computing body COM pos
        self.nut_heights = torch.tensor(
            self.nut_heights, device=self._device
        ).unsqueeze(-1)
        self.bolt_head_heights = torch.tensor(
            self.bolt_head_heights, device=self._device
        ).unsqueeze(-1)

        # For setting initial state
        self.nut_widths_max = torch.tensor(
            self.nut_widths_max, device=self._device
        ).unsqueeze(-1)
        self.bolt_shank_lengths = torch.tensor(
            self.bolt_shank_lengths, device=self._device
        ).unsqueeze(-1)

        # For defining success or failure
        self.bolt_widths = torch.tensor(
            self.bolt_widths, device=self._device
        ).unsqueeze(-1)
        self.thread_pitches = torch.tensor(
            self.thread_pitches, device=self._device
        ).unsqueeze(-1)

    def refresh_env_tensors(self):
        """Refresh tensors."""

        # Nut tensors
        self.nut_pos, self.nut_quat = self.nuts.get_world_poses(clone=False)
        self.nut_pos -= self.env_pos

        self.nut_com_pos = fc.translate_along_local_z(
            pos=self.nut_pos,
            quat=self.nut_quat,
            offset=self.bolt_head_heights + self.nut_heights * 0.5,
            device=self.device,
        )
        self.nut_com_quat = self.nut_quat  # always equal

        nut_velocities = self.nuts.get_velocities(clone=False)
        self.nut_linvel = nut_velocities[:, 0:3]
        self.nut_angvel = nut_velocities[:, 3:6]

        self.nut_com_linvel = self.nut_linvel + torch.cross(
            self.nut_angvel, (self.nut_com_pos - self.nut_pos), dim=1
        )
        self.nut_com_angvel = self.nut_angvel  # always equal

        self.nut_force = self.nuts.get_net_contact_forces(clone=False)

        # Bolt tensors
        self.bolt_pos, self.bolt_quat = self.bolts.get_world_poses(clone=False)
        self.bolt_pos -= self.env_pos

        self.bolt_force = self.bolts.get_net_contact_forces(clone=False)
