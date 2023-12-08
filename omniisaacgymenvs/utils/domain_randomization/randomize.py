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


import copy

import numpy as np
import torch

import omni
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.extensions import enable_extension


class Randomizer:
    def __init__(self, main_config, task_config):
        self._cfg = task_config
        self._config = main_config

        self.randomize = False
        dr_config = self._cfg.get("domain_randomization", None)
        self.distributions = dict()
        self.active_domain_randomizations = dict()
        self._observations_dr_params = None
        self._actions_dr_params = None

        if dr_config is not None:
            randomize = dr_config.get("randomize", False)
            randomization_params = dr_config.get("randomization_params", None)
            if randomize and randomization_params is not None:
                self.randomize = True
                self.min_frequency = dr_config.get("min_frequency", 1)

                # import DR extensions
                enable_extension("omni.replicator.isaac")
                import omni.replicator.core as rep
                import omni.replicator.isaac as dr

                self.rep = rep
                self.dr = dr

    def apply_on_startup_domain_randomization(self, task):
        if self.randomize:
            torch.manual_seed(self._config["seed"])
            randomization_params = self._cfg["domain_randomization"]["randomization_params"]
            for opt in randomization_params.keys():
                if opt == "rigid_prim_views":
                    if randomization_params["rigid_prim_views"] is not None:
                        for view_name in randomization_params["rigid_prim_views"].keys():
                            if randomization_params["rigid_prim_views"][view_name] is not None:
                                for attribute, params in randomization_params["rigid_prim_views"][view_name].items():
                                    params = randomization_params["rigid_prim_views"][view_name][attribute]
                                    if attribute in ["scale", "mass", "density"] and params is not None:
                                        if "on_startup" in params.keys():
                                            if not set(
                                                ("operation", "distribution", "distribution_parameters")
                                            ).issubset(params["on_startup"]):
                                                raise ValueError(
                                                    f"Please ensure the following randomization parameters for {view_name} {attribute} "
                                                    + "on_startup are provided: operation, distribution, distribution_parameters."
                                                )
                                            view = task.world.scene._scene_registry.rigid_prim_views[view_name]
                                            if attribute == "scale":
                                                self.randomize_scale_on_startup(
                                                    view=view,
                                                    distribution=params["on_startup"]["distribution"],
                                                    distribution_parameters=params["on_startup"][
                                                        "distribution_parameters"
                                                    ],
                                                    operation=params["on_startup"]["operation"],
                                                    sync_dim_noise=True,
                                                )
                                            elif attribute == "mass":
                                                self.randomize_mass_on_startup(
                                                    view=view,
                                                    distribution=params["on_startup"]["distribution"],
                                                    distribution_parameters=params["on_startup"][
                                                        "distribution_parameters"
                                                    ],
                                                    operation=params["on_startup"]["operation"],
                                                )
                                            elif attribute == "density":
                                                self.randomize_density_on_startup(
                                                    view=view,
                                                    distribution=params["on_startup"]["distribution"],
                                                    distribution_parameters=params["on_startup"][
                                                        "distribution_parameters"
                                                    ],
                                                    operation=params["on_startup"]["operation"],
                                                )
                if opt == "articulation_views":
                    if randomization_params["articulation_views"] is not None:
                        for view_name in randomization_params["articulation_views"].keys():
                            if randomization_params["articulation_views"][view_name] is not None:
                                for attribute, params in randomization_params["articulation_views"][view_name].items():
                                    params = randomization_params["articulation_views"][view_name][attribute]
                                    if attribute in ["scale"] and params is not None:
                                        if "on_startup" in params.keys():
                                            if not set(
                                                ("operation", "distribution", "distribution_parameters")
                                            ).issubset(params["on_startup"]):
                                                raise ValueError(
                                                    f"Please ensure the following randomization parameters for {view_name} {attribute} "
                                                    + "on_startup are provided: operation, distribution, distribution_parameters."
                                                )
                                            view = task.world.scene._scene_registry.articulated_views[view_name]
                                            if attribute == "scale":
                                                self.randomize_scale_on_startup(
                                                    view=view,
                                                    distribution=params["on_startup"]["distribution"],
                                                    distribution_parameters=params["on_startup"][
                                                        "distribution_parameters"
                                                    ],
                                                    operation=params["on_startup"]["operation"],
                                                    sync_dim_noise=True,
                                                )
        else:
            dr_config = self._cfg.get("domain_randomization", None)
            if dr_config is None:
                raise ValueError("No domain randomization parameters are specified in the task yaml config file")
            randomize = dr_config.get("randomize", False)
            randomization_params = dr_config.get("randomization_params", None)
            if randomize == False or randomization_params is None:
                print("On Startup Domain randomization will not be applied.")

    def set_up_domain_randomization(self, task):
        if self.randomize:
            randomization_params = self._cfg["domain_randomization"]["randomization_params"]
            self.rep.set_global_seed(self._config["seed"])
            with self.dr.trigger.on_rl_frame(num_envs=self._cfg["env"]["numEnvs"]):
                for opt in randomization_params.keys():
                    if opt == "observations":
                        self._set_up_observations_randomization(task)
                    elif opt == "actions":
                        self._set_up_actions_randomization(task)
                    elif opt == "simulation":
                        if randomization_params["simulation"] is not None:
                            self.distributions["simulation"] = dict()
                            self.dr.physics_view.register_simulation_context(task.world)
                            for attribute, params in randomization_params["simulation"].items():
                                self._set_up_simulation_randomization(attribute, params)
                    elif opt == "rigid_prim_views":
                        if randomization_params["rigid_prim_views"] is not None:
                            self.distributions["rigid_prim_views"] = dict()
                            for view_name in randomization_params["rigid_prim_views"].keys():
                                if randomization_params["rigid_prim_views"][view_name] is not None:
                                    self.distributions["rigid_prim_views"][view_name] = dict()
                                    self.dr.physics_view.register_rigid_prim_view(
                                        rigid_prim_view=task.world.scene._scene_registry.rigid_prim_views[
                                            view_name
                                        ],
                                    )
                                    for attribute, params in randomization_params["rigid_prim_views"][
                                        view_name
                                    ].items():
                                        if attribute not in ["scale", "density"]:
                                            self._set_up_rigid_prim_view_randomization(view_name, attribute, params)
                    elif opt == "articulation_views":
                        if randomization_params["articulation_views"] is not None:
                            self.distributions["articulation_views"] = dict()
                            for view_name in randomization_params["articulation_views"].keys():
                                if randomization_params["articulation_views"][view_name] is not None:
                                    self.distributions["articulation_views"][view_name] = dict()
                                    self.dr.physics_view.register_articulation_view(
                                        articulation_view=task.world.scene._scene_registry.articulated_views[
                                            view_name
                                        ],
                                    )
                                    for attribute, params in randomization_params["articulation_views"][
                                        view_name
                                    ].items():
                                        if attribute not in ["scale"]:
                                            self._set_up_articulation_view_randomization(view_name, attribute, params)
            self.rep.orchestrator.run()
            if self._config.get("enable_recording", False):
                # we need to deal with initializing render product here because it has to be initialized after orchestrator.run.
                # otherwise, replicator will stop the simulation
                task._env.create_viewport_render_product(resolution=(task.viewport_camera_width, task.viewport_camera_height))
                if not task.is_extension:
                    task.world.render()
        else:
            dr_config = self._cfg.get("domain_randomization", None)
            if dr_config is None:
                raise ValueError("No domain randomization parameters are specified in the task yaml config file")
            randomize = dr_config.get("randomize", False)
            randomization_params = dr_config.get("randomization_params", None)
            if randomize == False or randomization_params is None:
                print("Domain randomization will not be applied.")

    def _set_up_observations_randomization(self, task):
        task.randomize_observations = True
        self._observations_dr_params = self._cfg["domain_randomization"]["randomization_params"]["observations"]
        if self._observations_dr_params is None:
            raise ValueError(f"Observations randomization parameters are not provided.")
        if "on_reset" in self._observations_dr_params.keys():
            if not set(("operation", "distribution", "distribution_parameters")).issubset(
                self._observations_dr_params["on_reset"].keys()
            ):
                raise ValueError(
                    f"Please ensure the following observations on_reset randomization parameters are provided: "
                    + "operation, distribution, distribution_parameters."
                )
            self.active_domain_randomizations[("observations", "on_reset")] = np.array(
                self._observations_dr_params["on_reset"]["distribution_parameters"]
            )
        if "on_interval" in self._observations_dr_params.keys():
            if not set(("frequency_interval", "operation", "distribution", "distribution_parameters")).issubset(
                self._observations_dr_params["on_interval"].keys()
            ):
                raise ValueError(
                    f"Please ensure the following observations on_interval randomization parameters are provided: "
                    + "frequency_interval, operation, distribution, distribution_parameters."
                )
            self.active_domain_randomizations[("observations", "on_interval")] = np.array(
                self._observations_dr_params["on_interval"]["distribution_parameters"]
            )
        self._observations_counter_buffer = torch.zeros(
            (self._cfg["env"]["numEnvs"]), dtype=torch.int, device=self._config["rl_device"]
        )
        self._observations_correlated_noise = torch.zeros(
            (self._cfg["env"]["numEnvs"], task.num_observations), device=self._config["rl_device"]
        )

    def _set_up_actions_randomization(self, task):
        task.randomize_actions = True
        self._actions_dr_params = self._cfg["domain_randomization"]["randomization_params"]["actions"]
        if self._actions_dr_params is None:
            raise ValueError(f"Actions randomization parameters are not provided.")
        if "on_reset" in self._actions_dr_params.keys():
            if not set(("operation", "distribution", "distribution_parameters")).issubset(
                self._actions_dr_params["on_reset"].keys()
            ):
                raise ValueError(
                    f"Please ensure the following actions on_reset randomization parameters are provided: "
                    + "operation, distribution, distribution_parameters."
                )
            self.active_domain_randomizations[("actions", "on_reset")] = np.array(
                self._actions_dr_params["on_reset"]["distribution_parameters"]
            )
        if "on_interval" in self._actions_dr_params.keys():
            if not set(("frequency_interval", "operation", "distribution", "distribution_parameters")).issubset(
                self._actions_dr_params["on_interval"].keys()
            ):
                raise ValueError(
                    f"Please ensure the following actions on_interval randomization parameters are provided: "
                    + "frequency_interval, operation, distribution, distribution_parameters."
                )
            self.active_domain_randomizations[("actions", "on_interval")] = np.array(
                self._actions_dr_params["on_interval"]["distribution_parameters"]
            )
        self._actions_counter_buffer = torch.zeros(
            (self._cfg["env"]["numEnvs"]), dtype=torch.int, device=self._config["rl_device"]
        )
        self._actions_correlated_noise = torch.zeros(
            (self._cfg["env"]["numEnvs"], task.num_actions), device=self._config["rl_device"]
        )

    def apply_observations_randomization(self, observations, reset_buf):
        env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._observations_counter_buffer[env_ids] = 0
        self._observations_counter_buffer += 1

        if "on_reset" in self._observations_dr_params.keys():
            observations[:] = self._apply_correlated_noise(
                buffer_type="observations",
                buffer=observations,
                reset_ids=env_ids,
                operation=self._observations_dr_params["on_reset"]["operation"],
                distribution=self._observations_dr_params["on_reset"]["distribution"],
                distribution_parameters=self._observations_dr_params["on_reset"]["distribution_parameters"],
            )

        if "on_interval" in self._observations_dr_params.keys():
            randomize_ids = (
                (self._observations_counter_buffer >= self._observations_dr_params["on_interval"]["frequency_interval"])
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            self._observations_counter_buffer[randomize_ids] = 0
            observations[:] = self._apply_uncorrelated_noise(
                buffer=observations,
                randomize_ids=randomize_ids,
                operation=self._observations_dr_params["on_interval"]["operation"],
                distribution=self._observations_dr_params["on_interval"]["distribution"],
                distribution_parameters=self._observations_dr_params["on_interval"]["distribution_parameters"],
            )
        return observations

    def apply_actions_randomization(self, actions, reset_buf):
        env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._actions_counter_buffer[env_ids] = 0
        self._actions_counter_buffer += 1

        if "on_reset" in self._actions_dr_params.keys():
            actions[:] = self._apply_correlated_noise(
                buffer_type="actions",
                buffer=actions,
                reset_ids=env_ids,
                operation=self._actions_dr_params["on_reset"]["operation"],
                distribution=self._actions_dr_params["on_reset"]["distribution"],
                distribution_parameters=self._actions_dr_params["on_reset"]["distribution_parameters"],
            )
        if "on_interval" in self._actions_dr_params.keys():
            randomize_ids = (
                (self._actions_counter_buffer >= self._actions_dr_params["on_interval"]["frequency_interval"])
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            self._actions_counter_buffer[randomize_ids] = 0
            actions[:] = self._apply_uncorrelated_noise(
                buffer=actions,
                randomize_ids=randomize_ids,
                operation=self._actions_dr_params["on_interval"]["operation"],
                distribution=self._actions_dr_params["on_interval"]["distribution"],
                distribution_parameters=self._actions_dr_params["on_interval"]["distribution_parameters"],
            )
        return actions

    def _apply_uncorrelated_noise(self, buffer, randomize_ids, operation, distribution, distribution_parameters):
        if distribution == "gaussian" or distribution == "normal":
            noise = torch.normal(
                mean=distribution_parameters[0],
                std=distribution_parameters[1],
                size=(len(randomize_ids), buffer.shape[1]),
                device=self._config["rl_device"],
            )
        elif distribution == "uniform":
            noise = (distribution_parameters[1] - distribution_parameters[0]) * torch.rand(
                (len(randomize_ids), buffer.shape[1]), device=self._config["rl_device"]
            ) + distribution_parameters[0]
        elif distribution == "loguniform" or distribution == "log_uniform":
            noise = torch.exp(
                (np.log(distribution_parameters[1]) - np.log(distribution_parameters[0]))
                * torch.rand((len(randomize_ids), buffer.shape[1]), device=self._config["rl_device"])
                + np.log(distribution_parameters[0])
            )
        else:
            print(f"The specified {distribution} distribution is not supported.")

        if operation == "additive":
            buffer[randomize_ids] += noise
        elif operation == "scaling":
            buffer[randomize_ids] *= noise
        else:
            print(f"The specified {operation} operation type is not supported.")
        return buffer

    def _apply_correlated_noise(self, buffer_type, buffer, reset_ids, operation, distribution, distribution_parameters):
        if buffer_type == "observations":
            correlated_noise_buffer = self._observations_correlated_noise
        elif buffer_type == "actions":
            correlated_noise_buffer = self._actions_correlated_noise

        if len(reset_ids) > 0:
            if distribution == "gaussian" or distribution == "normal":
                correlated_noise_buffer[reset_ids] = torch.normal(
                    mean=distribution_parameters[0],
                    std=distribution_parameters[1],
                    size=(len(reset_ids), buffer.shape[1]),
                    device=self._config["rl_device"],
                )
            elif distribution == "uniform":
                correlated_noise_buffer[reset_ids] = (
                    distribution_parameters[1] - distribution_parameters[0]
                ) * torch.rand(
                    (len(reset_ids), buffer.shape[1]), device=self._config["rl_device"]
                ) + distribution_parameters[
                    0
                ]
            elif distribution == "loguniform" or distribution == "log_uniform":
                correlated_noise_buffer[reset_ids] = torch.exp(
                    (np.log(distribution_parameters[1]) - np.log(distribution_parameters[0]))
                    * torch.rand((len(reset_ids), buffer.shape[1]), device=self._config["rl_device"])
                    + np.log(distribution_parameters[0])
                )
            else:
                print(f"The specified {distribution} distribution is not supported.")

        if operation == "additive":
            buffer += correlated_noise_buffer
        elif operation == "scaling":
            buffer *= correlated_noise_buffer
        else:
            print(f"The specified {operation} operation type is not supported.")
        return buffer

    def _set_up_simulation_randomization(self, attribute, params):
        if params is None:
            raise ValueError(f"Randomization parameters for simulation {attribute} is not provided.")
        if attribute in self.dr.SIMULATION_CONTEXT_ATTRIBUTES:
            self.distributions["simulation"][attribute] = dict()
            if "on_reset" in params.keys():
                if not set(("operation", "distribution", "distribution_parameters")).issubset(params["on_reset"]):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for simulation {attribute} on_reset are provided: "
                        + "operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[("simulation", attribute, "on_reset")] = np.array(
                    params["on_reset"]["distribution_parameters"]
                )
                kwargs = {"operation": params["on_reset"]["operation"]}
                self.distributions["simulation"][attribute]["on_reset"] = self._generate_distribution(
                    dimension=self.dr.physics_view._simulation_context_initial_values[attribute].shape[0],
                    view_name="simulation",
                    attribute=attribute,
                    params=params["on_reset"],
                )
                kwargs[attribute] = self.distributions["simulation"][attribute]["on_reset"]
                with self.dr.gate.on_env_reset():
                    self.dr.physics_view.randomize_simulation_context(**kwargs)
            if "on_interval" in params.keys():
                if not set(("frequency_interval", "operation", "distribution", "distribution_parameters")).issubset(
                    params["on_interval"]
                ):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for simulation {attribute} on_interval are provided: "
                        + "frequency_interval, operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[("simulation", attribute, "on_interval")] = np.array(
                    params["on_interval"]["distribution_parameters"]
                )
                kwargs = {"operation": params["on_interval"]["operation"]}
                self.distributions["simulation"][attribute]["on_interval"] = self._generate_distribution(
                    dimension=self.dr.physics_view._simulation_context_initial_values[attribute].shape[0],
                    view_name="simulation",
                    attribute=attribute,
                    params=params["on_interval"],
                )
                kwargs[attribute] = self.distributions["simulation"][attribute]["on_interval"]
                with self.dr.gate.on_interval(interval=params["on_interval"]["frequency_interval"]):
                    self.dr.physics_view.randomize_simulation_context(**kwargs)

    def _set_up_rigid_prim_view_randomization(self, view_name, attribute, params):
        if params is None:
            raise ValueError(f"Randomization parameters for rigid prim view {view_name} {attribute} is not provided.")
        if attribute in self.dr.RIGID_PRIM_ATTRIBUTES:
            self.distributions["rigid_prim_views"][view_name][attribute] = dict()
            if "on_reset" in params.keys():
                if not set(("operation", "distribution", "distribution_parameters")).issubset(params["on_reset"]):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for {view_name} {attribute} on_reset are provided: "
                        + "operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[("rigid_prim_views", view_name, attribute, "on_reset")] = np.array(
                    params["on_reset"]["distribution_parameters"]
                )
                kwargs = {"view_name": view_name, "operation": params["on_reset"]["operation"]}
                if attribute == "material_properties" and "num_buckets" in params["on_reset"].keys():
                    kwargs["num_buckets"] = params["on_reset"]["num_buckets"]

                self.distributions["rigid_prim_views"][view_name][attribute]["on_reset"] = self._generate_distribution(
                    dimension=self.dr.physics_view._rigid_prim_views_initial_values[view_name][attribute].shape[1],
                    view_name=view_name,
                    attribute=attribute,
                    params=params["on_reset"],
                )
                kwargs[attribute] = self.distributions["rigid_prim_views"][view_name][attribute]["on_reset"]
                with self.dr.gate.on_env_reset():
                    self.dr.physics_view.randomize_rigid_prim_view(**kwargs)
            if "on_interval" in params.keys():
                if not set(("frequency_interval", "operation", "distribution", "distribution_parameters")).issubset(
                    params["on_interval"]
                ):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for {view_name} {attribute} on_interval are provided: "
                        + "frequency_interval, operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[("rigid_prim_views", view_name, attribute, "on_interval")] = np.array(
                    params["on_interval"]["distribution_parameters"]
                )
                kwargs = {"view_name": view_name, "operation": params["on_interval"]["operation"]}
                if attribute == "material_properties" and "num_buckets" in params["on_interval"].keys():
                    kwargs["num_buckets"] = params["on_interval"]["num_buckets"]

                self.distributions["rigid_prim_views"][view_name][attribute][
                    "on_interval"
                ] = self._generate_distribution(
                    dimension=self.dr.physics_view._rigid_prim_views_initial_values[view_name][attribute].shape[1],
                    view_name=view_name,
                    attribute=attribute,
                    params=params["on_interval"],
                )
                kwargs[attribute] = self.distributions["rigid_prim_views"][view_name][attribute]["on_interval"]
                with self.dr.gate.on_interval(interval=params["on_interval"]["frequency_interval"]):
                    self.dr.physics_view.randomize_rigid_prim_view(**kwargs)
        else:
            raise ValueError(f"The attribute {attribute} for {view_name} is invalid for domain randomization.")

    def _set_up_articulation_view_randomization(self, view_name, attribute, params):
        if params is None:
            raise ValueError(f"Randomization parameters for articulation view {view_name} {attribute} is not provided.")
        if attribute in self.dr.ARTICULATION_ATTRIBUTES:
            self.distributions["articulation_views"][view_name][attribute] = dict()
            if "on_reset" in params.keys():
                if not set(("operation", "distribution", "distribution_parameters")).issubset(params["on_reset"]):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for {view_name} {attribute} on_reset are provided: "
                        + "operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[("articulation_views", view_name, attribute, "on_reset")] = np.array(
                    params["on_reset"]["distribution_parameters"]
                )
                kwargs = {"view_name": view_name, "operation": params["on_reset"]["operation"]}
                if attribute == "material_properties" and "num_buckets" in params["on_reset"].keys():
                    kwargs["num_buckets"] = params["on_reset"]["num_buckets"]

                self.distributions["articulation_views"][view_name][attribute][
                    "on_reset"
                ] = self._generate_distribution(
                    dimension=self.dr.physics_view._articulation_views_initial_values[view_name][attribute].shape[1],
                    view_name=view_name,
                    attribute=attribute,
                    params=params["on_reset"],
                )
                kwargs[attribute] = self.distributions["articulation_views"][view_name][attribute]["on_reset"]
                with self.dr.gate.on_env_reset():
                    self.dr.physics_view.randomize_articulation_view(**kwargs)
            if "on_interval" in params.keys():
                if not set(("frequency_interval", "operation", "distribution", "distribution_parameters")).issubset(
                    params["on_interval"]
                ):
                    raise ValueError(
                        f"Please ensure the following randomization parameters for {view_name} {attribute} on_interval are provided: "
                        + "frequency_interval, operation, distribution, distribution_parameters."
                    )
                self.active_domain_randomizations[
                    ("articulation_views", view_name, attribute, "on_interval")
                ] = np.array(params["on_interval"]["distribution_parameters"])
                kwargs = {"view_name": view_name, "operation": params["on_interval"]["operation"]}
                if attribute == "material_properties" and "num_buckets" in params["on_interval"].keys():
                    kwargs["num_buckets"] = params["on_interval"]["num_buckets"]

                self.distributions["articulation_views"][view_name][attribute][
                    "on_interval"
                ] = self._generate_distribution(
                    dimension=self.dr.physics_view._articulation_views_initial_values[view_name][attribute].shape[1],
                    view_name=view_name,
                    attribute=attribute,
                    params=params["on_interval"],
                )
                kwargs[attribute] = self.distributions["articulation_views"][view_name][attribute]["on_interval"]
                with self.dr.gate.on_interval(interval=params["on_interval"]["frequency_interval"]):
                    self.dr.physics_view.randomize_articulation_view(**kwargs)
        else:
            raise ValueError(f"The attribute {attribute} for {view_name} is invalid for domain randomization.")

    def _generate_distribution(self, view_name, attribute, dimension, params):
        dist_params = self._sanitize_distribution_parameters(attribute, dimension, params["distribution_parameters"])
        if params["distribution"] == "uniform":
            return self.rep.distribution.uniform(tuple(dist_params[0]), tuple(dist_params[1]))
        elif params["distribution"] == "gaussian" or params["distribution"] == "normal":
            return self.rep.distribution.normal(tuple(dist_params[0]), tuple(dist_params[1]))
        elif params["distribution"] == "loguniform" or params["distribution"] == "log_uniform":
            return self.rep.distribution.log_uniform(tuple(dist_params[0]), tuple(dist_params[1]))
        else:
            raise ValueError(
                f"The provided distribution for {view_name} {attribute} is not supported. "
                + "Options: uniform, gaussian/normal, loguniform/log_uniform"
            )

    def _sanitize_distribution_parameters(self, attribute, dimension, params):
        distribution_parameters = np.array(params)
        if distribution_parameters.shape == (2,):
            # if the user does not provide a set of parameters for each dimension
            dist_params = [[distribution_parameters[0]] * dimension, [distribution_parameters[1]] * dimension]
        elif distribution_parameters.shape == (2, dimension):
            # if the user provides a set of parameters for each dimension in the format [[...], [...]]
            dist_params = distribution_parameters.tolist()
        elif attribute in ["material_properties", "body_inertias"] and distribution_parameters.shape == (2, 3):
            # if the user only provides the parameters for one body in the articulation, assume the same parameters for all other links
            dist_params = [
                [distribution_parameters[0]] * (dimension // 3),
                [distribution_parameters[1]] * (dimension // 3),
            ]
        else:
            raise ValueError(
                f"The provided distribution_parameters for {view_name} {attribute} is invalid due to incorrect dimensions."
            )
        return dist_params

    def set_dr_distribution_parameters(self, distribution_parameters, *distribution_path):
        if distribution_path not in self.active_domain_randomizations.keys():
            raise ValueError(
                f"Cannot find a valid domain randomization distribution using the path {distribution_path}."
            )
        if distribution_path[0] == "observations":
            if len(distribution_parameters) == 2:
                self._observations_dr_params[distribution_path[1]]["distribution_parameters"] = distribution_parameters
            else:
                raise ValueError(
                    f"Please provide distribution_parameters for observations {distribution_path[1]} "
                    + "in the form of [dist_param_1, dist_param_2]"
                )
        elif distribution_path[0] == "actions":
            if len(distribution_parameters) == 2:
                self._actions_dr_params[distribution_path[1]]["distribution_parameters"] = distribution_parameters
            else:
                raise ValueError(
                    f"Please provide distribution_parameters for actions {distribution_path[1]} "
                    + "in the form of [dist_param_1, dist_param_2]"
                )
        else:
            replicator_distribution = self.distributions[distribution_path[0]][distribution_path[1]][
                distribution_path[2]
            ]
            if distribution_path[0] == "rigid_prim_views" or distribution_path[0] == "articulation_views":
                replicator_distribution = replicator_distribution[distribution_path[3]]
            if (
                replicator_distribution.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleUniform"
                or replicator_distribution.node.get_node_type().get_node_type()
                == "omni.replicator.core.OgnSampleLogUniform"
            ):
                dimension = len(self.dr.utils.get_distribution_params(replicator_distribution, ["lower"])[0])
                dist_params = self._sanitize_distribution_parameters(
                    distribution_path[-2], dimension, distribution_parameters
                )
                self.dr.utils.set_distribution_params(
                    replicator_distribution, {"lower": dist_params[0], "upper": dist_params[1]}
                )
            elif replicator_distribution.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleNormal":
                dimension = len(self.dr.utils.get_distribution_params(replicator_distribution, ["mean"])[0])
                dist_params = self._sanitize_distribution_parameters(
                    distribution_path[-2], dimension, distribution_parameters
                )
                self.dr.utils.set_distribution_params(
                    replicator_distribution, {"mean": dist_params[0], "std": dist_params[1]}
                )

    def get_dr_distribution_parameters(self, *distribution_path):
        if distribution_path not in self.active_domain_randomizations.keys():
            raise ValueError(
                f"Cannot find a valid domain randomization distribution using the path {distribution_path}."
            )
        if distribution_path[0] == "observations":
            return self._observations_dr_params[distribution_path[1]]["distribution_parameters"]
        elif distribution_path[0] == "actions":
            return self._actions_dr_params[distribution_path[1]]["distribution_parameters"]
        else:
            replicator_distribution = self.distributions[distribution_path[0]][distribution_path[1]][
                distribution_path[2]
            ]
            if distribution_path[0] == "rigid_prim_views" or distribution_path[0] == "articulation_views":
                replicator_distribution = replicator_distribution[distribution_path[3]]
            if (
                replicator_distribution.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleUniform"
                or replicator_distribution.node.get_node_type().get_node_type()
                == "omni.replicator.core.OgnSampleLogUniform"
            ):
                return self.dr.utils.get_distribution_params(replicator_distribution, ["lower", "upper"])
            elif replicator_distribution.node.get_node_type().get_node_type() == "omni.replicator.core.OgnSampleNormal":
                return self.dr.utils.get_distribution_params(replicator_distribution, ["mean", "std"])

    def get_initial_dr_distribution_parameters(self, *distribution_path):
        if distribution_path not in self.active_domain_randomizations.keys():
            raise ValueError(
                f"Cannot find a valid domain randomization distribution using the path {distribution_path}."
            )
        return self.active_domain_randomizations[distribution_path].copy()

    def _generate_noise(self, distribution, distribution_parameters, size, device):
        if distribution == "gaussian" or distribution == "normal":
            noise = torch.normal(
                mean=distribution_parameters[0], std=distribution_parameters[1], size=size, device=device
            )
        elif distribution == "uniform":
            noise = (distribution_parameters[1] - distribution_parameters[0]) * torch.rand(
                size, device=device
            ) + distribution_parameters[0]
        elif distribution == "loguniform" or distribution == "log_uniform":
            noise = torch.exp(
                (np.log(distribution_parameters[1]) - np.log(distribution_parameters[0]))
                * torch.rand(size, device=device)
                + np.log(distribution_parameters[0])
            )
        else:
            print(f"The specified {distribution} distribution is not supported.")
        return noise

    def randomize_scale_on_startup(self, view, distribution, distribution_parameters, operation, sync_dim_noise=True):
        scales = view.get_local_scales()
        if sync_dim_noise:
            dist_params = np.asarray(
                self._sanitize_distribution_parameters(attribute="scale", dimension=1, params=distribution_parameters)
            )
            noise = (
                self._generate_noise(distribution, dist_params.squeeze(), (view.count,), view._device).repeat(3, 1).T
            )
        else:
            dist_params = np.asarray(
                self._sanitize_distribution_parameters(attribute="scale", dimension=3, params=distribution_parameters)
            )
            noise = torch.zeros((view.count, 3), device=view._device)
            for i in range(3):
                noise[:, i] = self._generate_noise(distribution, dist_params[:, i], (view.count,), view._device)

        if operation == "additive":
            scales += noise
        elif operation == "scaling":
            scales *= noise
        elif operation == "direct":
            scales = noise
        else:
            print(f"The specified {operation} operation type is not supported.")
        view.set_local_scales(scales=scales)

    def randomize_mass_on_startup(self, view, distribution, distribution_parameters, operation):
        if isinstance(view, omni.isaac.core.prims.RigidPrimView) or isinstance(view, RigidPrimView):
            masses = view.get_masses()
            dist_params = np.asarray(
                self._sanitize_distribution_parameters(
                    attribute=f"{view.name} mass", dimension=1, params=distribution_parameters
                )
            )
            noise = self._generate_noise(distribution, dist_params.squeeze(), (view.count,), view._device)
            set_masses = view.set_masses

        if operation == "additive":
            masses += noise
        elif operation == "scaling":
            masses *= noise
        elif operation == "direct":
            masses = noise
        else:
            print(f"The specified {operation} operation type is not supported.")
        set_masses(masses)

    def randomize_density_on_startup(self, view, distribution, distribution_parameters, operation):
        if isinstance(view, omni.isaac.core.prims.RigidPrimView) or isinstance(view, RigidPrimView):
            densities = view.get_densities()
            dist_params = np.asarray(
                self._sanitize_distribution_parameters(
                    attribute=f"{view.name} density", dimension=1, params=distribution_parameters
                )
            )
            noise = self._generate_noise(distribution, dist_params.squeeze(), (view.count,), view._device)
            set_densities = view.set_densities

        if operation == "additive":
            densities += noise
        elif operation == "scaling":
            densities *= noise
        elif operation == "direct":
            densities = noise
        else:
            print(f"The specified {operation} operation type is not supported.")
        set_densities(densities)
