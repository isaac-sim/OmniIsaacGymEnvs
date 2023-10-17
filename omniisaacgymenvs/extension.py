# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import inspect
import os
import traceback
import weakref
from abc import abstractmethod

import hydra
import omni.ext
import omni.timeline
import omni.ui as ui
import omni.usd
from hydra import compose, initialize
from omegaconf import OmegaConf
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
from omni.isaac.core.utils.torch.maths import set_seed
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.world import World
from omniisaacgymenvs.envs.vec_env_rlgames_mt import VecEnvRLGamesMT
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_train_mt import RLGTrainer, Trainer
from omniisaacgymenvs.utils.task_util import import_tasks, initialize_task
from omni.isaac.ui.callbacks import on_open_folder_clicked, on_open_IDE_clicked
from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import (
    btn_builder,
    dropdown_builder,
    get_style,
    int_builder,
    multi_btn_builder,
    multi_cb_builder,
    scrolling_frame_builder,
    setup_ui_headers,
    str_builder,
)
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

ext_instance = None


class RLExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._render_modes = ["Full render", "UI only", "None"]
        self._env = None
        self._task = None
        self._ext_id = ext_id
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        extension_path = ext_manager.get_extension_path(ext_id)
        self._ext_path = os.path.dirname(extension_path) if os.path.isfile(extension_path) else extension_path
        self._ext_file_path = os.path.abspath(__file__)

        self._initialize_task_list()
        self.start_extension(
            "",
            "",
            "RL Examples",
            "RL Examples",
            "",
            "A set of reinforcement learning examples.",
            self._ext_file_path,
        )

        self._task_initialized = False
        self._task_changed = False
        self._is_training = False
        self._render = True
        self._resume = False
        self._test = False
        self._evaluate = False
        self._checkpoint_path = ""

        self._timeline = omni.timeline.get_timeline_interface()
        self._viewport = get_active_viewport()
        self._viewport.updates_enabled = True

        global ext_instance
        ext_instance = self

    def _initialize_task_list(self):
        self._task_map, _ = import_tasks()
        self._task_list = list(self._task_map.keys())
        self._task_list.sort()
        self._task_list.remove("CartpoleCamera") # we cannot run camera-based training from extension workflow for now. it requires a specialized app file.
        self._task_name = self._task_list[0]
        self._parse_config(self._task_name)
        self._update_task_file_paths(self._task_name)

    def _update_task_file_paths(self, task):
        self._task_file_path = os.path.abspath(inspect.getfile(self._task_map[task]))
        self._task_cfg_file_path = os.path.join(os.path.dirname(self._ext_file_path), f"cfg/task/{task}.yaml")
        self._train_cfg_file_path = os.path.join(os.path.dirname(self._ext_file_path), f"cfg/train/{task}PPO.yaml")

    def _parse_config(self, task, num_envs=None, overrides=None):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="cfg")
        overrides_list = [f"task={task}"]
        if overrides is not None:
            overrides_list += overrides
        if num_envs is None:
            self._cfg = compose(config_name="config", overrides=overrides_list)
        else:
            self._cfg = compose(config_name="config", overrides=overrides_list + [f"num_envs={num_envs}"])
        self._cfg_dict = omegaconf_to_dict(self._cfg)
        self._sim_config = SimConfig(self._cfg_dict)

    def start_extension(
        self,
        menu_name: str,
        submenu_name: str,
        name: str,
        title: str,
        doc_link: str,
        overview: str,
        file_path: str,
        number_of_extra_frames=1,
        window_width=550,
        keep_window_open=False,
    ):
        window = ui.Workspace.get_window("Property")
        if window:
            window.visible = False
        window = ui.Workspace.get_window("Render Settings")
        if window:
            window.visible = False

        menu_items = [make_menu_item_description(self._ext_id, name, lambda a=weakref.proxy(self): a._menu_callback())]
        if menu_name == "" or menu_name is None:
            self._menu_items = menu_items
        elif submenu_name == "" or submenu_name is None:
            self._menu_items = [MenuItemDescription(name=menu_name, sub_menu=menu_items)]
        else:
            self._menu_items = [
                MenuItemDescription(
                    name=menu_name, sub_menu=[MenuItemDescription(name=submenu_name, sub_menu=menu_items)]
                )
            ]
        add_menu_items(self._menu_items, "Isaac Examples")

        self._task_dropdown = None
        self._cbs = None
        self._build_ui(
            name=name,
            title=title,
            doc_link=doc_link,
            overview=overview,
            file_path=file_path,
            number_of_extra_frames=number_of_extra_frames,
            window_width=window_width,
            keep_window_open=keep_window_open,
        )
        return

    def _build_ui(
        self, name, title, doc_link, overview, file_path, number_of_extra_frames, window_width, keep_window_open
    ):
        self._window = omni.ui.Window(
            name, width=window_width, height=0, visible=keep_window_open, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            self._main_stack = ui.VStack(spacing=5, height=0)
            with self._main_stack:
                setup_ui_headers(self._ext_id, file_path, title, doc_link, overview)
                self._controls_frame = ui.CollapsableFrame(
                    title="World Controls",
                    width=ui.Fraction(1),
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )

                with self._controls_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        with ui.HStack(style=get_style()):
                            with ui.VStack(style=get_style(), width=ui.Fraction(20)):
                                dict = {
                                    "label": "Select Task",
                                    "type": "dropdown",
                                    "default_val": 0,
                                    "items": self._task_list,
                                    "tooltip": "Select a task",
                                    "on_clicked_fn": self._on_task_select,
                                }
                                self._task_dropdown = dropdown_builder(**dict)
                            with ui.Frame(tooltip="Open Source Code"):
                                ui.Button(
                                    name="IconButton",
                                    width=20,
                                    height=20,
                                    clicked_fn=lambda: on_open_IDE_clicked(self._ext_path, self._task_file_path),
                                    style=get_style()["IconButton.Image::OpenConfig"],
                                    alignment=ui.Alignment.LEFT_CENTER,
                                    tooltip="Open in IDE",
                                )
                            with ui.Frame(tooltip="Open Task Config"):
                                ui.Button(
                                    name="IconButton",
                                    width=20,
                                    height=20,
                                    clicked_fn=lambda: on_open_IDE_clicked(self._ext_path, self._task_cfg_file_path),
                                    style=get_style()["IconButton.Image::OpenConfig"],
                                    alignment=ui.Alignment.LEFT_CENTER,
                                    tooltip="Open in IDE",
                                )
                            with ui.Frame(tooltip="Open Training Config"):
                                ui.Button(
                                    name="IconButton",
                                    width=20,
                                    height=20,
                                    clicked_fn=lambda: on_open_IDE_clicked(self._ext_path, self._train_cfg_file_path),
                                    style=get_style()["IconButton.Image::OpenConfig"],
                                    alignment=ui.Alignment.LEFT_CENTER,
                                    tooltip="Open in IDE",
                                )

                        dict = {
                            "label": "Number of environments",
                            "tooltip": "Enter the number of environments to construct",
                            "min": 0,
                            "max": 8192,
                            "default_val": self._cfg.task.env.numEnvs,
                        }
                        self._num_envs_int = int_builder(**dict)
                        dict = {
                            "label": "Load Environment",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Load Environment and Task",
                            "on_clicked_fn": self._on_load_world,
                        }
                        self._load_env_button = btn_builder(**dict)
                        dict = {
                            "label": "Rendering Mode",
                            "type": "dropdown",
                            "default_val": 0,
                            "items": self._render_modes,
                            "tooltip": "Select a rendering mode",
                            "on_clicked_fn": self._on_render_mode_select,
                        }
                        self._render_dropdown = dropdown_builder(**dict)
                        dict = {
                            "label": "Configure Training",
                            "count": 3,
                            "text": ["Resume from Checkpoint", "Test", "Evaluate"],
                            "default_val": [False, False, False],
                            "tooltip": [
                                "",
                                "Resume training from checkpoint",
                                "Play a trained policy",
                                "Evaluate a policy during training",
                            ],
                            "on_clicked_fn": [
                                self._on_resume_cb_update,
                                self._on_test_cb_update,
                                self._on_evaluate_cb_update,
                            ],
                        }
                        self._cbs = multi_cb_builder(**dict)
                        dict = {
                            "label": "Load Checkpoint",
                            "tooltip": "Enter path to checkpoint file",
                            "on_clicked_fn": self._on_checkpoint_update,
                        }
                        self._checkpoint_str = str_builder(**dict)
                        dict = {
                            "label": "Train/Test",
                            "count": 2,
                            "text": ["Start", "Stop"],
                            "tooltip": [
                                "",
                                "Launch new training/inference run",
                                "Terminate current training/inference run",
                            ],
                            "on_clicked_fn": [self._on_train, self._on_train_stop],
                        }
                        self._buttons = multi_btn_builder(**dict)

        return

    def create_task(self):
        headless = self._cfg.headless
        enable_viewport = "enable_cameras" in self._cfg.task.sim and self._cfg.task.sim.enable_cameras
        self._env = VecEnvRLGamesMT(
            headless=headless,
            sim_device=self._cfg.device_id,
            enable_livestream=self._cfg.enable_livestream,
            enable_viewport=enable_viewport,
            launch_simulation_app=False,
        )
        self._task = initialize_task(self._cfg_dict, self._env, init_sim=False)
        self._task_initialized = True

    def _on_task_select(self, value):
        if self._task_initialized and value != self._task_name:
            self._task_changed = True
        self._task_initialized = False
        self._task_name = value
        self._parse_config(self._task_name)
        self._num_envs_int.set_value(self._cfg.task.env.numEnvs)
        self._update_task_file_paths(self._task_name)

    def _on_render_mode_select(self, value):
        if value == self._render_modes[0]:
            self._viewport.updates_enabled = True
            window = ui.Workspace.get_window("Viewport")
            window.visible = True
            if self._env:
                self._env._update_viewport = True
                self._env._render_mode = 0
        elif value == self._render_modes[1]:
            self._viewport.updates_enabled = False
            window = ui.Workspace.get_window("Viewport")
            window.visible = False
            if self._env:
                self._env._update_viewport = False
                self._env._render_mode = 1
        elif value == self._render_modes[2]:
            self._viewport.updates_enabled = False
            window = ui.Workspace.get_window("Viewport")
            window.visible = False
            if self._env:
                self._env._update_viewport = False
                self._env._render_mode = 2

    def _on_render_cb_update(self, value):
        self._render = value
        print("updates enabled", value)
        self._viewport.updates_enabled = value
        if self._env:
            self._env._update_viewport = value
        if value:
            window = ui.Workspace.get_window("Viewport")
            window.visible = True
        else:
            window = ui.Workspace.get_window("Viewport")
            window.visible = False

    def _on_single_env_cb_update(self, value):
        visibility = "invisible" if value else "inherited"
        stage = omni.usd.get_context().get_stage()
        env_root = stage.GetPrimAtPath("/World/envs")
        if env_root.IsValid():
            for i, p in enumerate(env_root.GetChildren()):
                p.GetAttribute("visibility").Set(visibility)
            if value:
                stage.GetPrimAtPath("/World/envs/env_0").GetAttribute("visibility").Set("inherited")
                env_pos = self._task._env_pos[0].cpu().numpy().tolist()
                camera_pos = [env_pos[0] + 10, env_pos[1] + 10, 3]
                camera_target = [env_pos[0], env_pos[1], env_pos[2]]
            else:
                camera_pos = [10, 10, 3]
                camera_target = [0, 0, 0]
            camera_state = ViewportCameraState("/OmniverseKit_Persp", get_active_viewport())
            camera_state.set_position_world(Gf.Vec3d(*camera_pos), True)
            camera_state.set_target_world(Gf.Vec3d(*camera_target), True)

    def _on_test_cb_update(self, value):
        self._test = value
        if value is True and self._checkpoint_path.strip() == "":
            self._checkpoint_str.set_value(f"runs/{self._task_name}/nn/{self._task_name}.pth")

    def _on_resume_cb_update(self, value):
        self._resume = value
        if value is True and self._checkpoint_path.strip() == "":
            self._checkpoint_str.set_value(f"runs/{self._task_name}/nn/{self._task_name}.pth")

    def _on_evaluate_cb_update(self, value):
        self._evaluate = value

    def _on_checkpoint_update(self, value):
        self._checkpoint_path = value.get_value_as_string()

    async def _on_load_world_async(self, use_existing_stage):
        # initialize task if not initialized
        if not self._task_initialized or not omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs").IsValid():
            self._parse_config(task=self._task_name, num_envs=self._num_envs_int.get_value_as_int())
            self.create_task()
        else:
            # update config
            self._parse_config(task=self._task_name, num_envs=self._num_envs_int.get_value_as_int())
            self._task.update_config(self._sim_config)
            # clear scene
            # self._env._world.scene.clear()

        self._env._world._sim_params = self._sim_config.get_physics_params()
        await self._env._world.initialize_simulation_context_async()
        set_camera_view(eye=[10, 10, 3], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        if not use_existing_stage:
            # clear scene
            self._env._world.scene.clear()
            # clear environments added to world
            omni.usd.get_context().get_stage().RemovePrim("/World/collisions")
            omni.usd.get_context().get_stage().RemovePrim("/World/envs")
            # create scene
            await self._env._world.reset_async_set_up_scene()
            # update num_envs in envs
            self._env.update_task_params()
        else:
            self._task.initialize_views(self._env._world.scene)

    def _on_load_world(self):
        # stop simulation before updating stage
        self._timeline.stop()
        asyncio.ensure_future(self._on_load_world_async(use_existing_stage=False))

    def _on_train_stop(self):
        if self._task_initialized:
            asyncio.ensure_future(self._env._world.stop_async())

    async def _on_train_async(self, overrides=None):
        try:
            # initialize task if not initialized
            print("task initialized:", self._task_initialized)
            if not self._task_initialized:
                # if this is the first launch of the extension, we do not want to re-create stage if stage already exists
                use_existing_stage = False
                if omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs").IsValid():
                    use_existing_stage = True

                print(use_existing_stage)
                await self._on_load_world_async(use_existing_stage)
            # update config
            self._parse_config(task=self._task_name, num_envs=self._num_envs_int.get_value_as_int(), overrides=overrides)
            sim_config = SimConfig(self._cfg_dict)
            self._task.update_config(sim_config)

            cfg_dict = omegaconf_to_dict(self._cfg)

            # sets seed. if seed is -1 will pick a random one
            self._cfg.seed = set_seed(self._cfg.seed, torch_deterministic=self._cfg.torch_deterministic)
            cfg_dict["seed"] = self._cfg.seed

            self._checkpoint_path = self._checkpoint_str.get_value_as_string()
            if self._resume or self._test:
                self._cfg.checkpoint = self._checkpoint_path
            self._cfg.test = self._test
            self._cfg.evaluation = self._evaluate
            cfg_dict["checkpoint"] = self._cfg.checkpoint
            cfg_dict["test"] = self._cfg.test
            cfg_dict["evaluation"] = self._cfg.evaluation

            rlg_trainer = RLGTrainer(self._cfg, cfg_dict)
            if not rlg_trainer._bad_checkpoint:
                trainer = Trainer(rlg_trainer, self._env)

                await self._env._world.reset_async_no_set_up_scene()
                self._env._render_mode = self._render_dropdown.get_item_value_model().as_int
                await self._env.run(trainer)
                await omni.kit.app.get_app().next_update_async()
        except Exception as e:
            print(traceback.format_exc())
        finally:
            self._is_training = False

    def _on_train(self):
        # stop simulation if still running
        self._timeline.stop()

        self._on_render_mode_select(self._render_modes[self._render_dropdown.get_item_value_model().as_int])

        if not self._is_training:
            self._is_training = True
            asyncio.ensure_future(self._on_train_async())
        return

    def _menu_callback(self):
        self._window.visible = not self._window.visible
        return

    def _on_window(self, status):
        return

    def on_shutdown(self):
        self._extra_frames = []
        if self._menu_items is not None:
            self._sample_window_cleanup()
        self.shutdown_cleanup()
        global ext_instance
        ext_instance = None
        return

    def shutdown_cleanup(self):
        return

    def _sample_window_cleanup(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None
        self._menu_items = None
        self._buttons = None
        self._load_env_button = None
        self._task_dropdown = None
        self._cbs = None
        self._checkpoint_str = None
        return

def get_instance():
    return ext_instance
