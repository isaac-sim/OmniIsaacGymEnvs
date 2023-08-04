# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
from datetime import date
import sys
import unittest
import weakref

import omni.kit.test
from omni.kit.test import AsyncTestSuite
from omni.kit.test.async_unittest import AsyncTextTestRunner
import omni.ui as ui
from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import btn_builder
from omni.kit.menu.utils import MenuItemDescription, add_menu_items
import omni.timeline
import omni.usd

from omniisaacgymenvs import RLExtension, get_instance


class GymRLTests(omni.kit.test.AsyncTestCase):

    def __init__(self, *args, **kwargs):
        super(GymRLTests, self).__init__(*args, **kwargs)
        self.ext = get_instance()

    async def _train(self, task, load=True, experiment=None, max_iterations=None):
        task_idx = self.ext._task_list.index(task)
        self.ext._task_dropdown.get_item_value_model().set_value(task_idx)
        if load:
            self.ext._on_load_world()
            while True:
                _, files_loaded, total_files = omni.usd.get_context().get_stage_loading_status()
                if files_loaded or total_files:
                    await omni.kit.app.get_app().next_update_async()
                else:
                    break
            for _ in range(100):
                await omni.kit.app.get_app().next_update_async()
        self.ext._render_dropdown.get_item_value_model().set_value(2)

        overrides = None
        if experiment is not None:
            overrides = [f"experiment={experiment}"]
        if max_iterations is not None:
            if overrides is None:
                overrides = [f"max_iterations={max_iterations}"]
            else:
                overrides += [f"max_iterations={max_iterations}"]

        await self.ext._on_train_async(overrides=overrides)

    async def test_train(self):
        date_str = date.today()
        tasks = self.ext._task_list
        for task in tasks:
            await self._train(task, load=True, experiment=f"{task}_{date_str}")

    async def test_train_determinism(self):
        date_str = date.today()
        tasks = self.ext._task_list
        for task in tasks:
            for i in range(3):
                await self._train(task, load=(i==0), experiment=f"{task}_{date_str}_{i}", max_iterations=100)

class TestRunner():
    def __init__(self):
        self._build_ui()

    def _build_ui(self):
        menu_items = [make_menu_item_description("RL Examples Tests", "RL Examples Tests", lambda a=weakref.proxy(self): a._menu_callback())]
        add_menu_items(menu_items, "Isaac Examples")

        self._window = omni.ui.Window(
            "RL Examples Tests", width=250, height=0, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            main_stack = ui.VStack(spacing=5, height=0)
            with main_stack:
                dict = {
                    "label": "Run Tests",
                    "type": "button",
                    "text": "Run Tests",
                    "tooltip": "Run all tests",
                    "on_clicked_fn": self._run_tests,
                }
                btn_builder(**dict)

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _run_tests(self):
        loader = unittest.TestLoader()
        loader.SuiteClass = AsyncTestSuite

        test_suite = AsyncTestSuite()
        test_suite.addTests(loader.loadTestsFromTestCase(GymRLTests))

        test_runner = AsyncTextTestRunner(verbosity=2, stream=sys.stdout)

        async def single_run():
            await test_runner.run(test_suite)

        print("=======================================")
        print(f"Running Tests")
        print("=======================================")

        asyncio.ensure_future(single_run())

TestRunner()