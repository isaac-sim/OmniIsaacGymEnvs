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
import datetime
import os
import queue
import threading
import traceback

import hydra
from omegaconf import DictConfig
from omni.isaac.gym.vec_env.vec_env_mt import TrainerMT
import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env_rlgames_mt import VecEnvRLGamesMT
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

        # ensure checkpoints can be specified as relative paths
        self._bad_checkpoint = False
        if self.cfg.checkpoint:
            self.cfg.checkpoint = retrieve_checkpoint_path(self.cfg.checkpoint)
            if not self.cfg.checkpoint:
                self._bad_checkpoint = True

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())

        # add evaluation parameters
        if self.cfg.evaluation:
            player_config = self.rlg_config_dict["params"]["config"].get("player", {})
            player_config["evaluation"] = True
            player_config["update_checkpoint_freq"] = 100
            player_config["dir_to_monitor"] = os.path.dirname(self.cfg.checkpoint)
            self.rlg_config_dict["params"]["config"]["player"] = player_config

        module_path = os.path.abspath(os.path.join(os.path.dirname(omniisaacgymenvs.__file__)))
        self.rlg_config_dict["params"]["config"]["train_dir"] = os.path.join(module_path, "runs")

        # load config
        runner.load(copy.deepcopy(self.rlg_config_dict))
        runner.reset()

        # dump config dict
        experiment_dir = os.path.join(module_path, "runs", self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.cfg.wandb_activate:
            # Make sure to install WandB if you actually use this.
            import wandb

            run_name = f"{self.cfg.wandb_name}_{time_str}"

            wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                entity=self.cfg.wandb_entity,
                config=self.cfg_dict,
                sync_tensorboard=True,
                id=run_name,
                resume="allow",
                monitor_gym=True,
            )

        runner.run(
            {"train": not self.cfg.test, "play": self.cfg.test, "checkpoint": self.cfg.checkpoint, "sigma": None}
        )

        if self.cfg.wandb_activate:
            wandb.finish()


class Trainer(TrainerMT):
    def __init__(self, trainer, env):
        self.ppo_thread = None
        self.action_queue = None
        self.data_queue = None

        self.trainer = trainer
        self.is_running = False
        self.env = env

        self.create_task()
        self.run()

    def create_task(self):
        self.trainer.launch_rlg_hydra(self.env)
        # task = initialize_task(self.trainer.cfg_dict, self.env, init_sim=False)
        self.task = self.env.task

    def run(self):
        self.is_running = True

        self.action_queue = queue.Queue(1)
        self.data_queue = queue.Queue(1)

        if "mt_timeout" in self.trainer.cfg_dict:
            self.env.initialize(self.action_queue, self.data_queue, self.trainer.cfg_dict["mt_timeout"])
        else:
            self.env.initialize(self.action_queue, self.data_queue)
        self.ppo_thread = PPOTrainer(self.env, self.task, self.trainer)
        self.ppo_thread.daemon = True
        self.ppo_thread.start()

    def stop(self):
        self.env.stop = True
        self.env.clear_queues()
        if self.action_queue:
            self.action_queue.join()
        if self.data_queue:
            self.data_queue.join()
        if self.ppo_thread:
            self.ppo_thread.join()

        self.action_queue = None
        self.data_queue = None
        self.ppo_thread = None
        self.is_running = False


class PPOTrainer(threading.Thread):
    def __init__(self, env, task, trainer):
        super().__init__()
        self.env = env
        self.task = task
        self.trainer = trainer

    def run(self):
        from omni.isaac.gym.vec_env import TaskStopException

        print("starting ppo...")

        try:
            self.trainer.run()
            # trainer finished - send stop signal to main thread
            self.env.should_run = False
            self.env.send_actions(None, block=False)
        except TaskStopException:
            print("Task Stopped!")
            self.env.should_run = False
            self.env.send_actions(None, block=False)
        except Exception as e:
            # an error occurred on the RL side - signal stop to main thread
            print(traceback.format_exc())
            self.env.should_run = False
            self.env.send_actions(None, block=False)
