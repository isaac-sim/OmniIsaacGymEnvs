# Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim

## About this repository

This repository contains Reinforcement Learning examples that can be run with the latest release of [Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). RL examples are trained using PPO from [rl_games](https://github.com/Denys88/rl_games) library and examples are built on top of Isaac Sim's `omni.isaac.core` and `omni.isaac.gym` frameworks.

Please see [release notes](docs/release_notes.md) for the latest updates.

<img src="https://user-images.githubusercontent.com/34286328/171454189-6afafbff-bb61-4aac-b518-24646007cb9f.gif" width="300" height="150"/>&emsp;<img src="https://user-images.githubusercontent.com/34286328/184172037-cdad9ee8-f705-466f-bbde-3caa6c7dea37.gif" width="300" height="150"/>

<img src="https://user-images.githubusercontent.com/34286328/171454182-0be1b830-bceb-4cfd-93fb-e1eb8871ec68.gif" width="300" height="150"/>&emsp;<img src="https://user-images.githubusercontent.com/34286328/171454193-e027885d-1510-4ef4-b838-06b37f70c1c7.gif" width="300" height="150"/>

<img src="https://user-images.githubusercontent.com/34286328/184174894-03767aa0-936c-4bfe-bbe9-a6865f539bb4.gif" width="300" height="150"/>&emsp;<img src="https://user-images.githubusercontent.com/34286328/184168200-152567a8-3354-4947-9ae0-9443a56fee4c.gif" width="300" height="150"/>

<img src="https://user-images.githubusercontent.com/34286328/184176312-df7d2727-f043-46e3-b537-48a583d321b9.gif" width="300" height="150"/>&emsp;<img src="https://user-images.githubusercontent.com/34286328/184178817-9c4b6b3c-c8a2-41fb-94be-cfc8ece51d5d.gif" width="300" height="150"/>

<img src="https://user-images.githubusercontent.com/34286328/171454160-8cb6739d-162a-4c84-922d-cda04382633f.gif" width="300" height="150"/>&emsp;<img src="https://user-images.githubusercontent.com/34286328/171454176-ce08f6d0-3087-4ecc-9273-7d30d8f73f6d.gif" width="300" height="150"/>

<img src="https://user-images.githubusercontent.com/34286328/184170040-3f76f761-e748-452e-b8c8-3cc1c7c8cb98.gif" width="614" height="307"/>

## System Requirements

It is recommended to have at least 32GB RAM and a GPU with at least 12GB VRAM. For detailed system requirements, please visit the [Isaac Sim System Requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements) page. Please refer to the [Troubleshooting](docs/troubleshoot.md#memory-consumption) page for a detailed breakdown of memory consumption.

## Installation

Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install the latest Isaac Sim release. 

*Examples in this repository rely on features from the most recent Isaac Sim release. Please make sure to update any existing Isaac Sim build to the latest release version, 2023.1.1, to ensure examples work as expected.*

Once installed, this repository can be used as a python module, `omniisaacgymenvs`, with the python executable provided in Isaac Sim.

To install `omniisaacgymenvs`, first clone this repository:

```bash
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
```

Once cloned, locate the [python executable in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html). By default, this should be `python.sh`. We will refer to this path as `PYTHON_PATH`.

To set a `PYTHON_PATH` variable in the terminal that links to the python executable, we can run a command that resembles the following. Make sure to update the paths to your local path.

```
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
For IsaacSim Docker: alias PYTHON_PATH=/isaac-sim/python.sh
```

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`:

```bash
PYTHON_PATH -m pip install -e .
```

The following error may appear during the initial installation. This error is harmless and can be ignored.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```


### Running the examples

*Note: All commands should be executed from `OmniIsaacGymEnvs/omniisaacgymenvs`.*

To train your first policy, run:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Cartpole
```

An Isaac Sim app window should be launched. Once Isaac Sim initialization completes, the Cartpole scene will be constructed and simulation will start running automatically. The process will terminate once training finishes.

Note that by default, we show a Viewport window with rendering, which slows down training. You can choose to close the Viewport window during training for better performance. The Viewport window can be re-enabled by selecting `Window > Viewport` from the top menu bar.

To achieve maximum performance, launch training in `headless` mode as follows:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant headless=True
```

#### A Note on the Startup Time of the Simulation

Some of the examples could take a few minutes to load because the startup time scales based on the number of environments. The startup time will continually
be optimized in future releases.

### Extension Workflow

The extension workflow provides a simple user interface for creating and launching RL tasks. To launch Isaac Sim for the extension workflow, run:

```bash
./<isaac_sim_root>/isaac-sim.gym.sh --ext-folder </parent/directory/to/OIGE>
```

Note: `isaac_sim_root` should be located in the same directory as `python.sh`.

The UI window can be activated from `Isaac Examples > RL Examples` by navigating the top menu bar.
For more details on the extension workflow, please refer to the [documentation](docs/framework/extension_workflow.md).

### Loading trained models // Checkpoints

Checkpoints are saved in the folder `runs/EXPERIMENT_NAME/nn` where `EXPERIMENT_NAME` 
defaults to the task name, but can also be overridden via the `experiment` argument.

To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth
```

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64
```

Note that if there are special characters such as `[` or `=` in the checkpoint names, 
you will need to escape them and put quotes around the string. For example,
`checkpoint="runs/Ant/nn/last_Antep\=501rew\[5981.31\].pth"`

We provide pre-trained checkpoints on the [Nucleus](https://docs.omniverse.nvidia.com/nucleus/latest/index.html) server under `Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints`. Run the following command
to launch inference with pre-trained checkpoint:

Localhost (To set up localhost, please refer to the [Isaac Sim installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)):

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ant.pth test=True num_envs=64
```

Production server:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant checkpoint=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ant.pth test=True num_envs=64
```

When running with a pre-trained checkpoint for the first time, we will automatically download the checkpoint file to `omniisaacgymenvs/checkpoints`. For subsequent runs, we will re-use the file that has already been downloaded, and will not overwrite existing checkpoints with the same name in the `checkpoints` folder.

## Runing from Docker

Latest Isaac Sim Docker image can be found on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim). A utility script is provided at `docker/run_docker.sh` to help initialize this repository and launch the Isaac Sim docker container. The script can be run with:

```bash
./docker/run_docker.sh
```

Then, training can be launched from the container with:

```bash
/isaac-sim/python.sh scripts/rlgames_train.py headless=True task=Ant
```

To run the Isaac Sim docker with UI, use the following script:

```bash
./docker/run_docker_viewer.sh
```

Then, training can be launched from the container with:

```bash
/isaac-sim/python.sh scripts/rlgames_train.py task=Ant
```

To avoid re-installing OIGE each time a container is launched, we also provide a dockerfile that can be used to build an image with OIGE installed. To build the image, run:

```bash
docker build -t isaac-sim-oige -f docker/dockerfile .
```

Then, start a container with the built image:

```bash
./docker/run_dockerfile.sh
```

Then, training can be launched from the container with:

```bash
/isaac-sim/python.sh scripts/rlgames_train.py task=Ant headless=True
```

### Isaac Sim Automator

Cloud instances for AWS, Azure, or GCP can be setup using [IsaacSim Automator](https://github.com/NVIDIA-Omniverse/IsaacSim-Automator/tree/main#omniverse-isaac-gym).

## Livestream

OmniIsaacGymEnvs supports livestream through the [Omniverse Streaming Client](https://docs.omniverse.nvidia.com/app_streaming-client/app_streaming-client/overview.html). To enable this feature, add the commandline argument `enable_livestream=True`:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Ant headless=True enable_livestream=True
```

Connect from the Omniverse Streaming Client once the SimulationApp has been created. Note that enabling livestream is equivalent to training with the viewer enabled, thus the speed of training/inferencing will decrease compared to running in headless mode.


## Training Scripts

All scripts provided in `omniisaacgymenvs/scripts` can be launched directly with `PYTHON_PATH`.

To test out a task without RL in the loop, run the random policy script with:

```bash
PYTHON_PATH scripts/random_policy.py task=Cartpole
```

This script will sample random actions from the action space and apply these actions to your task without running any RL policies. Simulation should start automatically after launching the script, and will run indefinitely until terminated.


To run a simple form of PPO from `rl_games`, use the single-threaded training script:

```bash
PYTHON_PATH scripts/rlgames_train.py task=Cartpole
```

This script creates an instance of the PPO runner in `rl_games` and automatically launches training and simulation. Once training completes (the total number of iterations have been reached), the script will exit. If running inference with `test=True checkpoint=<path/to/checkpoint>`, the script will run indefinitely until terminated. Note that this script will have limitations on interaction with the UI.


### Configuration and command line arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config.
 
Common arguments for the training scripts are:

* `task=TASK` - Selects which task to use. Any of `AllegroHand`, `Ant`, `Anymal`, `AnymalTerrain`, `BallBalance`, `Cartpole`, `CartpoleCamera`, `Crazyflie`, `FactoryTaskNutBoltPick`, `FactoryTaskNutBoltPlace`, `FactoryTaskNutBoltScrew`, `FrankaCabinet`, `FrankaDeformable`, `Humanoid`, `Ingenuity`, `Quadcopter`, `ShadowHand`, `ShadowHandOpenAI_FF`, `ShadowHandOpenAI_LSTM` (these correspond to the config for each environment in the folder `omniisaacgymenvs/cfg/task`)
* `train=TRAIN` - Selects which training config to use. Will automatically default to the correct config for the environment (ie. `<TASK>PPO`).
* `num_envs=NUM_ENVS` - Selects the number of environments to use (overriding the default number of environments set in the task config).
* `seed=SEED` - Sets a seed value for randomization, and overrides the default seed in the task config
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `sim_device=SIM_DEVICE` - Device used for physics simulation. Set to `gpu` (default) to use GPU and to `cpu` for CPU.
* `device_id=DEVICE_ID` - Device ID for GPU to use for simulation and task. Defaults to `0`. This parameter will only be used if simulation runs on GPU.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and follows PyTorch-like device syntax.
* `multi_gpu=MULTI_GPU` - Whether to train using multiple GPUs. Defaults to `False`. Note that this option is only available with `rlgames_train.py`.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `enable_livestream=ENABLE_LIVESTREAM` - Whether to enable Omniverse streaming.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for. Reasonable defaults are provided for the provided environments.
* `warp=WARP` - If set to True, launch the task implemented with Warp backend (Note: not all tasks have a Warp implementation).
* `kit_app=KIT_APP` - Specifies the absolute path to the kit app file to be used.

Hydra also allows setting variables inside config files directly as command line arguments. As an example, to set the minibatch size for a rl_games training run, you can use `train.params.config.minibatch_size=64`. Similarly, variables in task configs can also be set. For example, `task.env.episodeLength=100`.

#### Hydra Notes

Default values for each of these are found in the `omniisaacgymenvs/cfg/config.yaml` file.

The way that the `task` and `train` portions of the config works are through the use of config groups. 
You can learn more about how these work [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
The actual configs for `task` are in `omniisaacgymenvs/cfg/task/<TASK>.yaml` and for `train` in `omniisaacgymenvs/cfg/train/<TASK>PPO.yaml`. 

In some places in the config you will find other variables referenced (for example,
 `num_actors: ${....task.env.numEnvs}`). Each `.` represents going one level up in the config hierarchy.
 This is documented fully [here](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation).

### Tensorboard

Tensorboard can be launched during training via the following command:
```bash
PYTHON_PATH -m tensorboard.main --logdir runs/EXPERIMENT_NAME/summaries
```

## WandB support

You can run (WandB)[https://wandb.ai/] with OmniIsaacGymEnvs by setting `wandb_activate=True` flag from the command line. You can set the group, name, entity, and project for the run by setting the `wandb_group`, `wandb_name`, `wandb_entity` and `wandb_project` arguments. Make sure you have WandB installed in the Isaac Sim Python executable with `PYTHON_PATH -m pip install wandb` before activating.


## Training with Multiple GPUs

To train with multiple GPUs, use the following command, where `--proc_per_node` represents the number of available GPUs:
```bash
PYTHON_PATH -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rlgames_train.py headless=True task=Ant multi_gpu=True
```

## Multi-Node Training

To train across multiple nodes/machines, it is required to launch an individual process on each node.
For the master node, use the following command, where `--proc_per_node` represents the number of available GPUs, and `--nnodes` represents the number of nodes:
```bash
PYTHON_PATH -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/rlgames_train.py headless=True task=Ant multi_gpu=True
```

Note that the port (`5555`) can be replaced with any other available port.

For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
```bash
PYTHON_PATH -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/rlgames_train.py headless=True task=Ant multi_gpu=True
```

For more details on multi-node training with PyTorch, please visit [here](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html). As mentioned in the PyTorch documentation, "multinode training is bottlenecked by inter-node communication latencies". When this latency is high, it is possible multi-node training will perform worse than running on a single node instance.

## Tasks

Source code for tasks can be found in `omniisaacgymenvs/tasks`. 

Each task follows the frameworks provided in `omni.isaac.core` and `omni.isaac.gym` in Isaac Sim.

Refer to [docs/framework/framework.md](docs/framework/framework.md) for how to create your own tasks.

Full details on each of the tasks available can be found in the [RL examples documentation](docs/examples/rl_examples.md).


## Demo

We provide an interactable demo based on the `AnymalTerrain` RL example. In this demo, you can click on any of 
the ANYmals in the scene to go into third-person mode and manually control the robot with your keyboard as follows:

- `Up Arrow`: Forward linear velocity command
- `Down Arrow`: Backward linear velocity command
- `Left Arrow`: Leftward linear velocity command
- `Right Arrow`: Rightward linear velocity command
- `Z`: Counterclockwise yaw angular velocity command
- `X`: Clockwise yaw angular velocity command
- `C`: Toggles camera view between third-person and scene view while maintaining manual control
- `ESC`: Unselect a selected ANYmal and yields manual control

Launch this demo with the following command. Note that this demo limits the maximum number of ANYmals in the scene to 128.

```
PYTHON_PATH scripts/rlgames_demo.py task=AnymalTerrain num_envs=64 checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/anymal_terrain.pth 
```

<img src="https://user-images.githubusercontent.com/34286328/184688654-6e7899b2-5847-4184-8944-2a96b129b1ff.gif" width="600" height="300"/>

