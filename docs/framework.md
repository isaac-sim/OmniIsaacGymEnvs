## RL Framework

### Overview

Our RL examples are built on top of Isaac Sim's RL framework provided in `omni.isaac.gym`. Tasks are implemented following `omni.isaac.core`'s Task structure. PPO training is performed using the [rl_games](https://github.com/Denys88/rl_games) library, but we provide the flexibility to use other RL libraries for training. 

For a list of examples provided, refer to the
[RL List of Examples](rl_examples.md)


### Class Definition

The RL ecosystem can be viewed as three main pieces: the Task, the RL policy, and the Environment wrapper that provides an interface for communication between the task and the RL policy. 

#### Task
The Task class is where main task logic is implemented, such as computing observations and rewards. This is where we can collect states of actors in the scene and apply controls or actions to our actors. 

For convenience, we provide a base Task class, `RLTask`, which inherits from the `BaseTask` class in `omni.isaac.core`. This class is responsible for dealing with common configuration parsing, buffer initialization, and environment creation. Note that some config parameters and buffers in this class are specific to the rl_games library, and it is not necessary to inherit new tasks from `RLTask`.

A few key methods in `RLTask` include:
* `__init__(self, name: str, env: VecEnvBase, offset: np.ndarray = None)` - Parses config values common to all tasks and initializes action/observation spaces if not defined in the child class. Defines a GridCloner by default and creates a base USD scope for holding all environment prims. Can be called from child class.
* `set_up_scene(self, scene: Scene, replicate_physics=True, collision_filter_global_paths=[], filter_collisions=True)` - Adds ground plane and creates clones of environment 0 based on values specifid in config. Can be called from child class `set_up_scene()`.
* `pre_physics_step(self, actions: torch.Tensor)` - Takes in actions buffer from RL policy. Can be overriden by child class to process actions.
* `post_physics_step(self)` - Controls flow of RL data processing by triggering APIs to compute observations, retrieve states, compute rewards, resets, and extras. Will return observation, reward, reset, and extras buffers.

#### Environment Wrappers

As part of the RL framework in Isaac Sim, we have introduced environment wrapper classes in `omni.isaac.gym` for RL policies to communicate with simulation in Isaac Sim. This class provides a vectorized interface for common RL APIs used by `gym.Env` and can be easily extended towards RL libraries that require additional APIs. We show an example of this extension process in this repository, where we extend `VecEnvBase` as provided in `omni.isaac.gym` to include additional APIs required by the rl_games library.

Commonly used APIs provided by the base wrapper class `VecEnvBase` include:
* `render(self, mode: str = "human")` - renders the current frame
* `close(self)` - closes the simulator
* `seed(self, seed: int = -1)` - sets a seed. Use `-1` for a random seed.
* `step(self, actions: Union[np.ndarray, torch.Tensor])` - triggers task `pre_physics_step` with actions, steps simulation and renderer, computes observations, rewards, dones, and returns state buffers
* `reset(self)` - triggers task `reset()`, steps simulation, and re-computes observations

##### Multi-Threaded Environment Wrapper for Extension Workflows

`VecEnvBase` is a simple interface that’s designed to provide commonly used `gym.Env` APIs required by RL libraries. Users can create an instance of this class, attach your task to the interface, and provide your wrapper instance to the RL policy. Since the RL algorithm maintains the main loop of execution, interaction with the UI and environments in the scene can be limited and may interfere with the training loop.

We also provide another environment wrapper class called `VecEnvMT`, which is designed to isolate the RL policy in a new thread, separate from the main simulation and rendering thread. This class provides the same set of interface as `VecEnvBase`, but also provides threaded queues for sending and receiving actions and states between the RL policy and the task. In order to use this wrapper interface, users have to implement a `TrainerMT` class, which should implement a `run()` method that initiates the RL loop on a new thread. We show an example of this in OmniIsaacGymEnvs under `omniisaacgymenvs/utils/rlgames/rlgames_train_mt.py`. The setup for using `VecEnvMT` is more involved compared to the single-threaded `VecEnvBase` interface, but will allow users to have more control over starting and stopping the training loop through interaction with the UI.

Note that `VecEnvMT` has a timeout variable, which defaults to 90 seconds. If either the RL thread waiting for physics state exceeds the timeout amount or the simulation thread waiting for RL actions exceeds the timeout amount, the threaded queues will throw an exception and terminate training. For larger scenes that require longer simulation or training time, try increasing the timeout variable in `VecEnvMT` to prevent unnecessary timeouts. This can be done by passing in a `timeout` argument when calling `VecEnvMT.initialize()`.

This wrapper is currently only supported with the [extension workflow](extension_workflow.md).

### Creating New Examples

For simplicity, we will focus on using the single-threaded `VecEnvBase` interface in this tutorial.

To run any example, first make sure an instance of `VecEnvBase` or descendant of `VecEnvBase` is initialized.
This will be required as an argumet to our new Task. For example:

``` python
env = VecEnvBase(headless=False)
```

The headless parameter indicates whether a viewer should be created for visualizing results.

Then, create our task class, extending it from `RLTask`:

```python
class MyNewTask(RLTask):
    def __init__(
        self,
        name: str,                # name of the Task
        sim_config: SimConfig,    # SimConfig instance for parsing cfg
        env: VecEnvBase,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        ...
        self._num_observations = 4
        self._num_actions = 1

        # call parent class’s __init__
        RLTask.__init__(self, name, env)
```

The `__init__` method should take 4 arguments: 
* `name`: a string for the name of the task (required by BaseTask)
* `sim_config`: an instance of `SimConfig` used for config parsing, can be `None`. This object is created in `omniisaacgymenvs/utils/task_utils.py`.
* `env`: an instance of `VecEnvBase` or an inherited class of `VecEnvBase`
* `offset`: any offset required to place the `Task` in `World` (required by `BaseTask`)

In the `__init__` method of `MyNewTask`, we can populate any task-specific parameters, such as dimension of observations and actions, and retrieve data from config dictionaries. Make sure to make a call to `RLTask`’s `__init__` at the end of the method to perform additional data initialization.

Next, we can implement the methods required by the RL framework. These methods follow APIs defined in `omni.isaac.core` `BaseTask` class. Below is an example of a simple implementation for each method.

```python
def set_up_scene(self, scene: Scene) -> None:
    # implement environment setup here
    add_prim_to_stage(my_robot) # add a robot actor to the stage
    super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
    self._my_robots = ArticulationView(...) # create a view of robots
    scene.add(self._my_robots) # add view to scene for initialization

def post_reset(self):
    # implement any logic required for simulation on-start here
    pass

def pre_physics_step(self, actions: torch.Tensor) -> None:
    # implement logic to be performed before physics steps
    self.perform_reset()
    self.apply_action(actions)

def get_observations(self) -> dict:
    # implement logic to retrieve observation states
    self.obs_buf = self.compute_observations()

def calculate_metrics(self) -> None:
    # implement logic to compute rewards
    self.rew_buf = self.compute_rewards()

def is_done(self) -> None:
    # implement logic to update dones/reset buffer
    self.reset_buf = self.compute_resets()

```

To launch the new example from one of our training scripts, add `MyNewTask` to `omniisaacgymenvs/utils/task_util.py`. In `initialize_task()`, add an import to the `MyNewTask` class and add an instance to the `task_map` dictionary to register it into the command line parsing. 

To use the Hydra config parsing system, also add a task and train config files into `omniisaacgymenvs/cfg`. The config files should be named `cfg/task/MyNewTask.yaml` and `cfg/train/MyNewTaskPPO.yaml`.

Finally, we can launch `MyNewTask` with:

```bash
PYTHON_PATH random_policy.py task=MyNewTask
```

### Using a New RL Library

In this repository, we provide an example of extending Isaac Sim's environment wrapper classes to work with the rl_games library, which can be found at `omniisaacgymenvs/envs/vec_env_rlgames.py` and `omniisaacgymenvs/envs/vec_env_rlgames_mt.py`.

The first script, `omniisaacgymenvs/envs/vec_env_rlgames.py`, extends from `VecEnvBase`.

```python
from omni.isaac.gym.vec_env import VecEnvBase

class VecEnvRLGames(VecEnvBase):
```

One of the features in rl_games is the support for asymmetrical actor-critic policies, which requires a `states` buffer in addition to the `observations` buffer. Thus, we have overriden a few of the class in `VecEnvBase` to incorporate this requirement.

```python
def set_task(
        self, task, backend="numpy", sim_params=None, init_sim=True
    ) -> None:
    super().set_task(task, backend, sim_params, init_sim) # class VecEnvBase's set_task to register task to the environment instance

    # special variables required by rl_games
    self.num_states = self._task.num_states
    self.state_space = self._task.state_space

def step(self, actions):
    # we clamp the actions so that values are within a defined range
    actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

    # pass actions buffer to task for processing
    self._task.pre_physics_step(actions)

    # allow users to specify the control frequency through config
    for _ in range(self._task.control_frequency_inv):
        self._world.step(render=self._render)
        self.sim_frame_count += 1

    # compute new buffers
    self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()
    self._states = self._task.get_states() # special buffer required by rl_games
    
    # return buffers in format required by rl_games
    obs_dict = {"obs": self._obs, "states": self._states}

    return obs_dict, self._rew, self._resets, self._extras
```

Similarly, we also have a multi-threaded version of the rl_games environment wrapper implementation, `omniisaacgymenvs/envs/vec_env_rlgames_mt.py`. This class extends from `VecEnvMT` and `VecEnvRLGames`:

```python
from omni.isaac.gym.vec_env import VecEnvMT
from .vec_env_rlgames import VecEnvRLGames

class VecEnvRLGamesMT(VecEnvRLGames, VecEnvMT):
```

In this class, we also have a special method `_parse_data(self, data)`, which is required to be implemented to parse dictionary values passed through queues. Since multiple buffers of data are required by the RL policy, we concatenate all of the buffers in a single dictionary, and send that to the queue to be received by the RL thread.

```python
def _parse_data(self, data):
    self._obs = torch.clamp(data["obs"], -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
    self._rew = data["rew"].to(self._task.rl_device).clone()
    self._states = torch.clamp(data["states"], -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()
    self._resets = data["reset"].to(self._task.rl_device).clone()
    self._extras = data["extras"].copy()
```

### API Limitations

#### omni.isaac.core Setter APIs

Setter APIs in omni.isaac.core for ArticulationView, RigidPrimView, and RigidContactView should only be called once per simulation step for 
each view instance per API. This means that for use cases where multiple calls to the same setter API from the same view instance is required,
users will need to cache the states to be set for intermmediate calls, and make only one call to the setter API prior to stepping physics with
the complete buffer containing all cached states. 

If multiple calls to the same setter API from the same view object are made within the simulation step, 
subsequent calls will override the states that have been set by prior calls to the same API, 
voiding the previous calls to the API. The API can be called again once a simulation step is made.

For example, the below code will override states.

```python
my_view.set_world_poses(positions=[[0, 0, 1]], orientations=[[1, 0, 0, 0]], indices=[0])
# this call will void the previous call
my_view.set_world_poses(positions=[[0, 1, 1]], orientations=[[1, 0, 0, 0]], indices=[1])
my_world.step()
```

Instead, the below code should be used.

```python
my_view.set_world_poses(positions=[[0, 0, 1], [0, 1, 1]], orientations=[[1, 0, 0, 0], [1, 0, 0, 0]], indices=[0, 1])
my_world.step()
```

#### omni.isaac.core Getter APIs

Getter APIs for cloth simulation may return stale states when used with the GPU pipeline. This is because the physics simulation requires a simulation step
to occur in order to refresh the GPU buffers with new states. Therefore, when a getter API is called after a setter API before a 
simulation step, the states returned from the getter API may not reflect the values that were set using the setter API.

For example:

```python
my_view.set_world_positions(positions=[[0, 0, 1]], indices=[0])
# Values may be stale when called before step
positions = my_view.get_world_positions()    # positions may not match [[0, 0, 1]]
my_world.step()
# Values will be updated when called after step
positions = my_view.get_world_positions()    # positions will reflect the new states
```

#### Performing Resets

When resetting the states of actors, impulses generated by previous target or effort controls 
will continue to be carried over from the previous states in simulation.
Therefore, depending on the time step, the masses of the objects, and the magnitude of the impulses, 
the difference between the desired reset state and the observed first state after reset can be large.
To eliminate this issue, users should also reset any position/velocity targets or effort controllers
to the reset state or zero state when resetting actor states. For setting joint positions and velocities
using the omni.isaac.core ArticulationView APIs, position targets and velocity targets will 
automatically be set to the same states as joint positions and velocities.


#### Massless Links

It may be helpful in some scenarios to introduce dummy bodies into articulations for
retrieving transformations at certain locations of the articulation. Although it is possible
to introduce rigid bodies with no mass and colliders APIs and attach them to the articulation
with fixed joints, this can sometimes cause physics instabilities in simulation. To prevent 
instabilities from occurring, it is recommended to add a dummy geometry to the rigid body
and include both Mass and Collision APIs. The mass of the geometry can be set to a very
small value, such as 0.0001, to avoid modifying physical behaviors of the articulation.
Similarly, we can also disable collision on the Collision API of the geometry to preserve
contact behavior of the articulation.