## Troubleshooting

### Common Issues

#### Simulation
* When running simulation on GPU, be mindful of the dimensions of GPU buffers defined for physics. We have exposed GPU buffer size settings in the task confg files, which should be modified accordingly based on the number of actors and environments in the scene. If you see an error such as the following, try increasing the size of the corresponding buffer in the config `yaml` file.

```bash
PhysX error: the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity parameter to 3072, otherwise the simulation will miss interactions
```

* When running with the GPU pipeline, updates to states in the scene will not sync to USD. Therefore, values in the UI may appear wrong when simulation is running. Although objects may be updating in the Viewport, attribute values in the UI will not update along with them. Similarly, during simulation, any updates made through the USD APIs will not be synced with physics.
* To enable USD sync, please use the CPU pipeline with `pipeline=cpu` and disable fabric by setting `use_fabric: False` in the task config.


#### Load Time
* At initial load up of Isaac Sim, the process may appear to be frozen at an `app ready` message. This is normal and may take a few minutes for everything to load on the first run of Isaac Sim. Subsequent runs should be faster to start up, but may still take some time.
* Please note that once the Isaac Sim app loads, the environment creation time may scale linearly with the number of environments. Please expect a longer load time if running with thousands of environments or if each environment contains a larger number of assets. We are continually working on improving the time needed for this.
* When an instance of Isaac Sim is already running, launching another Isaac Sim instance in a different process may appear to hang at startup for the first time. Please be patient and give it some time as the second process will take longer to start up due to shader compilation.


#### Memory Consumption
* Memory consumption will increase with the number of environments and number of objects in the simulation scene. Below is a rough estimate of the amount of memory required for CPU and GPU for some of our example tasks and how they vary with the number of environments under the current settings defined in the task config files. If your machine is running out of memory, or if you are hitting Segmentation Faults, please try reducing the number of environments and the GPU buffer sizes in the task config file.

|     Task    | # of Envs | CPU Mem | GPU Mem |
|:-----------:|:---------:|:-------:|:-------:|
|   Humanoid  |   1024    | 4.85 GB | 3.55 GB |
|   Humanoid  |   2048    | 5.39 GB | 3.88 GB |
|   Humanoid  |   4096    | 6.55 GB | 4.46 GB |
| Shadow Hand |   1024    | 9.43 GB | 5.97 GB |
| Shadow Hand |   2048    | 10.5 GB | 6.74 GB |
| Shadow Hand |   4096    | 12.4 GB | 7.83 GB |
| Shadow Hand |   8192    | 16.3 GB | 9.90 GB |
| Shadow Hand |   16384   | 18.5 GB | 14.0 GB |


#### Interaction with the Environment
* During training mode, we have set `enable_scene_query_support=False` in our task config files by default. This will prevent certain interactions with the environments, such as raycasting and manipulating objects with the UI. If you wish to allow interaction during training, set `enable_scene_query_support=True`. This variable will always be set to `True` in inference/test mode.
* Please note that the single-threaded training script `rlgames_train.py` provides limited interaction with the UI during training. The main loop is controlled by the RL library and therefore, we have to terminate the process once the RL loop terminates. In both training and infrencing modes, once simulation is terminated from the UI, all views defined and used by the task will become invalid. Therefore, if we try to restart simulation from the UI, an error will occur.


#### RL Training
* rl-games requires `minibatch_size` defined in the training config to be a factor of `horizon_length * num_envs`. If this is not the case, you may see an assertion error `assert(self.batch_size % self.minibatch_size == 0)`. Please adjust the parameters in the training config `yaml` file accordingly.
* In the train configuration `yaml` file (*e.g.* [HumanoidPPO.yaml](../omniisaacgymenvs/cfg/train/HumanoidPPO.yaml)), setting the parameter `mixed_precision` to
`True` should only be used with gpu pipeline. It is recommended to set `mixed_precision` to `False` when using cpu pipeline to prevent crashes.
* If running with the multi-threaded environment wrapper class `VecEnvMT`, you may see a timeout error that looks something like `Getting states: timeout occurred.`. If you hit this error with your environment, try increasing the timeout variable in `VecEnvMT`, which can be passed as a parameter on the `initialize()` call. It may also be easier to debug an environment using the single-threaded class first to iron out any bugs.


### Known Issues
* Terminating a training or inferencing process launched from python with ctrl-c may result in a segmentation fault error if multiple ctrl-c events occurred. To prevent the error, please use a single ctrl-c command to terminate the process.
* SAC examples are currently broken due to a bug in rl-games v1.6.1. If you would like to run SAC, please use the latest master branch of rl-games: https://github.com/Denys88/rl_games.
* OmniIsaacGymEnvs versions 2022.2.1 and prior will no longer work with Isaac Sim version 2023.1.0 and later. For best compatibility, please update OmniIsaacGymEnvs to the same version as Isaac Sim.
* The following warning may appear at the beginning of training when loading assets from Nucleus: `[Warning] [omni.client.python] Detected a blocking function. This will cause hitches or hangs in the UI. Please switch to the async version`. The warning can be safely ignored.