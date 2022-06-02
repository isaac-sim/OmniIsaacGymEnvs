## Troubleshooting

### Common Issues

#### Simulation
* When running simulation on GPU, be mindful of the dimensions of GPU buffers defined for physics. We have exposed GPU buffer size settings in the task confg files, which should be modified accordingly based on the number of actors and environments in the scene. If you see an error such as the following, try increasing the size of the corresponding buffer in the config `yaml` file.

```bash
PhysX error: the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity parameter to 3072, otherwise the simulation will miss interactions
```

#### Load Time
* At initial load up of Isaac Sim, the process may appear to be frozen at an `app ready` message. This is normal and may take a few minutes for everything to load on the first run of Isaac Sim. Subsequent runs should be faster to start up, but may still take some time.
* Please note that once the Isaac Sim app loads, the environment creation time scales linearly with the number of environments. Please expect a longer load time if running with thousands of environments. We will be working on improving the time needed for this in future release.

#### Multi-GPU Machines
* The Isaac Sim GPU pipeline has mostly been tested with single-GPU setups. On machines with more than one GPU, it is possible to come across errors related to device mixing because we cannot guarantee the simulation device which physics runs on. To alleviate this issue, please try setting the `CUDA_VISIBLE_DEVICES=0` environment variable such that only one GPU is visible to Isaac Sim.
* Note that the `sim_device` command line argument in OmniIsaacGymEnvs controls the device of which the task runs on, but cannot control the device which physics simulation runs on. In most cases, the two devices should be identical in order for things to run.

#### Interaction with the Environment
* During training mode, we have set `enable_scene_query_support=False` in our task config files by default. This will prevent certain interactions with the environments in the UI. If you wish to allow interaction during training, set `enable_scene_query_support=True`. This variable will always be set to `True` in inference/test mode.
* Please note that the single-threaded training script `rlgames_train.py` provides limited interaction with the UI during training. The main loop is controlled by the RL library and therefore, we have to terminate the process once the RL loop terminates. In both training and infrencing modes, once simulation is terminated from the UI, all views defined and used by the task will become invalid. Therefore, if we try to restart simulation from the UI, an error will occur.
* The multi-threaded training script `rlgames_train_mt.py` will allow for better control in stopping and restarting simulation during training and inferencing.

#### RL Training
* rl-games requires `minibatch_size` defined in the training config to be a factor of `horizon_length * num_envs`. If this is not the case, you may see an assertion error `assert(self.batch_size % self.minibatch_size == 0)`. Please adjust the parameters in the training config `yaml` file accordingly.
* In the train configuration `yaml` file (*e.g.* [HumanoidPPO.yaml](../omniisaacgymenvs/cfg/train/HumanoidPPO.yaml)), setting the parameter `mixed_precision` to
`True` should only be used with gpu pipeline. It is recommended to set `mixed_precision` to `False` when using cpu pipeline to prevent crashes.
* If running with the multi-threaded environment wrapper class `VecEnvMT`, you may see a timeout error that looks something like `Getting states: timeout occurred.`. If you hit this error with your environment, try increasing the timeout variable in `VecEnvMT`, which can be passed as a parameter on the `initialize()` call. It may also be easier to debug an environment using the single-threaded class first to iron out any bugs.