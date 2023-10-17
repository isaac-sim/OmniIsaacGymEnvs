## Reinforcement Learning with Vision in the Loop

Some reinforcement learning tasks can benefit from having image data in the pipeline by collecting sensor data from cameras to use as observations. However, high fidelity rendering can be expensive when scaled up towards thousands of environments during training.

Although Isaac Sim does not currently have the capability to scale towards thousands of environments, we are continually working on improvements to reach the goal. As a starting point, we are providing a simple example showcasing a proof-of-concept for reinforcement learning with vision in the loop.


### CartpoleCamera [cartpole_camera.py](../omniisaacgymenvs/tasks/cartpole_camera.py)

As an example showcasing the possiblity of reinforcmenet learning with vision in the loop, we provide a variation of the Cartpole task, which uses RGB image data as observations. This example
can be launched with command line argument `task=CartpoleCamera`. 

Config files used for this task are:

-   **Task config**: [CartpoleCamera.yaml](../omniisaacgymenvs/cfg/task/CartpoleCamera.yaml)
-   **rl_games training config**: [CartpoleCameraPPO.yaml](../omniisaacgymenvs/cfg/train/CartpoleCameraPPO.yaml)
  

### Working with Cameras

We have provided an individual app file `apps/omni.isaac.sim.python.gym.camera.kit`, designed specifically towards vision-based RL tasks. This app file provides necessary settings to enable multiple cameras to be rendered each frame. Additional settings are also applied to increase performance when rendering cameras across multiple environments.

In addition, the following settings can be added to the app file to increase performance at a cost of accuracy. By setting these flags to `false`, data collected from the cameras may have a 1 to 2 frame delay.

```
app.renderer.waitIdle=false
app.hydraEngine.waitIdle=false
```

We can also render in white-mode by adding the following line:

```
rtx.debugMaterialType=0
```

### Config Settings

In order for rendering to occur during training, tasks using camera rendering must have the `enable_cameras` flag set to `True` in the task config file. By default, the `omni.isaac.sim.python.gym.camera.kit` app file will be used automatically when `enable_cameras` is set to `True`. This flag is located in the task config file, under the `sim` section.

In addition, the `rendering_dt` parameter can be used to specify the rendering frequency desired. Similar to `dt` for physics simulation frequency, the `rendering_dt` specifies the amount of time in `s` between each rendering step. The `rendering_dt` should be larger or equal to the physics `dt`, and be a multiple of physics `dt`. Note that specifying the `controlFrequencyInv` flag will reduce the control frequency in terms of the physics simulation frequency.

For example, assume control frequency is 30hz, physics simulation frequency is 120 hz, and rendering frequency is 10hz. In the task config file, we can set `dt: 1/120`, `controlFrequencyInv: 4`, such that control is applied every 4 physics steps, and `rendering_dt: 1/10`. In this case, render data will only be updated once every 12 physics steps. Note that both `dt` and `rendering_dt` parameters are under the `sim` section of the config file, while `controlFrequencyInv` is under the `env` section.


### Environment Setup

To set up a task for vision-based RL, we will first need to add a camera to each environment in the scene and wrap it in a Replicator `render_product` to use the vectorized rendering API available in Replicator.

This can be done with the following code in `set_up_scene`:

```python
self.render_products = []
env_pos = self._env_pos.cpu()
for i in range(self._num_envs):
    camera = self.rep.create.camera(
        position=(-4.2 + env_pos[i][0], env_pos[i][1], 3.0), look_at=(env_pos[i][0], env_pos[i][1], 2.55))
    render_product = self.rep.create.render_product(camera, resolution=(self.camera_width, self.camera_height))
    self.render_products.append(render_product)
```

Next, we need to initialize Replicator and the PytorchListener, which will be used to collect rendered data.

```python
# start replicator to capture image data
self.rep.orchestrator._orchestrator._is_started = True

# initialize pytorch writer for vectorized collection
self.pytorch_listener = self.PytorchListener()
self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda")
self.pytorch_writer.attach(self.render_products)
```

Then, we can simply collect rendered data from each environment using a single API call:

```python
# retrieve RGB data from all render products
images = self.pytorch_listener.get_rgb_data()
```