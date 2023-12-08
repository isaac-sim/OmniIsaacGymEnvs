## Reinforcement Learning Examples

We introduce the following reinforcement learning examples that are implemented using 
Isaac Sim's RL framework. 

Pre-trained checkpoints can be found on the Nucleus server. To set up localhost, please refer to the [Isaac Sim installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).

*Note: All commands should be executed from `omniisaacgymenvs/omniisaacgymenvs`.*

- [Reinforcement Learning Examples](#reinforcement-learning-examples)
  - [Cartpole cartpole.py](#cartpole-cartpolepy)
  - [Ant ant.py](#ant-antpy)
  - [Humanoid humanoid.py](#humanoid-humanoidpy)
  - [Shadow Hand Object Manipulation shadow_hand.py](#shadow-hand-object-manipulation-shadow_handpy)
    - [OpenAI Variant](#openai-variant)
    - [LSTM Training Variant](#lstm-training-variant)
  - [Allegro Hand Object Manipulation allegro_hand.py](#allegro-hand-object-manipulation-allegro_handpy)
  - [ANYmal anymal.py](#anymal-anymalpy)
  - [Anymal Rough Terrain anymal_terrain.py](#anymal-rough-terrain-anymal_terrainpy)
  - [NASA Ingenuity Helicopter ingenuity.py](#nasa-ingenuity-helicopter-ingenuitypy)
  - [Quadcopter quadcopter.py](#quadcopter-quadcopterpy)
  - [Crazyflie crazyflie.py](#crazyflie-crazyfliepy)
  - [Ball Balance ball_balance.py](#ball-balance-ball_balancepy)
  - [Franka Cabinet franka_cabinet.py](#franka-cabinet-franka_cabinetpy)
  - [Franka Deformable franka_deformable.py](#franka-deformablepy)
  - [Factory: Fast Contact for Robotic Assembly](#factory-fast-contact-for-robotic-assembly)


### Cartpole [cartpole.py](../../omniisaacgymenvs/tasks/cartpole.py)

Cartpole is a simple example that demonstrates getting and setting usage of DOF states using 
`ArticulationView` from `omni.isaac.core`. The goal of this task is to move a cart horizontally
such that the pole, which is connected to the cart via a revolute joint, stays upright.

Joint positions and joint velocities are retrieved using `get_joint_positions` and 
`get_joint_velocities` respectively, which are required in computing observations. Actions are 
applied onto the cartpoles via `set_joint_efforts`. Cartpoles are reset by using `set_joint_positions` 
and `set_joint_velocities`. 

Training can be launched with command line argument `task=Cartpole`.
Training using the Warp backend can be launched with `task=Cartpole warp=True`.

Running inference with pre-trained model can be launched with command line argument `task=Cartpole test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/cartpole.pth`

Config files used for this task are:

-   **Task config**: [Cartpole.yaml](../../omniisaacgymenvs/cfg/task/Cartpole.yaml)
-   **rl_games training config**: [CartpolePPO.yaml](../../omniisaacgymenvs/cfg/train/CartpolePPO.yaml)

#### CartpoleCamera [cartpole_camera.py](../../omniisaacgymenvs/tasks/cartpole_camera.py)

A variation of the Cartpole task showcases the usage of RGB image data as observations. This example
can be launched with command line argument `task=CartpoleCamera`. Note that to use camera data as
observations, `enable_cameras` must be set to `True` in the task config file. In addition, the example must be run with the `omni.isaac.sim.python.gym.camera.kit` app file provided under `apps`, which applies necessary settings to enable camera training. By default, this app file will be used automatically when `enable_cameras` is set to `True`. Due to this limitation, this 
example is currently not available in the extension workflow.

Config files used for this task are:

-   **Task config**: [CartpoleCamera.yaml](../../omniisaacgymenvs/cfg/task/CartpoleCamera.yaml)
-   **rl_games training config**: [CartpoleCameraPPO.yaml](../../omniisaacgymenvs/cfg/train/CartpoleCameraPPO.yaml)

For more details on training with camera data, please visit [here](training_with_camera.md).
  
<img src="https://user-images.githubusercontent.com/34286328/171454189-6afafbff-bb61-4aac-b518-24646007cb9f.gif" width="300" height="150"/>

### Ant [ant.py](../../omniisaacgymenvs/tasks/ant.py)

Ant is an example of a simple locomotion task. The goal of this task is to train 
quadruped robots (ants) to run forward as fast as possible. This example inherets 
from [LocomotionTask](../../omniisaacgymenvs/tasks/shared/locomotion.py), 
which is a shared class between this example and the humanoid example; this simplifies
implementations for both environemnts since they compute rewards, observations, 
and resets in a similar manner. This framework allows us to easily switch between
robots used in the task.

The Ant task includes more examples of utilizing `ArticulationView` from `omni.isaac.core`, which
provides various functions to get and set both DOF states and articulation root states 
in a tensorized fashion across all of the actors in the environment. `get_world_poses`, 
 `get_linear_velocities`, and `get_angular_velocities`, can be used to determine whether the 
ants have been moving towards the desired direction and whether they have fallen or flipped over.
Actions are applied onto the ants via `set_joint_efforts`, which moves the ants by setting
torques to the DOFs. 

Note that the previously used force sensors and `get_force_sensor_forces` API are now deprecated.
Force sensors can now be retrieved directly using `get_measured_joint_forces` from `ArticulationView`. 

Training with PPO can be launched with command line argument `task=Ant`. 
Training with SAC with command line arguments `task=AntSAC train=AntSAC`.
Training using the Warp backend can be launched with `task=Ant warp=True`.

Running inference with pre-trained model can be launched with command line argument `task=Ant test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ant.pth`

Config files used for this task are:

-   **PPO task config**: [Ant.yaml](../../omniisaacgymenvs/cfg/task/Ant.yaml)
-   **rl_games PPO training config**: [AntPPO.yaml](../../omniisaacgymenvs/cfg/train/AntPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454182-0be1b830-bceb-4cfd-93fb-e1eb8871ec68.gif" width="300" height="150"/>



### Humanoid [humanoid.py](../../omniisaacgymenvs/tasks/humanoid.py)

Humanoid is another environment that uses 
[LocomotionTask](../../omniisaacgymenvs/tasks/shared/locomotion.py). It is conceptually
very similar to the Ant example, where the goal for the humanoid is to run forward
as fast as possible.

Training can be launched with command line argument `task=Humanoid`. 
Training with SAC with command line arguments `task=HumanoidSAC train=HumanoidSAC`.
Training using the Warp backend can be launched with `task=Humanoid warp=True`.

Running inference with pre-trained model can be launched with command line argument `task=Humanoid test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/humanoid.pth`

Config files used for this task are:

-   **PPO task config**: [Humanoid.yaml](../../omniisaacgymenvs/cfg/task/Humanoid.yaml)
-   **rl_games PPO training config**: [HumanoidPPO.yaml](../../omniisaacgymenvs/cfg/train/HumanoidPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454193-e027885d-1510-4ef4-b838-06b37f70c1c7.gif" width="300" height="150"/>


### Shadow Hand Object Manipulation [shadow_hand.py](../../omniisaacgymenvs/tasks/shadow_hand.py)

The Shadow Hand task is an example of a challenging dexterity manipulation task with complex contact 
dynamics. It resembles OpenAI's [Learning Dexterity](https://openai.com/blog/learning-dexterity/)
project and [Robotics Shadow Hand](https://github.com/openai/gym/tree/v0.21.0/gym/envs/robotics)
training environments. The goal of this task is to orient the object in the robot hand to match 
a random target orientation, which is visually displayed by a goal object in the scene.

This example inherets from [InHandManipulationTask](../../omniisaacgymenvs/tasks/shared/in_hand_manipulation.py), 
which is a shared class between this example and the Allegro Hand example. The idea of 
this shared [InHandManipulationTask](../../omniisaacgymenvs/tasks/shared/in_hand_manipulation.py) class
is similar to that of the [LocomotionTask](../../omniisaacgymenvs/tasks/shared/locomotion.py); 
since the Shadow Hand example and the Allegro Hand example only differ by the robot hand used
in the task, using this shared class simplifies implementation across the two.

In this example, motion of the hand is controlled using position targets with `set_joint_position_targets`.
The object and the goal object are reset using `set_world_poses`; their states are retrieved via
`get_world_poses` for computing observations. It is worth noting that the Shadow Hand model in 
this example also demonstrates the use of tendons, which are imported using the `omni.isaac.mjcf` extension. 

Training can be launched with command line argument `task=ShadowHand`.

Training with Domain Randomization can be launched with command line argument `task.domain_randomization.randomize=True`.
For best training results with DR, use `num_envs=16384`.

Running inference with pre-trained model can be launched with command line argument `task=ShadowHand test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/shadow_hand.pth`

Config files used for this task are:

-   **Task config**: [ShadowHand.yaml](../../omniisaacgymenvs/cfg/task/ShadowHand.yaml)
-   **rl_games training config**: [ShadowHandPPO.yaml](../../omniisaacgymenvs/cfg/train/ShadowHandPPO.yaml)
  
#### OpenAI Variant

In addition to the basic version of this task, there is an additional variant matching OpenAI's 
[Learning Dexterity](https://openai.com/blog/learning-dexterity/) project. This variant uses the **openai** 
observations in the policy network, but asymmetric observations of the **full_state** in the value network.
This can be launched with command line argument `task=ShadowHandOpenAI_FF`.

Running inference with pre-trained model can be launched with command line argument `task=ShadowHandOpenAI_FF test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/shadow_hand_openai_ff.pth`

Config files used for this are:

-   **Task config**: [ShadowHandOpenAI_FF.yaml](../../omniisaacgymenvs/cfg/task/ShadowHandOpenAI_FF.yaml)
-   **rl_games training config**: [ShadowHandOpenAI_FFPPO.yaml](../../omniisaacgymenvs/cfg/train/ShadowHandOpenAI_FFPPO.yaml).

#### LSTM Training Variant
This variant uses LSTM policy and value networks instead of feed forward networks, and also asymmetric
LSTM critic designed for the OpenAI variant of the task. This can be launched with command line argument 
`task=ShadowHandOpenAI_LSTM`.

Running inference with pre-trained model can be launched with command line argument `task=ShadowHandOpenAI_LSTM test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/shadow_hand_openai_lstm.pth`

Config files used for this are:

-   **Task config**: [ShadowHandOpenAI_LSTM.yaml](../../omniisaacgymenvs/cfg/task/ShadowHandOpenAI_LSTM.yaml)
-   **rl_games training config**: [ShadowHandOpenAI_LSTMPPO.yaml](../../omniisaacgymenvs/cfg/train/ShadowHandOpenAI_LSTMPPO.yaml).

<img src="https://user-images.githubusercontent.com/34286328/171454160-8cb6739d-162a-4c84-922d-cda04382633f.gif" width="300" height="150"/>


### Allegro Hand Object Manipulation [allegro_hand.py](../../omniisaacgymenvs/tasks/allegro_hand.py)

This example performs the same object orientation task as the Shadow Hand example, 
but using the Allegro hand instead of the Shadow hand.

Training can be launched with command line argument `task=AllegroHand`.

Running inference with pre-trained model can be launched with command line argument `task=AllegroHand test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/allegro_hand.pth`

Config files used for this task are:

-   **Task config**: [AllegroHand.yaml](../../omniisaacgymenvs/cfg/task/Allegro.yaml)
-   **rl_games training config**: [AllegroHandPPO.yaml](../../omniisaacgymenvs/cfg/train/AllegroHandPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454176-ce08f6d0-3087-4ecc-9273-7d30d8f73f6d.gif" width="300" height="150"/>


### ANYmal [anymal.py](../../omniisaacgymenvs/tasks/anymal.py)

This example trains a model of the ANYmal quadruped robot from ANYbotics
to follow randomly chosen x, y, and yaw target velocities.

Training can be launched with command line argument `task=Anymal`.

Running inference with pre-trained model can be launched with command line argument `task=Anymal test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/anymal.pth`

Config files used for this task are:

-   **Task config**: [Anymal.yaml](../../omniisaacgymenvs/cfg/task/Anymal.yaml)
-   **rl_games training config**: [AnymalPPO.yaml](../../omniisaacgymenvs/cfg/train/AnymalPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/184168200-152567a8-3354-4947-9ae0-9443a56fee4c.gif" width="300" height="150"/>


### Anymal Rough Terrain [anymal_terrain.py](../../omniisaacgymenvs/tasks/anymal_terrain.py)

A more complex version of the above Anymal environment that supports
traversing various forms of rough terrain.

Training can be launched with command line argument `task=AnymalTerrain`.

Running inference with pre-trained model can be launched with command line argument `task=AnymalTerrain test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/anymal_terrain.pth`

-   **Task config**: [AnymalTerrain.yaml](../../omniisaacgymenvs/cfg/task/AnymalTerrain.yaml)
-   **rl_games training config**: [AnymalTerrainPPO.yaml](../../omniisaacgymenvs/cfg/train/AnymalTerrainPPO.yaml)

**Note** during test time use the last weights generated, rather than the usual best weights. 
Due to curriculum training, the reward goes down as the task gets more challenging, so the best weights
do not typically correspond to the best outcome.

**Note** if you use the ANYmal rough terrain environment in your work, please ensure you cite the following work:
```
@misc{rudin2021learning,
      title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning}, 
      author={Nikita Rudin and David Hoeller and Philipp Reist and Marco Hutter},
      year={2021},
      journal = {arXiv preprint arXiv:2109.11978}
```
**Note** The OmniIsaacGymEnvs implementation slightly differs from the implementation used in the paper above, which also
uses a different RL library and PPO implementation. The original implementation is made available [here](https://github.com/leggedrobotics/legged_gym). Results reported in the Isaac Gym technical paper are based on that repository, not this one.

<img src="https://user-images.githubusercontent.com/34286328/184170040-3f76f761-e748-452e-b8c8-3cc1c7c8cb98.gif" width="300" height="150"/>


### NASA Ingenuity Helicopter [ingenuity.py](../../omniisaacgymenvs/tasks/ingenuity.py)

This example trains a simplified model of NASA's Ingenuity helicopter to navigate to a moving target.
It showcases the use of velocity tensors and applying force vectors to rigid bodies.
Note that we are applying force directly to the chassis, rather than simulating aerodynamics.
This example also demonstrates using different values for gravitational forces.
Ingenuity Helicopter visual 3D Model courtesy of NASA: https://mars.nasa.gov/resources/25043/mars-ingenuity-helicopter-3d-model/.

Training can be launched with command line argument `task=Ingenuity`.

Running inference with pre-trained model can be launched with command line argument `task=Ingenuity test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ingenuity.pth`

Config files used for this task are:

-   **Task config**: [Ingenuity.yaml](../../omniisaacgymenvs/cfg/task/Ingenuity.yaml)
-   **rl_games training config**: [IngenuityPPO.yaml](../../omniisaacgymenvs/cfg/train/IngenuityPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/184176312-df7d2727-f043-46e3-b537-48a583d321b9.gif" width="300" height="150"/>

### Quadcopter [quadcopter.py](../../omniisaacgymenvs/tasks/quadcopter.py)

This example trains a very simple quadcopter model to reach and hover near a fixed position.  
Lift is achieved by applying thrust forces to the "rotor" bodies, which are modeled as flat cylinders.  
In addition to thrust, the pitch and roll of each rotor is controlled using DOF position targets.

Training can be launched with command line argument `task=Quadcopter`.

Running inference with pre-trained model can be launched with command line argument `task=Quadcopter test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/quadcopter.pth`

Config files used for this task are:

-   **Task config**: [Quadcopter.yaml](../../omniisaacgymenvs/cfg/task/Quadcopter.yaml)
-   **rl_games training config**: [QuadcopterPPO.yaml](../../omniisaacgymenvs/cfg/train/QuadcopterPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/184178817-9c4b6b3c-c8a2-41fb-94be-cfc8ece51d5d.gif" width="300" height="150"/>

### Crazyflie [crazyflie.py](../../omniisaacgymenvs/tasks/crazyflie.py)

This example trains the Crazyflie drone model to hover near a fixed position. It is achieved by applying thrust forces to the four rotors.

Training can be launched with command line argument `task=Crazyflie`.

Running inference with pre-trained model can be launched with command line argument `task=Crazyflie test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/crazyflie.pth`

Config files used for this task are:

-   **Task config**: [Crazyflie.yaml](../../omniisaacgymenvs/cfg/task/Crazyflie.yaml)
-   **rl_games training config**: [CrazyfliePPO.yaml](../../omniisaacgymenvs/cfg/train/CrazyfliePPO.yaml)

<img src="https://user-images.githubusercontent.com/6352136/185715165-b430a0c7-948b-4dce-b3bb-7832be714c37.gif" width="300" height="150"/>

### Ball Balance [ball_balance.py](../../omniisaacgymenvs/tasks/ball_balance.py)

This example trains balancing tables to balance a ball on the table top.
This is a great example to showcase the use of force and torque sensors, as well as DOF states for the table and root states for the ball. 
In this example, the three-legged table has a force sensor attached to each leg. 
We use the force sensor APIs to collect force and torque data on the legs, which guide position target outputs produced by the policy.

Training can be launched with command line argument `task=BallBalance`.

Running inference with pre-trained model can be launched with command line argument `task=BallBalance test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ball_balance.pth`

Config files used for this task are:

-   **Task config**: [BallBalance.yaml](../../omniisaacgymenvs/cfg/task/BallBalance.yaml)
-   **rl_games training config**: [BallBalancePPO.yaml](../../omniisaacgymenvs/cfg/train/BallBalancePPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/184172037-cdad9ee8-f705-466f-bbde-3caa6c7dea37.gif" width="300" height="150"/>

### Franka Cabinet [franka_cabinet.py](../../omniisaacgymenvs/tasks/franka_cabinet.py)

This Franka example demonstrates interaction between Franka arm and cabinet, as well as setting states of objects inside the drawer.
It also showcases control of the Franka arm using position targets.
In this example, we use DOF state tensors to retrieve the state of the Franka arm, as well as the state of the drawer on the cabinet.
Actions are applied as position targets to the Franka arm DOFs.

Training can be launched with command line argument `task=FrankaCabinet`.

Running inference with pre-trained model can be launched with command line argument `task=FrankaCabinet test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/franka_cabinet.pth`

Config files used for this task are:

-   **Task config**: [FrankaCabinet.yaml](../../omniisaacgymenvs/cfg/task/FrankaCabinet.yaml)
-   **rl_games training config**: [FrankaCabinetPPO.yaml](../../omniisaacgymenvs/cfg/train/FrankaCabinetPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/184174894-03767aa0-936c-4bfe-bbe9-a6865f539bb4.gif" width="300" height="150"/>

### Franka Deformable [franka_deformable.py](../../omniisaacgymenvs/tasks/franka_deformable.py)

This Franka example demonstrates interaction between Franka arm and a deformable tube. It demonstrates the manipulation of deformable objects, using nodal positions and velocities of the simulation mesh as observations.

Training can be launched with command line argument `task=FrankaDeformable`.

Running inference with pre-trained model can be launched with command line argument `task=FrankaDeformable test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/franka_deformable.pth`

Config files used for this task are:

-   **Task config**: [FrankaDeformable.yaml](../../omniisaacgymenvs/cfg/task/FrankaDeformable.yaml)
-   **rl_games training config**: [FrankaCabinetFrankaDeformable.yaml](../../omniisaacgymenvs/cfg/train/FrankaDeformablePPO.yaml)


### Factory: Fast Contact for Robotic Assembly

We provide a set of Factory example tasks, [**FactoryTaskNutBoltPick**](../../omniisaacgymenvs/tasks/factory/factory_task_nut_bolt_pick.py), [**FactoryTaskNutBoltPlace**](../../omniisaacgymenvs/tasks/factory/factory_task_nut_bolt_place.py), and [**FactoryTaskNutBoltScrew**](../../omniisaacgymenvs/tasks/factory/factory_task_nut_bolt_screw.py), 

`FactoryTaskNutBoltPick` can be executed with `python train.py task=FactoryTaskNutBoltPick`. This task trains policy for the Pick task, a simplified version of the corresponding task in the Factory paper. The policy may take ~1 hour to achieve high success rates on a modern GPU.

- The general configuration file for the above task is [FactoryTaskNutBoltPick.yaml](../../omniisaacgymenvs/cfg/task/FactoryTaskNutBoltPick.yaml).
- The training configuration file for the above task is [FactoryTaskNutBoltPickPPO.yaml](../../omniisaacgymenvs/cfg/train/FactoryTaskNutBoltPickPPO.yaml).

Running inference with pre-trained model can be launched with command line argument `task=FactoryTaskNutBoltPick test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/factory_task_nut_bolt_pick.pth`

`FactoryTaskNutBoltPlace` can be executed with `python train.py task=FactoryTaskNutBoltPlace`. This task trains policy for the Place task.

- The general configuration file for the above task is [FactoryTaskNutBoltPlace.yaml](../../omniisaacgymenvs/cfg/task/FactoryTaskNutBoltPlace.yaml).
- The training configuration file for the above task is [FactoryTaskNutBoltPlacePPO.yaml](../../omniisaacgymenvs/cfg/train/FactoryTaskNutBoltPlacePPO.yaml).

Running inference with pre-trained model can be launched with command line argument `task=FactoryTaskNutBoltPlace test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/factory_task_nut_bolt_place.pth`

`FactoryTaskNutBoltScrew` can be executed with `python train.py task=FactoryTaskNutBoltScrew`. This task trains policy for the Screw task.

- The general configuration file for the above task is [FactoryTaskNutBoltScrew.yaml](../../omniisaacgymenvs/cfg/task/FactoryTaskNutBoltScrew.yaml).
- The training configuration file for the above task is [FactoryTaskNutBoltScrewPPO.yaml](../../omniisaacgymenvs/cfg/train/FactoryTaskNutBoltScrewPPO.yaml).

Running inference with pre-trained model can be launched with command line argument `task=FactoryTaskNutBoltScrew test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/factory_task_nut_bolt_screw.pth`

If you use the Factory simulation methods (e.g., SDF collisions, contact reduction) or Factory learning tools (e.g., assets, environments, or controllers) in your work, please cite the following paper:
```
@inproceedings{
  narang2022factory,
  author = {Yashraj Narang and Kier Storey and Iretiayo Akinola and Miles Macklin and Philipp Reist and Lukasz Wawrzyniak and Yunrong Guo and Adam Moravanszky and Gavriel State and Michelle Lu and Ankur Handa and Dieter Fox},
  title = {Factory: Fast contact for robotic assembly},
  booktitle = {Robotics: Science and Systems},
  year = {2022}
} 
```

Also note that our original formulations of SDF collisions and contact reduction were developed by [Macklin, et al.](https://dl.acm.org/doi/abs/10.1145/3384538) and [Moravanszky and Terdiman](https://scholar.google.com/scholar?q=Game+Programming+Gems+4%2C+chapter+Fast+Contact+Reduction+for+Dynamics+Simulation), respectively.

<img src="https://user-images.githubusercontent.com/6352136/205978286-fa2ae714-a3cb-4acd-9f5f-a467338a8bb3.gif"/>
