## Reinforcement Learning Examples

We introduce the following reinforcement learning examples that are implemented using 
Isaac Sim's RL framework. 

Pre-trained checkpoints can be found on the Nucleus server. To set up localhost, please refer to the [Isaac Sim installation guide](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html).

*Note: All commands should be executed from `omniisaacgymenvs/omniisaacgymenvs`.*

### Cartpole [cartpole.py](../omniisaacgymenvs/tasks/cartpole.py)

Cartpole is a simple example that demonstrates getting and setting usage of DOF states using 
`ArticulationView` from `omni.isaac.core`. The goal of this task is to move a cart horizontally
such that the pole, which is connected to the cart via a revolute joint, stays upright.

Joint positions and joint velocities are retrieved using `get_joint_positions` and 
`get_joint_velocities` respectively, which are required in computing observations. Actions are 
applied onto the cartpoles via `set_joint_efforts`. Cartpoles are reset by using `set_joint_positions` 
and `set_joint_velocities`. 

Training can be launched with command line argument `task=Cartpole`.

Running inference with pre-trained model can be launched with command line argument `task=Cartpole test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/cartpole.pth`

Config files used for this task are:

-   **Task config**: [Cartpole.yaml](../omniisaacgymenvs/cfg/task/Cartpole.yaml)
-   **rl_games training config**: [CartpolePPO.yaml](../omniisaacgymenvs/cfg/train/CartpolePPO.yaml)
  
<img src="https://user-images.githubusercontent.com/34286328/171454189-6afafbff-bb61-4aac-b518-24646007cb9f.gif" width="300" height="150"/>

### Ant [ant.py](../omniisaacgymenvs/tasks/ant.py)

Ant is an example of a simple locomotion task. The goal of this task is to train 
quadruped robots (ants) to run forward as fast as possible. This example inherets 
from [LocomotionTask](../omniisaacgymenvs/tasks/shared/locomotion.py), 
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
torques to the DOFs. Force sensors are also placed on each of the legs to observe contacts
with the ground plane; the sensor values can be retrieved using `get_force_sensor_forces`. 

Training can be launched with command line argument `task=Ant`.

Running inference with pre-trained model can be launched with command line argument `task=Ant test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/ant.pth`

Config files used for this task are:

-   **Task config**: [Ant.yaml](../omniisaacgymenvs/cfg/task/Ant.yaml)
-   **rl_games training config**: [AntPPO.yaml](../omniisaacgymenvs/cfg/train/AntPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454182-0be1b830-bceb-4cfd-93fb-e1eb8871ec68.gif" width="300" height="150"/>



### Humanoid [humanoid.py](../omniisaacgymenvs/tasks/humanoid.py)

Humanoid is another environment that uses 
[LocomotionTask](../omniisaacgymenvs/tasks/shared/locomotion.py). It is conceptually
very similar to the Ant example, where the goal for the humanoid is to run forward
as fast as possible.

Training can be launched with command line argument `task=Humanoid`.

Running inference with pre-trained model can be launched with command line argument `task=Humanoid test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/humanoid.pth`

Config files used for this task are:

-   **Task config**: [Humanoid.yaml](../omniisaacgymenvs/cfg/task/Humanoid.yaml)
-   **rl_games training config**: [HumanoidPPO.yaml](../omniisaacgymenvs/cfg/train/HumanoidPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454193-e027885d-1510-4ef4-b838-06b37f70c1c7.gif" width="300" height="150"/>


### Shadow Hand Object Manipulation [shadow_hand.py](../omniisaacgymenvs/tasks/shadow_hand.py)

The Shadow Hand task is an example of a challenging dexterity manipulation task with complex contact 
dynamics. It resembles OpenAI's [Learning Dexterity](https://openai.com/blog/learning-dexterity/)
project and [Robotics Shadow Hand](https://github.com/openai/gym/tree/master/gym/envs/robotics)
training environments. The goal of this task is to orient the object in the robot hand to match 
a random target orientation, which is visually displayed by a goal object in the scene.

This example inherets from [InHandManipulationTask](../omniisaacgymenvs/tasks/shared/in_hand_manipulation.py), 
which is a shared class between this example and the Allegro Hand example. The idea of 
this shared [InHandManipulationTask](../omniisaacgymenvs/tasks/shared/in_hand_manipulation.py) class
is similar to that of the [LocomotionTask](../omniisaacgymenvs/tasks/shared/locomotion.py); 
since the Shadow Hand example and the Allegro Hand example only differ by the robot hand used
in the task, using this shared class simplifies implementation across the two.

In this example, motion of the hand is controlled using position targets with `set_joint_position_targets`.
The object and the goal object are reset using `set_world_poses`; their states are retrieved via
`get_world_poses` for computing observations. It is worth noting that the Shadow Hand model in 
this example also demonstrates the use of tendons, which are imported using the `omni.isaac.mjcf` extension. 

Training can be launched with command line argument `task=ShadowHand`.

Running inference with pre-trained model can be launched with command line argument `task=ShadowHand test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/shadow_hand.pth`

Config files used for this task are:

-   **Task config**: [ShadowHand.yaml](../omniisaacgymenvs/cfg/task/ShadowHand.yaml)
-   **rl_games training config**: [ShadowHandPPO.yaml](../omniisaacgymenvs/cfg/train/ShadowHandPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454160-8cb6739d-162a-4c84-922d-cda04382633f.gif" width="300" height="150"/>


### Allegro Hand Object Manipulation [allegro_hand.py](../omniisaacgymenvs/tasks/allegro_hand.py)

This example performs the same object orientation task as the Shadow Hand example, 
but using the Allegro hand instead of the Shadow hand.

Training can be launched with command line argument `task=AllegroHand`.

Running inference with pre-trained model can be launched with command line argument `task=AllegroHand test=True checkpoint=omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/allegro_hand.pth`

Config files used for this task are:

-   **Task config**: [AllegroHand.yaml](../omniisaacgymenvs/cfg/task/Allegro.yaml)
-   **rl_games training config**: [AllegroHandPPO.yaml](../omniisaacgymenvs/cfg/train/AllegroHandPPO.yaml)

<img src="https://user-images.githubusercontent.com/34286328/171454176-ce08f6d0-3087-4ecc-9273-7d30d8f73f6d.gif" width="300" height="150"/>