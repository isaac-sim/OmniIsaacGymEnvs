## Transfering Policies from Isaac Gym

This section delineates some of the differences between the standalone 
[Isaac Gym](https://developer.nvidia.com/isaac-gym) and the Isaac Sim RL 
framework in hopes of facilitating the process of transferring policies 
trained in the standalone Isaac Gym to Isaac Sim.


### Quaternion Convention

The Isaac Sim RL framework uses various classes and methods in `omni.isaac.core`,
which adopts `wxyz` as the quaternion convention. However, the quaternion 
convention used in Isaac Gym is `xyzw`. Therefore, if a policy trained
in Isaac Gym takes in quaternions as part of its observations, remember to
switch all quaternions to use the `xyzw` convention in the observation buffer 
`self.obs_buf`. Similarly, please ensure all quaternions are in `wxyz` before 
passing them in any of the utility functions in `omni.isaac.core`. 


### Joint Order

Isaac Sim's `ArticulationView` in `omni.isaac.core` assumes a breadth-first 
ordering for the joints in a given kinematic tree. Specifically, for the following
kinematic tree, the method `ArticulationView.get_joint_positions` returns a 
tensor of shape `(number of articulations in the view, number of joints in the articulation)`.
Along the second dimension of this tensor, the values represent the articulation's joint positions 
in the following order: `[Joint 1, Joint 2, Joint 4, Joint 3, Joint 5]`. On the other hand, 
Isaac Gym assumes a depth-first ordering for the joints in the kinematic tree; In the example 
below, the joint orders would be the following: `[Joint 1, Joint 2, Joint 3, Joint 4, Joint 5]`.

<img src="./media/KinematicTree.png" height="300"/>

With this in mind, it is thus important to change the joint order to depth-first in 
the observation buffer before feeding it into an Isaac Gym trained policy. Similarly,
the user would also need to change the joint order in the output (the action buffer) 
of the Isaac Gym trained policy to breadth-first before applying joint actions to 
articulations via methods in `ArticulationView`.


### Physics Parameters

One factor that could dictate the success of policy transfer from Isaac Gym to
Isaac Sim is to ensure the physics parameters used in both simulations are
identical or very similar. In general, the `sim` parameters specified in the 
task configuration `yaml` file overwrite the corresponding parameters in the USD asset.
However, there are additional parameters in the USD asset that are not included
in the task configuration `yaml` file. These additional parameters could sometimes
impact the performance of the Isaac Gym trained policy and hence need to be modified
in the USD asset itself to match the values set in Isaac Gym.

For instance, the following parameters in the `RigidBodyAPI` could be modified in the 
USD asset to yield better policy transfer performance: 

| RigidBodyAPI Parameter | Default Value in Isaac Sim | Default Value in Isaac Gym |
|:----------------------:|:--------------------------:|:--------------------------:|
|     Linear Damping     |            0.00            |            0.00            |
|     Angular Damping    |            0.05            |            0.00            |
|   Max Linear Velocity  |             inf            |            1000            |
|  Max Angular Velocity  |     5729.58008 (deg/s)     |         64 (rad/s)         |
|   Max Contact Impulse  |             inf            |            1e32            |

<img src="./media/RigidBodyAPI.png" width="500"/>

Parameters in the `JointAPI` as well as the `DriveAPI` could be altered as well. Note 
that the Isaac Sim UI assumes the unit of angle to be degress. It is particularly
worth noting that the `Damping` and `Stiffness` paramters in the `DriveAPI` have the unit
of `1/deg` in the Isaac Sim UI but `1/rad` in Isaac Gym.

|     Joint Parameter    | Default Value in Isaac Sim | Default Value in Isaac Gym |
|:----------------------:|:--------------------------:|:--------------------------:|
| Maximum Joint Velocity |       1000000.0 (deg)      |         100.0 (rad)        |

<img src="./media/JointAPI.png" width="500"/>

Another difference in physics parameters is the definition of substeps for simulation. In Isaac Sim, we keep substeps fixed at `1` and use the `controlFrequencyInv` and `dt` parameters instead to achieve the same substeps definition as in standalone Isaac Gym. For example, if in standalone Isaac Gym we set `substeps: 2`, `dt: 1/60` and `controlFrequencyInv: 1`, we can achieve the equivalent in Isaac Sim by setting `controlFrequencyInv: 2` and `dt: 1/120`.