Domain Randomization
====================

Overview
--------

We sometimes need our reinforcement learning agents to be robust to
different physics than they are trained with, such as when attempting a
sim2real policy transfer. Using domain randomization (DR), we repeatedly
randomize the simulation dynamics during training in order to learn a
good policy under a wide range of physical parameters.

OmniverseIsaacGymEnvs supports "on the fly" domain randomization, allowing 
dynamics to be changed without requiring reloading of assets. This allows 
us to efficiently apply domain randomizations without common overheads like
re-parsing asset files. 

The OmniverseIsaacGymEnvs DR framework utilizes the `omni.replicator.isaac`
extension in its backend to perform "on the fly" randomization. Users can
add domain randomization by either directly using methods provided in 
`omni.replicator.isaac` in python, or specifying DR settings in the 
task configuration `yaml` file. The following sections will focus on setting
up DR using the `yaml` file interface. For more detailed documentations
regarding methods provided in the `omni.replicator.isaac` extension, please
visit [here](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.replicator.isaac/docs/index.html).


Domain Randomization Options
-------------------------------

We will first explain what can be randomized in the scene and the sampling 
distributions. There are five main parameter groups that support randomization. 
They are:

-   `observations`: Add noise directly to the agent observations

-   `actions`: Add noise directly to the agent actions

-   `simulation`: Add noise to physical parameters defined for the entire
                  scene, such as `gravity`

-   `rigid_prim_views`: Add noise to properties belonging to rigid prims, 
                        such as `material_properties`.
-   `articulation_views`: Add noise to properties belonging to articulations, 
                        such as `stiffness` of joints.

For each parameter you wish to randomize, you can specify two ways that
determine when the randomization is applied:

-   `on_reset`: Adds correlated noise to a parameter of an environment when
                that environment gets reset. This correlated noise will remain 
                with an environment until that environemnt gets reset again, which 
                will then set a new correlated noise. To trigger `on_reset`,
                the indices for the environemnts that need to be reset must be passed in
                to `omni.replicator.isaac.physics_view.step_randomization(reset_inds)`.

-   `on_interval`: Adds uncorrelated noise to a parameter at a frequency specified
                   by `frequency_interval`. If a parameter also has `on_reset` randomization,
                   the `on_interval` noise is combined with the noise applied at `on_reset`.

-   `on_startup`: Applies randomization once prior to the start of the simulation. Only available
                  to rigid prim scale, mass, density and articulation scale parameters.
            
For `on_reset`, `on_interval`, and `on_startup`, you can specify the following settings:

-   `distribution`: The distribution to generate a sample `x` from. The available distributions 
                    are listed below. Note that parameters `a` and `b` are defined by the 
                    `distribution_parameters` setting.
    -   `uniform`: `x ~ unif(a, b)`
    -   `loguniform`: `x ~ exp(unif(log(a), log(b)))`
    -   `gaussian`: `x ~ normal(a, b)`

-   `distribution_parameters`: The parameters to the distribution.
    -   For observations and actions, this setting is specified as a tuple `[a, b]` of
        real values.
    -   For simulation and view parameters, this setting is specified as a nested tuple 
        in the form of `[[a_1, a_2, ..., a_n], [[b_1, b_2, ..., b_n]]`, where the `n` is
        the dimension of the parameter (*i.e.* `n` is 3 for position). It can also be
        specified as a tuple in the form of `[a, b]`, which will be broadcasted to the 
        correct dimensions.
    -   For `uniform` and `loguniform` distributions, `a` and `b` are the lower and 
        upper bounds.
    -   For `gaussian`, `a` is the distribution mean and `b` is the variance.

-   `operation`: Defines how the generated sample `x` will be applied to the original 
                 simulation parameter. The options are `additive`, `scaling`, `direct`.
    -   `additive`:, add the sample to the original value.
    -   `scaling`: multiply the original value by the sample.
    -   `direct`: directly sets the sample as the parameter value. 

-   `frequency_interval`: Specifies the number of steps to apply randomization. 
    -   Only used with `on_interval`. 
    -   Steps of each environemnt are incremented with each 
        `omni.replicator.isaac.physics_view.step_randomization(reset_inds)` call and 
        reset if the environment index is in `reset_inds`.

-   `num_buckets`: Only used for `material_properties` randomization
    -   Physx only allows 64000 unique physics materials in the scene at once. If more than
        64000 materials are needed, increase `num_buckets` to allow materials to be shared
        between prims.


YAML Interface
--------------

Now that we know what options are available for domain randomization,
let's put it all together in the YAML config. In your `omniverseisaacgymenvs/cfg/task` 
yaml file, you can specify your domain randomization parameters under the
`domain_randomization` key. First, we turn on domain randomization by setting
`randomize` to `True`:
```yaml
    domain_randomization:
        randomize: True
        randomization_params:
            ...
```

This can also be set as a command line argument at launch time with `task.domain_randomization.randomize=True`. 

Next, we will define our parameters under the `randomization_params`
keys. Here you can see how we used the previous settings to define some
randomization parameters for a ShadowHand cube manipulation task:

```yaml
    randomization_params:
        randomization_params:
            observations:
              on_reset:
                operation: "additive"
                distribution: "gaussian"
                distribution_parameters: [0, .0001]
              on_interval:
                frequency_interval: 1
                operation: "additive"
                distribution: "gaussian"
                distribution_parameters: [0, .002]
            actions:
              on_reset:
                operation: "additive"
                distribution: "gaussian"
                distribution_parameters: [0, 0.015]
              on_interval:
                frequency_interval: 1
                operation: "additive"
                distribution: "gaussian"
                distribution_parameters: [0., 0.05]
        simulation:
          gravity:
            on_reset:
              operation: "additive"
              distribution: "gaussian"
              distribution_parameters: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]]
        rigid_prim_views:
          object_view:
            material_properties:
              on_reset:
                num_buckets: 250
                operation: "scaling"
                distribution: "uniform"
                distribution_parameters: [[0.7, 1, 1], [1.3, 1, 1]]
        articulation_views:
          shadow_hand_view:
            stiffness:
              on_reset:
                operation: "scaling"
                distribution: "uniform"
                distribution_parameters: [0.75, 1.5]
```    

Note how we structured `rigid_prim_views` and `articulation_views`. When creating
a `RigidPrimView` or `ArticulationView` in the task python file, you have the option to
pass in `name` as an argument. **To use domain randomization, the name of the `RigidPrimView` or 
`ArticulationView` must match the name provided in the randomization `yaml` file.** In the 
example above, `object_view` is the name of a `RigidPrimView` and `shadow_hand_view` is the name
of the `ArticulationView`. 

The exact parameters that can be randomized are listed below:

**simulation**:
- gravity (dim=3): The gravity vector of the entire scene.

**rigid\_prim\_views**:
- position (dim=3): The position of the rigid prim. In meters.
- orientation (dim=3): The orientation of the rigid prim, specified with euler angles. In radians.
- linear_velocity (dim=3): The linear velocity of the rigid prim. In m/s. **CPU pipeline only**
- angular_velocity (dim=3): The angular velocity of the rigid prim. In rad/s. **CPU pipeline only**
- velocity (dim=6): The linear + angular velocity of the rigid prim.
- force (dim=3): Apply a force to the rigid prim. In N.
- mass (dim=1): Mass of the rigid prim. In kg. **CPU pipeline only during runtime**. 
- inertia (dim=3): The diagonal values of the inertia matrix. **CPU pipeline only**
- material_properties (dim=3): Static friction, Dynamic friction, and Restitution.
- contact_offset (dim=1): A small distance from the surface of the collision geometry at 
                          which contacts start being generated.
- rest_offset (dim=1): A small distance from the surface of the collision geometry at 
                       which the effective contact with the shape takes place.
- scale (dim=1): The scale of the rigid prim. `on_startup` only. 
- density (dim=1): Density of the rigid prim. `on_startup` only. 

**articulation\_views**:
- position (dim=3): The position of the articulation root. In meters.
- orientation (dim=3): The orientation of the articulation root, specified with euler angles. In radians.
- linear_velocity (dim=3): The linear velocity of the articulation root. In m/s. **CPU pipeline only**
- angular_velocity (dim=3): The angular velocity of the articulation root. In rad/s. **CPU pipeline only**
- velocity (dim=6): The linear + angular velocity of the articulation root.
- stiffness (dim=num_dof): The stiffness of the joints.
- damping (dim=num_dof): The damping of the joints
- joint_friction (dim=num_dof): The friction coefficient of the joints.
- joint_positions (dim=num_dof): The joint positions. In radians or meters.
- joint_velocities (dim=num_dof): The joint velocities. In rad/s or m/s.
- lower_dof_limits (dim=num_dof): The lower limit of the joints. In radians or meters.
- upper_dof_limits (dim=num_dof): The upper limit of the joints. In radians or meters.
- max_efforts (dim=num_dof): The maximum force or torque that the joints can exert. In N or Nm.
- joint_armatures (dim=num_dof): A value added to the diagonal of the joint-space inertia matrix. 
                                 Physically, it corresponds to the rotating part of a motor 
- joint_max_velocities (dim=num_dof): The maximum velocity allowed on the joints. In rad/s or m/s.
- joint_efforts (dim=num_dof): Applies a force or a torque on the joints. In N or Nm.
- body_masses (dim=num_bodies): The mass of each body in the articulation. In kg. **CPU pipeline only**
- body_inertias (dim=num_bodies×3): The diagonal values of the inertia matrix of each body. **CPU pipeline only**
- material_properties (dim=num_bodies×3): The static friction, dynamic friction, and restitution of each body
                                          in the articulation, specified in the following order:
                                          [body_1_static_friciton, body_1_dynamic_friciton, body_1_restitution,
                                           body_1_static_friciton, body_2_dynamic_friciton, body_2_restitution,
                                           ... ]
- tendon_stiffnesses (dim=num_tendons): The stiffness of the fixed tendons in the articulation.
- tendon_dampings (dim=num_tendons): The damping of the fixed tendons in the articulation.
- tendon_limit_stiffnesses (dim=num_tendons): The limit stiffness of the fixed tendons in the articulation.
- tendon_lower_limits (dim=num_tendons): The lower limits of the fixed tendons in the articulation.
- tendon_upper_limits (dim=num_tendons): The upper limits of the fixed tendons in the articulation.
- tendon_rest_lengths (dim=num_tendons): The rest lengths of the fixed tendons in the articulation.
- tendon_offsets (dim=num_tendons): The offsets of the fixed tendons in the articulation.
- scale (dim=1): The scale of the articulation. `on_startup` only. 

Applying Domain Randomization
------------------------------

To parse the domain randomization configurations in the task `yaml` file and set up the DR pipeline, 
it is necessary to call `self._randomizer.set_up_domain_randomization(self)`, where `self._randomizer`
is the `Randomizer` object created in RLTask's `__init__`.

It is worth noting that the names of the views provided under `rigid_prim_views` or `articulation_views` 
in the task `yaml` file must match the names passed into `RigidPrimView` or `ArticulationView` objects
in the python task file. In addition, all `RigidPrimView` and `ArticulationView` that would have domain
randomizaiton applied must be added to the scene in the task's `set_up_scene()` via `scene.add()`.

To trigger `on_startup` randomizations, call `self._randomizer.apply_on_startup_domain_randomization(self)`
in `set_up_scene()` after all views are added to the scene. Note that `on_startup` randomizations
are only availble to rigid prim scale, mass, density and articulation scale parameters since these parameters
cannot be randomized after the simulation begins on GPU pipeline. Therefore, randomizations must be applied
to these parameters in `set_up_scene()` prior to the start of the simulation.  

To trigger `on_reset` and `on_interval` randomizations, it is required to step the interal
counter of the DR pipeline in `pre_physics_step()`:

```python
if self._randomizer.randomize:
    omni.replicator.isaac.physics_view.step_randomization(reset_inds)
```
`reset_inds` is a list of indices of the environments that need to be reset. For those environments, it will
trigger the randomizations defined with `on_reset`. All other environments will follow randomizations
defined with `on_interval`. 


Randomization Scheduling
----------------------------

We provide methods to modify distribution parameters defined in the `yaml` file during training, which
allows custom DR scheduling. There are three methods from the `Randomizer` class
that are relevant to DR scheduling:

- `get_initial_dr_distribution_parameters`: returns a numpy array of the initial parameters (as defined in 
                                          the `yaml` file) of a specified distribution
- `get_dr_distribution_parameters`: returns a numpy array of the current parameters of a specified distribution
- `set_dr_distribution_parameters`: sets new parameters to a specified distribution

Using the DR configuration example defined above, we can get the current parameters and set new parameters
to gravity randomization and shadow hand joint stiffness randomization as follows:

```python
current_gravity_dr_params = self._randomizer.get_dr_distribution_parameters(
    "simulation", 
    "gravity", 
    "on_reset",
)
self._randomizer.set_dr_distribution_parameters(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], 
    "simulation", 
    "gravity", 
    "on_reset",
)

current_joint_stiffness_dr_params = self._randomizer.get_dr_distribution_parameters(
    "articulation_views", 
    "shadow_hand_view", 
    "stiffness", 
    "on_reset",
)

self._randomizer.set_dr_distribution_parameters(
    [0.7, 1.55],
    "articulation_views", 
    "shadow_hand_view", 
    "stiffness", 
    "on_reset",
)
```

The following is an example of using these methods to perform linear scheduling of gaussian noise 
that is added to observations and actions in the above shadow hand example. The following method
linearly adds more noise to observations and actions every epoch up until the `schedule_epoch`.
This method can be added to the Task python class and be called in `pre_physics_step()`.

```python
def apply_observations_actions_noise_linear_scheduling(self, schedule_epoch=100):
    current_epoch = self._env.sim_frame_count // self._cfg["task"]["env"]["controlFrequencyInv"] // self._cfg["train"]["params"]["config"]["horizon_length"]
    if current_epoch <= schedule_epoch:
        if (self._env.sim_frame_count // self._cfg["task"]["env"]["controlFrequencyInv"]) % self._cfg["train"]["params"]["config"]["horizon_length"] == 0:
            for distribution_path in [("observations", "on_reset"), ("observations", "on_interval"), ("actions", "on_reset"), ("actions", "on_interval")]:
                scheduled_params = self._randomizer.get_initial_dr_distribution_parameters(*distribution_path)
                scheduled_params[1] = (1/schedule_epoch) * current_epoch * scheduled_params[1]
                self._randomizer.set_dr_distribution_parameters(scheduled_params, *distribution_path)
```
