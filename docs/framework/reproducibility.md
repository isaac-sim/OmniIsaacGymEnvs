Reproducibility and Determinism
===============================

Seeds
-----

To achieve deterministic behavior on multiple training runs, a seed
value can be set in the training config file for each task. This will potentially
allow for individual runs of the same task to be deterministic when
executed on the same machine and system setup. Alternatively, a seed can
also be set via command line argument `seed=<seed>` to override any
settings in config files. If no seed is specified in either config files
or command line arguments, we default to generating a random seed. In
this case, individual runs of the same task should not be expected to be
deterministic. For convenience, we also support setting `seed=-1` to
generate a random seed, which will override any seed values set in
config files. By default, we have explicitly set all seed values in
config files to be 42.


PyTorch Deterministic Training
------------------------------

We also include a `torch_deterministic` argument for use when running RL
training. Enabling this flag (by passing `torch_deterministic=True`) will
apply additional settings to PyTorch that can force the usage of deterministic 
algorithms in PyTorch, but may also negatively impact runtime performance. 
For more details regarding PyTorch reproducibility, refer to
<https://pytorch.org/docs/stable/notes/randomness.html>. If both
`torch_deterministic=True` and `seed=-1` are set, the seed value will be
fixed to 42.


Runtime Simulation Changes / Domain Randomization
-------------------------------------------------

Note that using a fixed seed value will only **potentially** allow for deterministic 
behavior. Due to GPU work scheduling, it is possible that runtime changes to 
simulation parameters can alter the order in which operations take place, as 
environment updates can happen while the GPU is doing other work. Because of the nature 
of floating point numeric storage, any alteration of execution ordering can 
cause small changes in the least significant bits of output data, leading
to divergent execution over the simulation of thousands of environments and
simulation frames.

As an example of this, runtime domain randomization of object scales 
is known to cause both determinancy and simulation issues when running on the GPU 
due to the way those parameters are passed from CPU to GPU in lower level APIs. Therefore,
this is only supported at setup time before starting simulation, which is specified by
the `on_startup` condition for Domain Randomization. 

At this time, we do not believe that other domain randomizations offered by this
framework cause issues with deterministic execution when running GPU simulation, 
but directly manipulating other simulation parameters outside of the omni.isaac.core View 
APIs may induce similar issues.

Also due to floating point precision, states across different environments in the simulation
may be non-deterministic when the same set of actions are applied to the same initial
states. This occurs as environments are placed further apart from the world origin at (0, 0, 0).
As actors get placed at different origins in the world, floating point errors may build up
and result in slight variance in results even when starting from the same initial states. One 
possible workaround for this issue is to place all actors/environments at the world origin
at (0, 0, 0) and filter out collisions between the environments. Note that this may induce
a performance degradation of around 15-50%, depending on the complexity of actors and 
environment.

Another known cause of non-determinism is from resetting actors into contact states. 
If actors within a scene is reset to a state where contacts are registered 
between actors, the simulation may not be able to produce deterministic results.
This is because contacts are not recorded and will be re-computed from scratch for 
each reset scenario where actors come into contact, which cannot guarantee 
deterministic behavior across different computations.

