Release Notes
=============

2023.1.1 - December 12, 2023
----------------------------

Additions
---------
- Add support for viewport recording during training/inferencing using gym wrapper class `RecordVideo`
- Add `enable_recording`, `recording_interval`, `recording_length`, and `recording_fps`, `recording_dir` arguments to config/command-line for video recording
- Add `moviepy` as dependency for video recording
- Add video tutorial for extension workflow, available at [docs/framework/extension_workflow.md](docs/framework/extension_workflow.md)
- Add camera clipping for CartpoleCamera to avoid seeing other environments in the background

Changes
-------
- Use rl_device for sampling random policy (https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/pull/51)
- Add FPS printouts for random policy
- Use absolute path for default checkpoint folder for consistency between Python and extension workflows
- Change camera creation API in CartpoleCamera to use USD APIs instead of `rep.create`

Fixes
-----
- Fix missing device in warp kernel launch for Ant and Humanoid
- Fix typo for velocity iteration (https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/pull/111)
- Clean up private variable access in task classes in favour of property getters
- Clean up private variable access in extension.py in favour of setter methods
- Unregister replicator in extension workflow on training completion to allow for restart


2023.1.0b - November 02, 2023
-----------------------------

Changes
-------
- Update docker scripts to Isaac Sim docker image 2023.1.0-hotfix.1
- Use omniisaacgymenvs module root for app file parsing
- Update FrankaDeformable physics dt for better training stability

Fixes
-----
- Fix CartpoleCamera num_observations value
- Fix missing import in startup randomization for mass and density


2023.1.0a - October 20, 2023
----------------------------

Fixes
-----
- Fix extension loading error in camera app file


2023.1.0 - October 18, 2023
---------------------------

Additions
---------
- Add support for Warp backend task implementation
- Add Warp-based RL examples: Cartpole, Ant, Humanoid
- Add new Factory environments for place and screw: FactoryTaskNutBoltPlace and FactoryTaskNutBoltScrew
- Add new camera-based Cartpole example: CartpoleCamera
- Add new deformable environment showing Franka picking up a deformable tube: FrankaDeformable
- Add support for running OIGE as an extension in Isaac Sim
- Add options to filter collisions between environments and specify global collision filter paths to `RLTask.set_to_scene()`
- Add multinode training support
- Add dockerfile with OIGE
- Add option to select kit app file from command line argument `kit_app`
- Add `rendering_dt` parameter to the task config file for setting rendering dt. Defaults to the same value as the physics dt.

Changes
-------
- `use_flatcache` flag has been renamed to `use_fabric`
- Update hydra-core version to 1.3.2, omegaconf version to 2.3.0
- Update rlgames to version 1.6.1.
- The `get_force_sensor_forces` API for articulations is now deprecated and replaced with `get_measured_joint_forces`
- Remove unnecessary cloning of buffers in VecEnv classes
- Only enable omni.replicator.isaac when domain randomization or cameras are enabled
- The multi-threaded launch script `rlgames_train_mt.py` has been re-designed to support the extension workflow. This script can no longer be used to launch a training run from python. Please use `rlgames_train.py` instead.
- Restructures for environments to support the new extension-based workflow
- Add async workflow to factory pick environment to support extension-based workflow
- Update docker scripts with cache directories

Fixes
-----
- Fix errors related to setting velocities to kinematic markers in Ingenuity and Quadcopter environments
- Fix contact-related issues with quadruped assets
- Fix errors in physics APIs when returning empty tensors
- Fix orientation correctness issues when using some assets with omni.isaac.core. Additional orientations applied to accommodate for the error are no longer required (i.e. ShadowHand)
- Updated the deprecated config name `seq_len` used with RNN networks to `seq_length`


2022.2.1 - March 16, 2023
-------------------------

Additions
---------
- Add FactoryTaskNutBoltPick example
- Add Ant and Humanoid SAC training examples
- Add multi-GPU support for training
- Add utility scripts for launching Isaac Sim docker with OIGE
- Add support for livestream through the Omniverse Streaming Client

Changes
-------
- Change rigid body fixed_base option to make_kinematic, avoiding creation of unnecessary articulations
- Update ShadowHand, Ingenuity, Quadcopter and Crazyflie marker objects to use kinematics
- Update ShadowHand GPU buffer parameters
- Disable PyTorch nvFuser for better performance
- Enable viewport and replicator extensions dynamically to maintain order of extension startup
- Separate app files for headless environments with rendering (requires Isaac Sim update)
- Update rl-games to v1.6.0

Fixes
-----
- Fix material property randomization at run-time, including friction and restitution (requires Isaac Sim update)
- Fix a bug in contact reporting API where incorrect values were being reported (requires Isaac Sim update)
- Enable render flag in Isaac Sim when enable_cameras is set to True
- Add root pose and velocity reset to BallBalance environment


2.0.0 - December 15, 2022
-------------------------

Additions
---------
- Update to Viewport 2.0
- Allow for runtime mass randomization on GPU pipeline
- Add runtime mass randomization to ShadowHand environments
- Introduce `disable_contact_processing` simulation parameter for faster contact processing
- Use physics replication for cloning by default for faster load time

Changes
-------
- Update AnymalTerrain environment to use contact forces
- Update Quadcopter example to apply local forces
- Update training parameters for ShadowHandOpenAI_FF environment
- Rename rlgames_play.py to rlgames_demo.py

Fixes
-----
- Remove fix_base option from articulation configs
- Fix in_hand_manipulation random joint position sampling on reset
- Fix mass and density randomization in MT training script
- Fix actions/observations noise randomization in MT training script
- Fix random seed when domain randomization is enabled
- Check whether simulation is running before executing pre_physics_step logic


1.1.0 - August 22, 2022
-----------------------

Additions
---------
- Additional examples: Anymal, AnymalTerrain, BallBalance, Crazyflie, FrankaCabinet, Ingenuity, Quadcopter
- Add OpenAI variantions for Feed-Forward and LSTM networks for ShadowHand
- Add domain randomization framework `using omni.replicator.isaac`
- Add AnymalTerrain interactable demo
- Automatically disable `omni.kit.window.viewport` and `omni.physx.flatcache` extensions in headless mode to improve start-up load time
- Introduce `reset_xform_properties` flag for initializing Views of cloned environments to reduce load time
- Add WandB support
- Update RL-Games version to 1.5.2

Fixes
-----
- Correctly sets simulation device for GPU simulation
- Fix omni.client import order
- Fix episode length reset condition for ShadowHand and AllegroHand


1.0.0 - June 03, 2022
----------------------
- Initial release for RL examples with Isaac Sim
- Examples provided: AllegroHand, Ant, Cartpole, Humanoid, ShadowHand