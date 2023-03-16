Release Notes
=============

2022.2.1.0 - March 16, 2023
---------------------------

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