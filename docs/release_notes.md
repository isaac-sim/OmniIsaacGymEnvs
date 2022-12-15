Release Notes
=============

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