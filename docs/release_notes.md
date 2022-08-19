Release Notes
=============

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