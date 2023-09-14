from omni.isaac.gym.vec_env import VecEnvBase

from omni.isaac.kit import SimulationApp
from omni.isaac.gym.vec_env import VecEnvBase
import os
import carb
import gym


class VecEnvBaseStream(VecEnvBase):
    """ This class provides a base interface for connecting RL policies with task implementations.
        APIs provided in this interface follow the interface in gym.Env.
        This class also provides utilities for initializing simulation apps, creating the World,
        and registering a task.
    """

    def __init__(
        self, headless: bool, sim_device: int = 0, enable_livestream: bool = False, enable_viewport: bool = False,stream_type: str = "webRTC"
    ) -> None:
        """ Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
        """

        experience = ""
        if headless:
            if enable_livestream:
                experience = ""
            elif enable_viewport:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.render.kit'
            else:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp({"headless": headless, "physics_gpu": sim_device}, experience=experience)
        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._render = not headless or enable_livestream or enable_viewport
        self.sim_frame_count = 0

        if enable_livestream:
            from omni.isaac.core.utils.extensions import enable_extension
            if stream_type == "webRTC":
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.services.streamclient.webrtc")
            elif stream_type == "native":
                self._simulation_app.set_setting("/app/livestream/enabled", True)
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.kit.livestream.native")
                enable_extension("omni.services.streaming.manager")
            else:
                raise NotImplementedError("unsopported stream type")