import math
import torch
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole

class MyCartpoleTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """
        name: parsed from task config yaml file
        sim_config: SimConfig obj with task and physics params
        env: environment obj from rlgames_train.py
        all parsed by task_util.py
        """
        self.update_config(sim_config)
        self._max_episode_length = 500

        # must be defined in task class
        self._num_observations = 4      # cart pos, cart vel, pole ang, pole ang vel
        self._num_actions = 1           # apply force to cart

        # call parent class constructor for RL variables
        RLTask.__init__(self, name, env)
        return
    
    def update_config(self, sim_config):
        # extract task config from main config dictionary
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # parse task config parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])        # x, y, z?

        # reset and actions related variables
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        return

    def set_up_scene(self, scene) -> None:
        # first create a single environment
        self.get_cartpole()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView object to hold our collection of environments
        self._cartpoles = ArticulationView(                             # NEED TO REVIEW
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )

        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._cartpoles)
        return

    def get_cartpole(self):
        # add a single robot to the stage
        cartpole = Cartpole(                                            # NEED TO REVIEW
            prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole")
        )
        return
    
    def post_reset(self):
        # retrieve cart and pole joint indices
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")

        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        return

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply randomized joint positions and velocities to environments
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # reset the reset buffer and progress buffer after applying reset
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions) -> None:
        # make sure simulation has not been stopped from the UI
        if not self._env._world.is_playing():
            return

        # extract environment indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # make sure actions buffer is on the same device as the simulation
        actions = actions.to(self._device)

        # compute forces from the actions
        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        # apply actions to all of the environments
        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)
        return

    def get_observations(self) -> dict:
        # retrieve joint positions and velocities
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        # extract joint states for the cart and pole joints
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        # populate the observations buffer
        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_pos
        self.obs_buf[:, 3] = pole_vel

        # construct the observations dictionary and return
        observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def calculate_metrics(self) -> None:
        # use states from the observation buffer to compute reward
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        # define the reward function based on pole angle and robot velocities
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.5 * torch.abs(pole_vel)
        # penalize the policy if the cart moves too far on the rail
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # penalize the policy if the pole moves beyond 90 degrees
        reward = torch.where(torch.abs(pole_angle) > math.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # assign rewards to the reward buffer
        self.rew_buf[:] = reward
        return

    def is_done(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        pole_pos = self.obs_buf[:, 2]

        # check for which conditions are met and mark the environments that satisfy the conditions
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        # assign the resets to the reset buffer
        self.reset_buf[:] = resets
        return