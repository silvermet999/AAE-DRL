import gymnasium as gym
from gym.spaces import Box
from gymnasium import spaces
import numpy as np
from AAE import AAE_archi
from AAE import AAE_training_testing

import torch



class AAE_env(gym.Env):

    def __init__(self):
        # Box is used to define a continuous space with bounds. If the latent space is not bounded, you can set the bounds to -float('inf') and float('inf')
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(AAE_archi.z_dim,), dtype=np.float32)

        # We have 2 actions, corresponding to parameter updates and architecture change
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: AAE_archi.hyperparams_g,
            1: AAE_archi.hyperparams_d,
            2: AAE_archi.encoder_generator
        }

    # Agent prediction, true prediction
    def _get_obs(self):
        return {"obs": self.init_observation, "agent": self._agent_location, "target": self._target_location}

    # loss
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _calculate_reward(self):
        distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        reward = -distance
        if distance < self.goal_tolerance:
            reward += self.goal_reward
        return reward

    def reset(self, seed=None, options=None):
        self.mu = AAE_training_testing.mu
        self.logvar = AAE_training_testing.logvar
        self.init_observation = AAE_archi.reparameterization(self.mu, self.logvar, AAE_archi.z_dim)
        self._agent_location = np.random.uniform(low=-1, high=1, size=(AAE_archi.z_dim,))

        self._target_location = np.random.uniform(low=-1, high=1, size=(AAE_archi.z_dim,))
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.random.uniform(low=-1, high=1, size=(AAE_archi.z_dim,))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self._agent_location += action
        reward = self._calculate_reward()
        terminated = np.linalg.norm(self._agent_location - self._target_location, ord=1) < self.goal_tolerance
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, info

    def close(self):
        pass


