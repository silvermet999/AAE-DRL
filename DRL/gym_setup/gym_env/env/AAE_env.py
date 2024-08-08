import gymnasium as gym
from gymnasium import spaces
import numpy as np
from AAE import AAE_archi
import json




class AAE_env(gym.Env):

    def __init__(self):

        self.observation_space = AAE_archi.reparameterization(0, 1, AAE_archi.z_dim) # this is the shape not the actual vector

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
        return {"agent": self._agent_location, "target": self._target_location}
    # loss
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def close(self):
        pass


