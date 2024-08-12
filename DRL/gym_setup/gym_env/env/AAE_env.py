import gymnasium as gym
from gymnasium import spaces
import numpy as np
from AAE import AAE_archi
from AAE import AAE_training_testing

import torch.nn as nn



class AAE_env(gym.Env):

    def __init__(self):
        # Box is used to define a continuous space with bounds. If the latent space is not bounded, you can set the bounds to -float('inf') and float('inf')
        self.observation_space = spaces.Dict(
            {
            "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(AAE_archi.z_dim,), dtype=np.float32),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(AAE_archi.z_dim,), dtype=np.float32)
        }
        )

        # We have 2 actions, corresponding to parameter updates and architecture change
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: AAE_archi.hyperparams_g,
            1: AAE_archi.encoder_generator
        }
        self.goal_tolerance = 0.1
        self.goal_reward = 10.0
        self.max_steps = 1000
        self.current_step = 0
        self.hyperparams = {
            'lr': AAE_archi.hyperparams_g["lr"],
            'beta1': AAE_archi.hyperparams_g["beta1"],
            'beta2' : AAE_archi.hyperparams_g["beta2"]
        }

        self.decrease_factor = 0.5
        self.increase_factor = 1.5
        self.new_layers = nn.Sequential(
            nn.Linear(in_features=310, out_features=310),
            nn.ReLU(),
            nn.BatchNorm1d(310)
        )


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

    def _calculate_reward(self):
        distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        reward = -distance
        if distance < self.goal_tolerance:
            reward += self.goal_reward
        return reward

    def update_action(self):
        # training high and unstable, val is high
        if (AAE_training_testing.avg_g_loss > 0.4 and
                AAE_training_testing.std_g_loss > 0.1 * AAE_training_testing.avg_g_loss and
                AAE_training_testing.avg_recon_loss > 0.4):
            current_lr = AAE_archi.hyperparams_g["lr"]
            new_lr = current_lr * self.decrease_factor
            AAE_archi.hyperparams_g["lr"] = new_lr
            print(f"Learning rate decreased to: {new_lr}")

        # training not moving, val high
        elif (AAE_training_testing.avg_recon_loss > 0.4 and
              AAE_training_testing.std_g_loss < 0.0001 * AAE_training_testing.avg_g_loss):
            current_lr = AAE_archi.hyperparams_g["lr"]
            new_lr = current_lr * self.increase_factor
            AAE_archi.hyperparams_g["lr"] = new_lr
            print(f"Learning rate increased to: {new_lr}")

        # training unstable, val high
        elif (AAE_training_testing.std_g_loss > 0.1 * AAE_training_testing.avg_g_loss and
              AAE_training_testing.avg_recon_loss > 0.4):
            current_b1 = AAE_archi.hyperparams_g["beta1"]
            new_b1 = current_b1 * self.increase_factor
            AAE_archi.hyperparams_g["beta1"] = new_b1
            print(f"beta1 increased to: {new_b1}")

        # training and loss not moving
        elif (AAE_training_testing.std_g_loss < 0.0001 * AAE_training_testing.avg_g_loss and
              AAE_training_testing.std_recon_loss < 0.0001 * AAE_training_testing.avg_recon_loss):
            current_b1 = AAE_archi.hyperparams_g["beta1"]
            new_b1 = current_b1 * self.decrease_factor
            AAE_archi.hyperparams_g["beta1"] = new_b1
            print(f"beta1 decreased to: {new_b1}")

        # training not moving, val is low
        elif (AAE_training_testing.avg_recon_loss < 0.05 and
              AAE_training_testing.std_g_loss < 0.0001 * AAE_training_testing.avg_g_loss):
            current_beta2 = AAE_archi.hyperparams_g["beta2"]
            new_beta2 = current_beta2 * self.decrease_factor
            AAE_archi.hyperparams_g["beta2"] = new_beta2
            print(f"beta2 decreased to: {new_beta2}")

        # training and val unstable
        elif (AAE_training_testing.std_g_loss > 0.1 * AAE_training_testing.avg_g_loss and
              AAE_training_testing.std_recon_loss > 0.1 * AAE_training_testing.avg_recon_loss):
            current_beta2 = AAE_archi.hyperparams_g["beta2"]
            new_beta2 = current_beta2 * self.increase_factor
            AAE_archi.hyperparams_g["beta2"] = new_beta2
            print(f"beta2 increased to: {new_beta2}")

        # training and val high
        if (AAE_training_testing.avg_g_loss > 0.4 and AAE_training_testing.avg_recon_loss > 0.4):
            if isinstance(AAE_archi.encoder_generator, nn.Sequential):
                updated_layers = nn.Sequential(
                    *list(AAE_archi.encoder_generator.children()) + list(self.new_layers.children())
                )
                AAE_archi.encoder_generator = updated_layers
                print(f"Layer number increased to: {len(updated_layers)}")

        # training low, val high
        if (AAE_training_testing.avg_g_loss < 0.4 and AAE_training_testing.avg_recon_loss > 0.4):
            current_lr = AAE_archi.encoder_generator
            new_layers = current_lr - self.layers
            AAE_archi.encoder_generator = new_layers
            print(f"Layer number decreased to: {new_layers}")

        else:
            pass

    def reset(self, seed=None, options=None):
        self.current_step = 0
        mu = AAE_training_testing.mu
        logvar = AAE_training_testing.logvar
        self._agent_location = AAE_archi.reparameterization(mu, logvar, AAE_archi.z_dim).cpu().detach().numpy()

        self._target_location = np.random.uniform(low=-1, high=1, size=(AAE_archi.z_dim,))
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.random.uniform(low=-1, high=1, size=(AAE_archi.z_dim,))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info



    def step(self, action):
        update_params_layers = self._action_to_direction.get(action)
        print(update_params_layers)

        if update_params_layers:
            self.update_action()

        self.current_step += 1
        reward = self._calculate_reward()
        terminated = np.linalg.norm(self._agent_location - self._target_location, ord=1) < self.goal_tolerance
        truncated = self.current_step >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info



