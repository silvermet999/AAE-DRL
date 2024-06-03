from gym_setup.gym_env.env.AAE_env import AAE_env
import gymnasium as gym
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.core.learner.learner import Learner
import shutil


def main():
    chkpt_root = "temp"
    """shutil.rmtree is a function that removes a directory tree recursively. It takes a path-like object or a string 
    as an argument and returns nothing."""
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
    # add local_mode
    ray.init(ignore_reinit_error=True)
    # register env
    select_env = "AAE_env"
    register_env(select_env, lambda config: AAE_env)
    print(select_env)

    config = PPOConfig().environment(select_env)
    config.framework("tf2")
    # config.model_config = {
    #     "fcnet_hiddens": [64, 64],
    #     "fcnet_activation": "relu"
    # }
    # config["lr"] = 0.005
    # config["model"]["custom_model_config"] = {
    #     "fcnet_hiddens": [64, 64],
    #     "fcnet_activation": "relu"
    # }
    agent = PPO(config=config)
    print(agent)
    agent.train()
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
        ))

    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    # rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)
    observation = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_actions(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        env.render()

        if terminated or truncated:
            print("sum rewards: ", sum_reward)
            observation = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()