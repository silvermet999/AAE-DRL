import gymnasium as gym
from gym_example.gym_example.envs import grid_world

# wrapped_env = DiscreteActions(env, env.action_space)
# print(wrapped_env.reset())
""" an action space is the set of all possible actions that an agent can take in a given environment. It defines the range of actions that can be performed.
    setting the seed for the random number generator associated with the action space."""
# env.action_space.seed(42)

def run_episodes(env, verbose = False):
    env.reset()
    sum_reward = 0
    for i in range(10):
        terminated = False
        truncated = False
        """env.action_space.sample(): gen random sample from the action space of the environment.
        env.step(): called on the environment to take a step in the simulation by providing an action
        observation: Represents the new state or observation of the environment after taking the specified action.
        teminated: true or false
        truncated: true or false
        info: metadata"""
        while not (terminated or truncated):
            action = env.action_space.sample()
            if verbose:
                print("action: ", action)

            observation, reward, terminated, truncated, info = env.step(action)
            sum_reward += reward
            if verbose:
                env.render()

            if terminated or truncated:
                if verbose:
                    print("done at step: ".format(i))
                break

        if verbose:
            print('sum rewards: ', sum_reward)



def main():
    env = gym.make('GridWorld-v0')
    sum_reward = run_episodes(env, verbose=True)

    # calculate a statistical baseline of rewards based on random actions
    history = []
    for _ in range(100):
        sum_reward = run_episodes(env, verbose=False)
        history.append(sum_reward)
    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))

if __name__ == "__main__":
    main()
