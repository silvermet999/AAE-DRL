from gymnasium.envs.registration import register

register(
    id="AAE_env_v0",
    entry_point="gym_setup.gym_env.env:AAE_env",
)
