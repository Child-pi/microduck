from gymnasium.envs.registration import register

register(
    id="Microduck-v0",
    entry_point="envs.microduck_env:MicroduckEnv",
    max_episode_steps=1500,
)
