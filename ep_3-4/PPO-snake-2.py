import gym
from stable_baselines3 import PPO
import os

from snakeenv2 import SnekEnv


models_dir = "models/PPO-v2"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SnekEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10_000
for i in range(1, 100_000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO-v2")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()