import gym
from stable_baselines3 import PPO

from snakeenv2 import SnekEnv

env = SnekEnv()
env.reset()

model_dir = "models/PPO-v2"
model_path = f"{model_dir}/900000"

model= PPO.load(model_path, env=env)


episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)

env.close()