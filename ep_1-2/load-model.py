import gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
env.reset()

model_dir = "models/PPO"
model_path = f"{model_dir}/240000.zip"

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