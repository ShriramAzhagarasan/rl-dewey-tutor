import gymnasium as gym
from stable_baselines3 import PPO
from src.envs.tutor_env import TutorEnv

def main():
    env = TutorEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("results/ppo_tutor")

if __name__ == "__main__":
    main()
