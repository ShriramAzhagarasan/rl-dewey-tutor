import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TutorEnv(gym.Env):
    def __init__(self):
        super(TutorEnv, self).__init__()
        # Actions: 0 = Easy Question, 1 = Medium, 2 = Hard
        self.action_space = spaces.Discrete(3)
        # State: student skill level (0–1) + last question difficulty
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.state = np.array([0.5, 0.0], dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.5, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        skill, _ = self.state
        difficulty = action / 2.0
        reward = -abs(difficulty - skill) + np.random.normal(0, 0.05)
        skill = np.clip(skill + (0.05 if reward > 0 else -0.02), 0, 1)
        self.state = np.array([skill, difficulty], dtype=np.float32)
        self.steps += 1
        terminated = self.steps >= 20
        return self.state, reward, terminated, False, {}

    def render(self):
        print(f"Step {self.steps}: skill={self.state[0]:.2f}, last diff={self.state[1]:.2f}")
