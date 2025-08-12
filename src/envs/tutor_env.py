import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional

class TutorEnv(gym.Env):
    """
    Enhanced Tutor Environment for RL-Dewey-Tutor
    
    Features:
    - Rich state representation (skill level, learning rate, confidence, topic mastery)
    - Dynamic difficulty adjustment based on student performance
    - Realistic student progression with forgetting curves
    - Multiple topic domains with transfer learning
    - Adaptive reward shaping (can be toggled off for ablations)
    """
    
    def __init__(self, 
                 n_topics: int = 3,
                 n_difficulty_levels: int = 5,
                 max_steps: int = 50,
                 skill_decay_rate: float = 0.02,
                 learning_rate_variance: float = 0.1,
                 reward_shaping: bool = True,
                 reward_shaping_scale: float = 1.0):
        super().__init__()
        
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.max_steps = max_steps
        self.skill_decay_rate = skill_decay_rate
        self.learning_rate_variance = learning_rate_variance
        self.reward_shaping = reward_shaping
        self.reward_shaping_scale = reward_shaping_scale
        
        # Action space: difficulty level for each topic
        self.action_space = spaces.MultiDiscrete([n_difficulty_levels] * n_topics)
        
        # State space: [skill_levels, learning_rates, confidence, topic_mastery, last_performance]
        state_dim = n_topics * 3 + 2  # 3 per topic + 2 global
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.steps = 0
        self.student_history = []
        
    def _get_state_vector(self) -> np.ndarray:
        """Convert internal state to observation vector"""
        state_list = []
        
        # Add skill levels, learning rates, and topic mastery for each topic
        for topic in range(self.n_topics):
            state_list.extend([
                self.skill_levels[topic],
                self.learning_rates[topic],
                self.topic_mastery[topic]
            ])
        
        # Add global confidence and last performance
        state_list.extend([self.global_confidence, self.last_performance])
        
        return np.array(state_list, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Initialize student state
        self.skill_levels = np.random.uniform(0.2, 0.6, self.n_topics)
        self.learning_rates = np.random.uniform(0.05, 0.15, self.n_topics)
        self.topic_mastery = np.random.uniform(0.1, 0.4, self.n_topics)
        self.global_confidence = 0.5
        self.last_performance = 0.5
        self.steps = 0
        self.student_history = []
        
        return self._get_state_vector(), {
            'skill_levels': self.skill_levels.copy(),
            'learning_rates': self.learning_rates.copy(),
            'topic_mastery': self.topic_mastery.copy()
        }
    
    def _calculate_performance(self, difficulty: float, skill: float, topic_mastery: float) -> float:
        """Calculate student performance based on difficulty, skill, and topic mastery"""
        # Base performance: skill vs difficulty
        skill_diff_gap = abs(difficulty - skill)
        base_performance = max(0, 1.0 - skill_diff_gap)
        
        # Topic mastery bonus
        mastery_bonus = topic_mastery * 0.3
        
        # Add some randomness to simulate real-world variability
        noise = np.random.normal(0, 0.05)
        
        return np.clip(base_performance + mastery_bonus + noise, 0, 1)
    
    def _update_student_state(self, action: np.ndarray, performance: np.ndarray):
        """Update student state based on performance"""
        for topic in range(self.n_topics):
            difficulty = action[topic] / (self.n_difficulty_levels - 1)
            perf = performance[topic]
            
            # Update skill level based on performance and difficulty
            if perf > 0.7:  # Good performance
                skill_gain = self.learning_rates[topic] * (1 + difficulty * 0.5)
                self.skill_levels[topic] = np.clip(
                    self.skill_levels[topic] + skill_gain, 0, 1
                )
            elif perf < 0.3:  # Poor performance
                skill_loss = self.skill_decay_rate * (1 + (1 - difficulty) * 0.5)
                self.skill_levels[topic] = np.clip(
                    self.skill_levels[topic] - skill_loss, 0, 1
                )
            
            # Update topic mastery based on consistent performance
            if perf > 0.6:
                self.topic_mastery[topic] = np.clip(
                    self.topic_mastery[topic] + 0.02, 0, 1
                )
            elif perf < 0.4:
                self.topic_mastery[topic] = np.clip(
                    self.topic_mastery[topic] - 0.01, 0, 1
                )
            
            # Learning rate adaptation
            if perf > 0.8:
                self.learning_rates[topic] = np.clip(
                    self.learning_rates[topic] * 1.01, 0.05, 0.2
                )
            elif perf < 0.2:
                self.learning_rates[topic] = np.clip(
                    self.learning_rates[topic] * 0.99, 0.05, 0.2
                )
        
        # Update global confidence based on average performance
        avg_performance = np.mean(performance)
        self.global_confidence = np.clip(
            self.global_confidence * 0.9 + avg_performance * 0.1, 0, 1
        )
        self.last_performance = avg_performance
        
        # Apply skill decay over time
        self.skill_levels = np.clip(
            self.skill_levels - self.skill_decay_rate * 0.1, 0, 1
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.steps >= self.max_steps:
            return self._get_state_vector(), 0.0, True, False, {}
        
        # Calculate performance for each topic
        performance = np.array([
            self._calculate_performance(
                action[topic] / (self.n_difficulty_levels - 1),
                self.skill_levels[topic],
                self.topic_mastery[topic]
            )
            for topic in range(self.n_topics)
        ])
        
        # Calculate reward
        if self.reward_shaping:
            reward = self._calculate_reward(action, performance) * self.reward_shaping_scale
        else:
            # Simple mean performance as reward (ablation)
            reward = float(np.mean(performance) * 2.0)
        
        # Update student state
        self._update_student_state(action, performance)
        
        # Record history
        self.student_history.append({
            'step': self.steps,
            'action': action.copy(),
            'performance': performance.copy(),
            'reward': reward,
            'skill_levels': self.skill_levels.copy(),
            'topic_mastery': self.topic_mastery.copy()
        })
        
        self.steps += 1
        terminated = self.steps >= self.max_steps
        
        # Terminal success bonus
        if terminated:
            final_skill = np.mean(self.skill_levels)
            final_mastery = np.mean(self.topic_mastery)
            if final_skill >= 0.7 and final_mastery >= 0.6:
                reward += 2.0
            elif final_skill >= 0.5 and final_mastery >= 0.4:
                reward += 1.0
        
        info = {
            'performance': performance,
            'skill_levels': self.skill_levels.copy(),
            'topic_mastery': self.topic_mastery.copy(),
            'step': self.steps
        }
        
        return self._get_state_vector(), float(reward), terminated, False, info
    
    def _calculate_reward(self, action: np.ndarray, performance: np.ndarray) -> float:
        """Calculate reward based on action and performance (shaped)"""
        reward = 0.0
        
        for topic in range(self.n_topics):
            difficulty = action[topic] / (self.n_difficulty_levels - 1)
            perf = performance[topic]
            skill = self.skill_levels[topic]
            
            # Base reward: performance quality
            reward += perf * 2.0
            
            # Difficulty appropriateness bonus
            optimal_diff = skill
            diff_penalty = -abs(difficulty - optimal_diff) * 0.5
            reward += diff_penalty
            
            # Consistency bonus for maintaining appropriate difficulty
            if 0.3 <= perf <= 0.8:
                reward += 0.3
            
            # Mastery progression bonus
            if perf > 0.8 and self.topic_mastery[topic] < 0.8:
                reward += 0.2
        
        # Global confidence bonus
        if self.global_confidence > 0.7:
            reward += 0.5
        
        return reward
    
    def render(self):
        """Render current environment state"""
        print(f"\n=== Step {self.steps}/{self.max_steps} ===")
        print(f"Global Confidence: {self.global_confidence:.3f}")
        print(f"Last Performance: {self.last_performance:.3f}")
        
        for topic in range(self.n_topics):
            print(f"Topic {topic}: Skill={self.skill_levels[topic]:.3f}, "
                  f"Learning Rate={self.learning_rates[topic]:.3f}, "
                  f"Mastery={self.topic_mastery[topic]:.3f}")
        
        if self.student_history:
            recent = self.student_history[-1]
            print(f"Last Action: {recent['action']}")
            print(f"Last Performance: {recent['performance']}")
            print(f"Last Reward: {recent['reward']:.3f}")
    
    def get_student_progress(self) -> Dict[str, Any]:
        """Get comprehensive student progress data"""
        if not self.student_history:
            return {}
        
        return {
            'total_steps': self.steps,
            'final_skill_levels': self.skill_levels.tolist(),
            'final_topic_mastery': self.topic_mastery.tolist(),
            'final_confidence': self.global_confidence,
            'performance_history': [h['performance'].tolist() for h in self.student_history],
            'reward_history': [h['reward'] for h in self.student_history],
            'skill_progression': [h['skill_levels'].tolist() for h in self.student_history]
        }
