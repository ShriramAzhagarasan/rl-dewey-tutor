"""
Baseline Heuristic Tutor for Performance Comparison

This module implements various baseline tutoring strategies to provide
comparison benchmarks for RL agents. These heuristics represent common
approaches used in traditional adaptive learning systems.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import random

class BaselineStrategy(Enum):
    """Different baseline strategies for comparison"""
    RANDOM = "random"
    FIXED_PROGRESSION = "fixed_progression"
    PERFORMANCE_BASED = "performance_based"
    ZONE_OF_PROXIMAL_DEVELOPMENT = "zpd"
    ADAPTIVE_DIFFICULTY = "adaptive_difficulty"
    MASTERY_BASED = "mastery_based"

class BaselineTutor(ABC):
    """Abstract base class for baseline tutoring strategies"""
    
    @abstractmethod
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select difficulty levels for each topic"""
        pass
    
    @abstractmethod
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Update internal strategy based on student performance"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass

class RandomTutor(BaselineTutor):
    """Random difficulty selection baseline"""
    
    def __init__(self, n_topics: int, n_difficulty_levels: int):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Randomly select difficulty levels"""
        return np.random.randint(0, self.n_difficulty_levels, size=self.n_topics)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """No learning in random strategy"""
        pass
    
    def get_name(self) -> str:
        return "Random Baseline"

class FixedProgressionTutor(BaselineTutor):
    """Fixed linear progression baseline"""
    
    def __init__(self, n_topics: int, n_difficulty_levels: int, progression_rate: float = 0.1):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.progression_rate = progression_rate
        self.current_difficulties = np.zeros(n_topics)
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select based on fixed progression"""
        return np.clip(self.current_difficulties.astype(int), 0, self.n_difficulty_levels - 1)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Fixed progression regardless of performance"""
        self.current_difficulties += self.progression_rate
        self.current_difficulties = np.clip(self.current_difficulties, 0, self.n_difficulty_levels - 1)
    
    def get_name(self) -> str:
        return "Fixed Progression Baseline"

class PerformanceBasedTutor(BaselineTutor):
    """Simple performance-based difficulty adjustment"""
    
    def __init__(self, 
                 n_topics: int, 
                 n_difficulty_levels: int,
                 success_threshold: float = 0.7,
                 failure_threshold: float = 0.3,
                 adjustment_rate: float = 0.5):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.adjustment_rate = adjustment_rate
        self.current_difficulties = np.ones(n_topics) * (n_difficulty_levels // 2)
        self.performance_history = [[] for _ in range(n_topics)]
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select based on recent performance"""
        return np.clip(self.current_difficulties.astype(int), 0, self.n_difficulty_levels - 1)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Adjust difficulty based on performance"""
        for topic in range(self.n_topics):
            self.performance_history[topic].append(performance[topic])
            
            # Keep only recent history
            if len(self.performance_history[topic]) > 10:
                self.performance_history[topic] = self.performance_history[topic][-10:]
            
            # Calculate recent average performance
            if len(self.performance_history[topic]) >= 3:
                recent_perf = np.mean(self.performance_history[topic][-3:])
                
                if recent_perf >= self.success_threshold:
                    # Increase difficulty
                    self.current_difficulties[topic] += self.adjustment_rate
                elif recent_perf <= self.failure_threshold:
                    # Decrease difficulty
                    self.current_difficulties[topic] -= self.adjustment_rate
                
                # Clip to valid range
                self.current_difficulties[topic] = np.clip(
                    self.current_difficulties[topic], 0, self.n_difficulty_levels - 1
                )
    
    def get_name(self) -> str:
        return "Performance-Based Baseline"

class ZPDTutor(BaselineTutor):
    """Zone of Proximal Development inspired tutor"""
    
    def __init__(self, 
                 n_topics: int, 
                 n_difficulty_levels: int,
                 optimal_challenge: float = 0.6,
                 zpd_range: float = 0.2):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.optimal_challenge = optimal_challenge
        self.zpd_range = zpd_range
        self.skill_estimates = np.ones(n_topics) * 0.5
        self.confidence_estimates = np.ones(n_topics) * 0.5
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select difficulty in Zone of Proximal Development"""
        difficulties = np.zeros(self.n_topics)
        
        for topic in range(self.n_topics):
            # Target difficulty slightly above current skill
            target_difficulty = self.skill_estimates[topic] + self.zpd_range
            
            # Adjust for confidence
            if self.confidence_estimates[topic] < 0.5:
                target_difficulty -= 0.1  # Easier if low confidence
            
            # Convert to discrete difficulty level
            difficulties[topic] = target_difficulty * (self.n_difficulty_levels - 1)
        
        return np.clip(difficulties.astype(int), 0, self.n_difficulty_levels - 1)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Update skill and confidence estimates"""
        for topic in range(self.n_topics):
            # Update skill estimate based on performance
            self.skill_estimates[topic] = (
                0.9 * self.skill_estimates[topic] + 0.1 * performance[topic]
            )
            
            # Update confidence based on performance consistency
            expected_perf = self.skill_estimates[topic]
            error = abs(performance[topic] - expected_perf)
            self.confidence_estimates[topic] = (
                0.9 * self.confidence_estimates[topic] + 0.1 * (1.0 - error)
            )
            
            # Clip to valid ranges
            self.skill_estimates[topic] = np.clip(self.skill_estimates[topic], 0, 1)
            self.confidence_estimates[topic] = np.clip(self.confidence_estimates[topic], 0, 1)
    
    def get_name(self) -> str:
        return "Zone of Proximal Development Baseline"

class AdaptiveDifficultyTutor(BaselineTutor):
    """More sophisticated adaptive difficulty baseline"""
    
    def __init__(self, 
                 n_topics: int, 
                 n_difficulty_levels: int,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.1):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Track performance for each topic-difficulty combination
        self.performance_matrix = np.ones((n_topics, n_difficulty_levels)) * 0.5
        self.visit_counts = np.zeros((n_topics, n_difficulty_levels))
        self.current_difficulties = np.ones(n_topics) * (n_difficulty_levels // 2)
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select difficulty using Upper Confidence Bound approach"""
        difficulties = np.zeros(self.n_topics)
        
        for topic in range(self.n_topics):
            if np.random.random() < self.exploration_rate:
                # Exploration: random difficulty
                difficulties[topic] = np.random.randint(0, self.n_difficulty_levels)
            else:
                # Exploitation: select best estimated difficulty
                ucb_values = np.zeros(self.n_difficulty_levels)
                
                for diff in range(self.n_difficulty_levels):
                    if self.visit_counts[topic, diff] > 0:
                        # UCB formula
                        avg_perf = self.performance_matrix[topic, diff]
                        confidence = np.sqrt(
                            2 * np.log(np.sum(self.visit_counts[topic])) / 
                            self.visit_counts[topic, diff]
                        )
                        ucb_values[diff] = avg_perf + confidence
                    else:
                        ucb_values[diff] = float('inf')  # Prioritize unvisited
                
                difficulties[topic] = np.argmax(ucb_values)
        
        return difficulties.astype(int)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Update performance estimates"""
        for topic in range(self.n_topics):
            diff = int(self.current_difficulties[topic])
            
            # Update visit count
            self.visit_counts[topic, diff] += 1
            
            # Update performance estimate
            old_estimate = self.performance_matrix[topic, diff]
            new_estimate = (
                old_estimate + 
                self.learning_rate * (performance[topic] - old_estimate)
            )
            self.performance_matrix[topic, diff] = new_estimate
        
        # Store current difficulties for next update
        self.current_difficulties = self.select_difficulty(student_state)
    
    def get_name(self) -> str:
        return "Adaptive Difficulty (UCB) Baseline"

class MasteryBasedTutor(BaselineTutor):
    """Mastery-based progression tutor"""
    
    def __init__(self, 
                 n_topics: int, 
                 n_difficulty_levels: int,
                 mastery_threshold: float = 0.8,
                 mastery_window: int = 5):
        self.n_topics = n_topics
        self.n_difficulty_levels = n_difficulty_levels
        self.mastery_threshold = mastery_threshold
        self.mastery_window = mastery_window
        
        self.topic_levels = np.zeros(n_topics)  # Current level for each topic
        self.performance_windows = [[] for _ in range(n_topics)]
        self.mastery_achieved = np.zeros(n_topics, dtype=bool)
    
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray:
        """Select based on mastery progression"""
        return np.clip(self.topic_levels.astype(int), 0, self.n_difficulty_levels - 1)
    
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray):
        """Update based on mastery achievement"""
        for topic in range(self.n_topics):
            self.performance_windows[topic].append(performance[topic])
            
            # Keep only recent window
            if len(self.performance_windows[topic]) > self.mastery_window:
                self.performance_windows[topic] = self.performance_windows[topic][-self.mastery_window:]
            
            # Check for mastery
            if len(self.performance_windows[topic]) >= self.mastery_window:
                avg_performance = np.mean(self.performance_windows[topic])
                
                if avg_performance >= self.mastery_threshold and not self.mastery_achieved[topic]:
                    # Mastery achieved, increase difficulty
                    self.topic_levels[topic] = min(
                        self.topic_levels[topic] + 1, 
                        self.n_difficulty_levels - 1
                    )
                    self.mastery_achieved[topic] = True
                    self.performance_windows[topic] = []  # Reset window
                elif avg_performance < self.mastery_threshold - 0.2:
                    # Struggling, decrease difficulty
                    self.topic_levels[topic] = max(self.topic_levels[topic] - 1, 0)
                    self.mastery_achieved[topic] = False
                    self.performance_windows[topic] = []  # Reset window
    
    def get_name(self) -> str:
        return "Mastery-Based Baseline"

class BaselineTutorFactory:
    """Factory for creating baseline tutors"""
    
    @staticmethod
    def create_tutor(strategy: BaselineStrategy, 
                    n_topics: int, 
                    n_difficulty_levels: int,
                    **kwargs) -> BaselineTutor:
        """
        Create a baseline tutor with specified strategy
        
        Args:
            strategy: Baseline strategy to use
            n_topics: Number of topics
            n_difficulty_levels: Number of difficulty levels
            **kwargs: Additional parameters for specific strategies
            
        Returns:
            Configured baseline tutor
        """
        if strategy == BaselineStrategy.RANDOM:
            return RandomTutor(n_topics, n_difficulty_levels)
        
        elif strategy == BaselineStrategy.FIXED_PROGRESSION:
            return FixedProgressionTutor(
                n_topics, n_difficulty_levels,
                progression_rate=kwargs.get('progression_rate', 0.1)
            )
        
        elif strategy == BaselineStrategy.PERFORMANCE_BASED:
            return PerformanceBasedTutor(
                n_topics, n_difficulty_levels,
                success_threshold=kwargs.get('success_threshold', 0.7),
                failure_threshold=kwargs.get('failure_threshold', 0.3),
                adjustment_rate=kwargs.get('adjustment_rate', 0.5)
            )
        
        elif strategy == BaselineStrategy.ZONE_OF_PROXIMAL_DEVELOPMENT:
            return ZPDTutor(
                n_topics, n_difficulty_levels,
                optimal_challenge=kwargs.get('optimal_challenge', 0.6),
                zpd_range=kwargs.get('zpd_range', 0.2)
            )
        
        elif strategy == BaselineStrategy.ADAPTIVE_DIFFICULTY:
            return AdaptiveDifficultyTutor(
                n_topics, n_difficulty_levels,
                learning_rate=kwargs.get('learning_rate', 0.1),
                exploration_rate=kwargs.get('exploration_rate', 0.1)
            )
        
        elif strategy == BaselineStrategy.MASTERY_BASED:
            return MasteryBasedTutor(
                n_topics, n_difficulty_levels,
                mastery_threshold=kwargs.get('mastery_threshold', 0.8),
                mastery_window=kwargs.get('mastery_window', 5)
            )
        
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

class BaselineEvaluator:
    """Evaluate baseline tutors for comparison with RL agents"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        Initialize evaluator
        
        Args:
            env_config: Environment configuration
        """
        self.env_config = env_config
        self.n_topics = env_config.get('n_topics', 3)
        self.n_difficulty_levels = env_config.get('n_difficulty_levels', 5)
        self.max_steps = env_config.get('max_steps', 50)
    
    def evaluate_baseline(self, 
                         strategy: BaselineStrategy,
                         n_episodes: int = 100,
                         **strategy_kwargs) -> Dict[str, Any]:
        """
        Evaluate a baseline strategy
        
        Args:
            strategy: Baseline strategy to evaluate
            n_episodes: Number of episodes to run
            **strategy_kwargs: Parameters for the strategy
            
        Returns:
            Evaluation results
        """
        from ..envs.tutor_env import TutorEnv
        
        # Create tutor and environment
        tutor = BaselineTutorFactory.create_tutor(
            strategy, self.n_topics, self.n_difficulty_levels, **strategy_kwargs
        )
        env = TutorEnv(**self.env_config)
        
        # Run evaluation episodes
        episode_rewards = []
        final_skills = []
        final_masteries = []
        difficulty_distributions = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_difficulties = []
            
            for step in range(self.max_steps):
                # Extract student state for baseline tutor
                student_state = {
                    'skill_levels': env.skill_levels.copy(),
                    'topic_mastery': env.topic_mastery.copy(),
                    'confidence': env.confidence,
                    'learning_rates': env.learning_rates.copy()
                }
                
                # Get action from baseline tutor
                action = tutor.select_difficulty(student_state)
                episode_difficulties.append(action.copy())
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Update tutor strategy
                tutor.update_strategy(student_state, info['performance'])
                
                if terminated or truncated:
                    break
            
            # Record episode results
            episode_rewards.append(episode_reward)
            final_skills.append(np.mean(env.skill_levels))
            final_masteries.append(np.mean(env.topic_mastery))
            difficulty_distributions.append(np.array(episode_difficulties))
        
        # Calculate statistics
        results = {
            'strategy_name': tutor.get_name(),
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_final_skill': float(np.mean(final_skills)),
            'std_final_skill': float(np.std(final_skills)),
            'mean_final_mastery': float(np.mean(final_masteries)),
            'std_final_mastery': float(np.std(final_masteries)),
            'reward_distribution': episode_rewards,
            'final_skill_distribution': final_skills,
            'final_mastery_distribution': final_masteries,
            'n_episodes': n_episodes,
            'difficulty_patterns': {
                'mean_difficulty_per_topic': [
                    float(np.mean([ep[step, topic] for ep in difficulty_distributions 
                                 for step in range(len(ep)) if len(ep) > step]))
                    for topic in range(self.n_topics)
                ],
                'difficulty_variance': float(np.var([
                    np.mean(ep) for ep in difficulty_distributions
                ]))
            }
        }
        
        return results
    
    def compare_all_baselines(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Compare all baseline strategies
        
        Args:
            n_episodes: Number of episodes per strategy
            
        Returns:
            Comparison results
        """
        results = {}
        
        for strategy in BaselineStrategy:
            print(f"Evaluating {strategy.value} baseline...")
            try:
                results[strategy.value] = self.evaluate_baseline(strategy, n_episodes)
            except Exception as e:
                print(f"Error evaluating {strategy.value}: {str(e)}")
                results[strategy.value] = {'error': str(e)}
        
        # Add comparison statistics
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            comparison = {
                'best_mean_reward': max(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['mean_reward']),
                'best_final_skill': max(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['mean_final_skill']),
                'best_final_mastery': max(valid_results.keys(), 
                                        key=lambda k: valid_results[k]['mean_final_mastery']),
                'most_stable': min(valid_results.keys(), 
                                 key=lambda k: valid_results[k]['std_reward']),
                'performance_ranking': sorted(valid_results.keys(), 
                                            key=lambda k: valid_results[k]['mean_reward'], 
                                            reverse=True)
            }
            results['comparison_summary'] = comparison
        
        return results
