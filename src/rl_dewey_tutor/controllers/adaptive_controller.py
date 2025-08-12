"""
Adaptive Controller for Dynamic Agent Orchestration

This module implements sophisticated controller logic with fallback strategies,
dynamic agent selection, and error recovery mechanisms.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from enum import Enum

class AgentType(Enum):
    PPO = "ppo"
    QLEARNING = "qlearning"
    HYBRID = "hybrid"

class ControllerState(Enum):
    NORMAL = "normal"
    FALLBACK = "fallback"
    RECOVERY = "recovery"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetrics:
    """Performance metrics for agent evaluation"""
    mean_reward: float
    reward_std: float
    convergence_rate: float
    stability_score: float
    exploration_efficiency: float
    learning_progress: float

class AdaptiveController:
    """
    Sophisticated controller that orchestrates multiple RL agents with:
    - Dynamic agent selection based on performance
    - Fallback strategies for failed training
    - Error recovery mechanisms
    - Performance monitoring and adaptation
    """
    
    def __init__(self, 
                 performance_window: int = 100,
                 stability_threshold: float = 0.1,
                 convergence_threshold: float = 0.05,
                 fallback_timeout: float = 300.0):
        """
        Initialize the adaptive controller
        
        Args:
            performance_window: Number of episodes to consider for performance evaluation
            stability_threshold: Threshold for reward stability (lower = more stable)
            convergence_threshold: Threshold for convergence detection
            fallback_timeout: Time limit before triggering fallback (seconds)
        """
        self.performance_window = performance_window
        self.stability_threshold = stability_threshold
        self.convergence_threshold = convergence_threshold
        self.fallback_timeout = fallback_timeout
        
        # State tracking
        self.state = ControllerState.NORMAL
        self.active_agent = AgentType.PPO
        self.performance_history: Dict[AgentType, List[float]] = {
            AgentType.PPO: [],
            AgentType.QLEARNING: []
        }
        self.error_counts: Dict[AgentType, int] = {
            AgentType.PPO: 0,
            AgentType.QLEARNING: 0
        }
        self.last_switch_time = time.time()
        self.training_start_time = None
        
        # Fallback strategies
        self.fallback_strategies = [
            self._reduce_learning_rate,
            self._increase_exploration,
            self._simplify_environment,
            self._switch_agent,
            self._emergency_reset
        ]
        self.current_fallback_index = 0
        
    def select_agent(self, current_performance: Dict[str, float]) -> AgentType:
        """
        Dynamically select the best agent based on current performance
        
        Args:
            current_performance: Current performance metrics
            
        Returns:
            Selected agent type
        """
        if self.state == ControllerState.EMERGENCY:
            return self._emergency_agent_selection()
            
        # Update performance history
        reward = current_performance.get('reward', 0.0)
        self.performance_history[self.active_agent].append(reward)
        
        # Limit history size
        if len(self.performance_history[self.active_agent]) > self.performance_window:
            self.performance_history[self.active_agent] = \
                self.performance_history[self.active_agent][-self.performance_window:]
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(self.active_agent)
        
        # Decision logic
        if self._should_switch_agent(metrics):
            new_agent = self._get_best_alternative_agent()
            if new_agent != self.active_agent:
                print(f"Controller: Switching from {self.active_agent.value} to {new_agent.value}")
                print(f"Reason: {self._get_switch_reason(metrics)}")
                self.active_agent = new_agent
                self.last_switch_time = time.time()
                
        return self.active_agent
    
    def handle_training_error(self, agent_type: AgentType, error: Exception) -> Dict[str, Any]:
        """
        Handle training errors with sophisticated fallback strategies
        
        Args:
            agent_type: Agent that encountered the error
            error: The exception that occurred
            
        Returns:
            Recovery strategy configuration
        """
        self.error_counts[agent_type] += 1
        error_count = self.error_counts[agent_type]
        
        print(f"Controller: Handling error for {agent_type.value} (count: {error_count})")
        print(f"Error: {str(error)}")
        
        # Escalate based on error frequency
        if error_count == 1:
            self.state = ControllerState.FALLBACK
            return self._apply_fallback_strategy()
        elif error_count <= 3:
            self.state = ControllerState.RECOVERY
            return self._apply_recovery_strategy()
        else:
            self.state = ControllerState.EMERGENCY
            return self._apply_emergency_strategy()
    
    def _calculate_performance_metrics(self, agent_type: AgentType) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for an agent"""
        history = self.performance_history[agent_type]
        
        if len(history) < 10:
            return PerformanceMetrics(0, 1, 0, 0, 0, 0)
        
        rewards = np.array(history)
        
        # Basic metrics
        mean_reward = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        
        # Convergence rate (slope of recent performance)
        if len(rewards) >= 20:
            x = np.arange(len(rewards))
            convergence_rate = float(np.polyfit(x, rewards, 1)[0])
        else:
            convergence_rate = 0.0
        
        # Stability score (inverse of coefficient of variation)
        if mean_reward > 0:
            stability_score = 1.0 / (1.0 + reward_std / mean_reward)
        else:
            stability_score = 0.0
        
        # Exploration efficiency (improvement over time)
        if len(rewards) >= 20:
            early_mean = np.mean(rewards[:10])
            late_mean = np.mean(rewards[-10:])
            exploration_efficiency = max(0, late_mean - early_mean)
        else:
            exploration_efficiency = 0.0
        
        # Learning progress (trend analysis)
        if len(rewards) >= 30:
            third = len(rewards) // 3
            first_third = np.mean(rewards[:third])
            last_third = np.mean(rewards[-third:])
            learning_progress = max(0, last_third - first_third)
        else:
            learning_progress = 0.0
        
        return PerformanceMetrics(
            mean_reward=mean_reward,
            reward_std=reward_std,
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            exploration_efficiency=exploration_efficiency,
            learning_progress=learning_progress
        )
    
    def _should_switch_agent(self, metrics: PerformanceMetrics) -> bool:
        """Determine if agent switching is beneficial"""
        # Avoid too frequent switching
        if time.time() - self.last_switch_time < 60:  # 1 minute cooldown
            return False
        
        # Switch if performance is poor
        if metrics.mean_reward < 0.1:
            return True
        
        # Switch if convergence stalled
        if abs(metrics.convergence_rate) < self.convergence_threshold and len(self.performance_history[self.active_agent]) > 50:
            return True
        
        # Switch if unstable
        if metrics.stability_score < self.stability_threshold:
            return True
        
        return False
    
    def _get_best_alternative_agent(self) -> AgentType:
        """Select the best alternative agent"""
        if self.active_agent == AgentType.PPO:
            return AgentType.QLEARNING
        else:
            return AgentType.PPO
    
    def _get_switch_reason(self, metrics: PerformanceMetrics) -> str:
        """Get human-readable reason for agent switch"""
        if metrics.mean_reward < 0.1:
            return f"Poor performance (reward: {metrics.mean_reward:.3f})"
        elif abs(metrics.convergence_rate) < self.convergence_threshold:
            return f"Stalled convergence (rate: {metrics.convergence_rate:.6f})"
        elif metrics.stability_score < self.stability_threshold:
            return f"Unstable learning (stability: {metrics.stability_score:.3f})"
        else:
            return "Proactive optimization"
    
    def _apply_fallback_strategy(self) -> Dict[str, Any]:
        """Apply first-level fallback strategy"""
        strategy_func = self.fallback_strategies[self.current_fallback_index]
        config = strategy_func()
        print(f"Controller: Applying fallback strategy {self.current_fallback_index + 1}: {config['name']}")
        return config
    
    def _apply_recovery_strategy(self) -> Dict[str, Any]:
        """Apply more aggressive recovery strategy"""
        self.current_fallback_index = min(self.current_fallback_index + 1, 
                                         len(self.fallback_strategies) - 2)
        return self._apply_fallback_strategy()
    
    def _apply_emergency_strategy(self) -> Dict[str, Any]:
        """Apply emergency strategy as last resort"""
        self.current_fallback_index = len(self.fallback_strategies) - 1
        return self._apply_fallback_strategy()
    
    def _emergency_agent_selection(self) -> AgentType:
        """Emergency agent selection with minimal requirements"""
        # Use the agent with fewer errors
        if self.error_counts[AgentType.PPO] <= self.error_counts[AgentType.QLEARNING]:
            return AgentType.PPO
        else:
            return AgentType.QLEARNING
    
    # Fallback strategy implementations
    def _reduce_learning_rate(self) -> Dict[str, Any]:
        """Reduce learning rate to improve stability"""
        return {
            'name': 'Reduce Learning Rate',
            'action': 'modify_config',
            'changes': {
                'learning_rate': 0.5,  # Multiply current by 0.5
                'batch_size': 1.5      # Increase batch size for stability
            }
        }
    
    def _increase_exploration(self) -> Dict[str, Any]:
        """Increase exploration to escape local optima"""
        return {
            'name': 'Increase Exploration',
            'action': 'modify_config',
            'changes': {
                'epsilon': 1.5,        # Increase epsilon for Q-learning
                'ent_coef': 1.5,       # Increase entropy coefficient for PPO
                'exploration_fraction': 1.2
            }
        }
    
    def _simplify_environment(self) -> Dict[str, Any]:
        """Simplify environment to reduce complexity"""
        return {
            'name': 'Simplify Environment',
            'action': 'modify_env',
            'changes': {
                'n_topics': 2,         # Reduce number of topics
                'max_steps': 30,       # Shorter episodes
                'reward_shaping_scale': 0.5  # Simpler rewards
            }
        }
    
    def _switch_agent(self) -> Dict[str, Any]:
        """Switch to alternative agent"""
        new_agent = self._get_best_alternative_agent()
        self.active_agent = new_agent
        return {
            'name': f'Switch to {new_agent.value}',
            'action': 'switch_agent',
            'agent': new_agent.value
        }
    
    def _emergency_reset(self) -> Dict[str, Any]:
        """Emergency reset with minimal configuration"""
        return {
            'name': 'Emergency Reset',
            'action': 'reset',
            'config': 'minimal'
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        report = {
            'controller_state': self.state.value,
            'active_agent': self.active_agent.value,
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'performance_summary': {}
        }
        
        # Add performance metrics for each agent
        for agent_type in AgentType:
            if agent_type in self.performance_history:
                metrics = self._calculate_performance_metrics(agent_type)
                report['performance_summary'][agent_type.value] = {
                    'mean_reward': metrics.mean_reward,
                    'stability_score': metrics.stability_score,
                    'convergence_rate': metrics.convergence_rate,
                    'sample_count': len(self.performance_history[agent_type])
                }
        
        return report
    
    def reset(self):
        """Reset controller state"""
        self.state = ControllerState.NORMAL
        self.active_agent = AgentType.PPO
        self.performance_history = {AgentType.PPO: [], AgentType.QLEARNING: []}
        self.error_counts = {AgentType.PPO: 0, AgentType.QLEARNING: 0}
        self.current_fallback_index = 0
        self.last_switch_time = time.time()
        print("Controller: Reset to initial state")
