"""
Thompson Sampling Exploration Strategy for RL-Dewey-Tutor

This module implements Thompson Sampling for adaptive exploration in the tutor environment,
allowing the agent to balance exploration and exploitation based on uncertainty.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings

class ThompsonSamplingExplorer:
    """
    Thompson Sampling Explorer for Adaptive Exploration
    
    Features:
    - Bayesian uncertainty quantification
    - Adaptive exploration based on state uncertainty
    - Integration with both PPO and Q-Learning agents
    - Confidence-based action selection
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dims: List[int],
                 exploration_strength: float = 1.0,
                 uncertainty_threshold: float = 0.1,
                 min_exploration: float = 0.05,
                 max_exploration: float = 0.3):
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.total_actions = int(np.prod(action_dims))
        
        # Exploration parameters
        self.exploration_strength = exploration_strength
        self.uncertainty_threshold = uncertainty_threshold
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration
        
        # Uncertainty tracking
        self.state_action_uncertainty: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Thompson Sampling parameters (Beta prior)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0
        
        # History
        self.exploration_history = []
    
    def _summarize_state(self, state: np.ndarray) -> np.ndarray:
        """Reduce high-dimensional continuous state to a compact summary.
        Expect state to be [skill1, lr1, mastery1, skill2, lr2, mastery2, ..., conf, last_perf].
        We keep per-topic skills & mastery averages and overall confidence/perf.
        """
        state = np.asarray(state, dtype=np.float32)
        # Heuristic: assume 3 features per topic; derive n_topics from length-2 (conf, perf)
        if len(state) >= 8:
            # guess topics
            n_topics = (len(state) - 2) // 3
            skills = state[0:3*n_topics:3]
            mastery = state[2:3*n_topics:3]
            conf, last_perf = state[-2], state[-1]
            # summary vector: mean skill, std skill, mean mastery, conf, last_perf
            return np.array([
                float(np.mean(skills)),
                float(np.std(skills)),
                float(np.mean(mastery)),
                float(conf),
                float(last_perf)
            ], dtype=np.float32)
        # Fallback: take first 5 dims
        take = min(5, len(state))
        return state[:take]
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert summarized state to a coarse discrete key."""
        s = self._summarize_state(state)
        # Coarser rounding so nearby states collide and visits accumulate
        discretized = np.round(s / 0.2) * 0.2
        return str(discretized.tolist())
    
    def _action_to_index(self, action: np.ndarray) -> int:
        idx = 0
        for i, a in enumerate(action):
            idx += int(a) * int(np.prod(self.action_dims[i+1:])) if i < len(self.action_dims) - 1 else int(a)
        return int(idx)
    
    def update_uncertainty(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        key = self._state_to_key(state)
        a_idx = self._action_to_index(action)
        
        if key not in self.state_action_uncertainty:
            self.state_action_uncertainty[key] = {
                'alpha': np.ones(self.total_actions, dtype=np.float32) * self.alpha_prior,
                'beta': np.ones(self.total_actions, dtype=np.float32) * self.beta_prior,
                'visits': np.zeros(self.total_actions, dtype=np.float32)
            }
        rec = self.state_action_uncertainty[key]
        rec['visits'][a_idx] += 1.0
        # Normalize reward to [0,1] with a smoother scale (assume roughly [-5, 10])
        norm_r = np.clip((reward + 5.0) / 15.0, 0.0, 1.0)
        rec['alpha'][a_idx] += norm_r
        rec['beta'][a_idx] += (1.0 - norm_r)
    
    def _beta_variance(self, alpha: float, beta: float) -> float:
        denom = (alpha + beta) ** 2 * (alpha + beta + 1.0)
        if denom <= 0:
            return 1.0
        return float((alpha * beta) / denom)
    
    def get_exploration_probability(self, state: np.ndarray, base_exploration: float = 0.1) -> float:
        key = self._state_to_key(state)
        if key not in self.state_action_uncertainty:
            return self.max_exploration
        rec = self.state_action_uncertainty[key]
        # Average beta variance across actions
        variances = [self._beta_variance(a, b) for a, b in zip(rec['alpha'], rec['beta'])]
        state_uncertainty = float(np.mean(variances))
        # Visit-based decay (more visits => lower exploration)
        total_visits = float(np.sum(rec['visits']))
        visit_factor = 1.0 / (1.0 + total_visits / 20.0)
        combined = 0.7 * state_uncertainty + 0.3 * visit_factor
        prob = base_exploration + combined * self.exploration_strength
        return float(np.clip(prob, self.min_exploration, self.max_exploration))
    
    def sample_action_thompson(self, state: np.ndarray, q_values: Optional[np.ndarray] = None) -> np.ndarray:
        key = self._state_to_key(state)
        if key not in self.state_action_uncertainty:
            return np.array([np.random.randint(0, int(dim)) for dim in self.action_dims])
        rec = self.state_action_uncertainty[key]
        samples = np.random.beta(rec['alpha'], rec['beta'])
        if q_values is not None:
            samples = 0.3 * samples + 0.7 * q_values
        best = int(np.argmax(samples))
        # convert flat index back to multi-discrete
        action = []
        remaining = best
        for i, dim in enumerate(self.action_dims):
            if i < len(self.action_dims) - 1:
                divisor = int(np.prod(self.action_dims[i+1:]))
                action.append(remaining // divisor)
                remaining = remaining % divisor
            else:
                action.append(remaining)
        return np.array(action)
    
    def get_uncertainty_map(self, state: np.ndarray) -> Dict[str, float]:
        """Get uncertainty information for a given state"""
        key = self._state_to_key(state)
        
        if key not in self.state_action_uncertainty:
            return {
                'state_uncertainty': 1.0,
                'action_uncertainties': np.ones(self.total_actions).tolist(),
                'exploration_probability': self.max_exploration
            }
        
        rec = self.state_action_uncertainty[key]
        
        # Calculate uncertainties
        action_uncertainties = [
            self._beta_variance(alpha, beta)
            for alpha, beta in zip(rec['alpha'], rec['beta'])
        ]
        
        state_uncertainty = np.mean(action_uncertainties)
        exploration_prob = self.get_exploration_probability(state)
        
        return {
            'state_uncertainty': state_uncertainty,
            'action_uncertainties': action_uncertainties,
            'exploration_probability': exploration_prob,
            'total_visits': int(np.sum(rec['visits']))
        }
    
    def record_exploration_decision(self, state: np.ndarray, action: np.ndarray,
                                  exploration_prob: float, uncertainty: float):
        """Record exploration decision for analysis"""
        self.exploration_history.append({
            'state': state.copy(),
            'action': action.copy(),
            'exploration_probability': exploration_prob,
            'uncertainty': uncertainty
        })
    
    def get_exploration_stats(self) -> Dict[str, List]:
        """Get exploration statistics for analysis"""
        if not self.exploration_history:
            return {}
        
        return {
            'exploration_probabilities': [h['exploration_probability'] for h in self.exploration_history],
            'uncertainties': [h['uncertainty'] for h in self.exploration_history],
            'total_decisions': len(self.exploration_history)
        }
    
    def reset(self):
        self.exploration_history = [] 