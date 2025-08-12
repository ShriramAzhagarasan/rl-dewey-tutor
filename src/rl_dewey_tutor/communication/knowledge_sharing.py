"""
Inter-Agent Communication and Knowledge Sharing System

This module implements sophisticated communication protocols between RL agents,
enabling knowledge transfer, shared uncertainty estimation, and collaborative learning.
"""

import numpy as np
import torch
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import time
import json

@dataclass
class Experience:
    """Shared experience structure"""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    agent_type: str
    timestamp: float
    confidence: float = 1.0

@dataclass
class ValueEstimate:
    """Value estimate from an agent"""
    state: np.ndarray
    value: float
    uncertainty: float
    agent_type: str
    timestamp: float

@dataclass
class PolicyInsight:
    """Policy insight for knowledge transfer"""
    state_features: np.ndarray
    action_preferences: np.ndarray
    confidence: float
    success_rate: float
    agent_type: str

class SharedKnowledgeBase:
    """
    Centralized knowledge base for inter-agent communication
    
    Features:
    - Experience sharing and replay
    - Value function alignment
    - Uncertainty propagation
    - Policy knowledge transfer
    - Performance feedback loops
    """
    
    def __init__(self, 
                 max_experiences: int = 10000,
                 max_value_estimates: int = 5000,
                 uncertainty_decay: float = 0.95,
                 knowledge_retention: float = 0.9):
        """
        Initialize shared knowledge base
        
        Args:
            max_experiences: Maximum shared experiences to store
            max_value_estimates: Maximum value estimates to store
            uncertainty_decay: Decay rate for uncertainty over time
            knowledge_retention: Retention rate for policy insights
        """
        self.max_experiences = max_experiences
        self.max_value_estimates = max_value_estimates
        self.uncertainty_decay = uncertainty_decay
        self.knowledge_retention = knowledge_retention
        
        # Shared data structures
        self.experiences = deque(maxlen=max_experiences)
        self.value_estimates = deque(maxlen=max_value_estimates)
        self.policy_insights: Dict[str, List[PolicyInsight]] = defaultdict(list)
        self.uncertainty_map: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Communication stats
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.last_update_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def share_experience(self, experience: Experience):
        """Share an experience with other agents"""
        with self._lock:
            self.experiences.append(experience)
            self.message_counts[f"{experience.agent_type}_experience"] += 1
            self._update_uncertainty_map(experience)
    
    def share_value_estimate(self, estimate: ValueEstimate):
        """Share a value estimate for uncertainty calibration"""
        with self._lock:
            self.value_estimates.append(estimate)
            self.message_counts[f"{estimate.agent_type}_value"] += 1
    
    def share_policy_insight(self, insight: PolicyInsight):
        """Share policy insights for knowledge transfer"""
        with self._lock:
            self.policy_insights[insight.agent_type].append(insight)
            # Limit insight history
            if len(self.policy_insights[insight.agent_type]) > 100:
                self.policy_insights[insight.agent_type] = \
                    self.policy_insights[insight.agent_type][-100:]
            self.message_counts[f"{insight.agent_type}_policy"] += 1
    
    def get_relevant_experiences(self, 
                               agent_type: str, 
                               state: np.ndarray, 
                               similarity_threshold: float = 0.8,
                               max_count: int = 10) -> List[Experience]:
        """
        Get relevant experiences from other agents for this state
        
        Args:
            agent_type: Requesting agent type
            state: Current state
            similarity_threshold: Minimum similarity for relevance
            max_count: Maximum experiences to return
            
        Returns:
            List of relevant experiences from other agents
        """
        with self._lock:
            relevant = []
            
            for exp in self.experiences:
                # Skip own experiences
                if exp.agent_type == agent_type:
                    continue
                
                # Calculate state similarity
                similarity = self._calculate_state_similarity(state, exp.state)
                
                if similarity >= similarity_threshold:
                    relevant.append((exp, similarity))
            
            # Sort by similarity and confidence
            relevant.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
            
            return [exp for exp, _ in relevant[:max_count]]
    
    def get_consensus_value(self, state: np.ndarray, agent_type: str) -> Tuple[float, float]:
        """
        Get consensus value estimate from all agents
        
        Args:
            state: State to evaluate
            agent_type: Requesting agent type
            
        Returns:
            Tuple of (consensus_value, uncertainty)
        """
        with self._lock:
            values = []
            weights = []
            
            for estimate in self.value_estimates:
                similarity = self._calculate_state_similarity(state, estimate.state)
                if similarity > 0.7:
                    # Weight by similarity and inverse uncertainty
                    weight = similarity / (1.0 + estimate.uncertainty)
                    values.append(estimate.value)
                    weights.append(weight)
            
            if not values:
                return 0.0, 1.0  # No information, high uncertainty
            
            weights = np.array(weights)
            values = np.array(values)
            
            # Weighted consensus
            consensus_value = np.average(values, weights=weights)
            
            # Uncertainty as weighted variance
            uncertainty = np.average((values - consensus_value) ** 2, weights=weights)
            uncertainty = np.sqrt(uncertainty)
            
            return float(consensus_value), float(uncertainty)
    
    def get_policy_guidance(self, 
                          state: np.ndarray, 
                          agent_type: str,
                          exclude_self: bool = True) -> Optional[np.ndarray]:
        """
        Get policy guidance from other agents
        
        Args:
            state: Current state
            agent_type: Requesting agent type
            exclude_self: Whether to exclude own insights
            
        Returns:
            Action preferences from other agents (or None)
        """
        with self._lock:
            relevant_insights = []
            
            for source_agent, insights in self.policy_insights.items():
                if exclude_self and source_agent == agent_type:
                    continue
                
                for insight in insights:
                    similarity = self._calculate_state_similarity(state, insight.state_features)
                    if similarity > 0.7:
                        # Weight by similarity and success rate
                        weight = similarity * insight.success_rate * insight.confidence
                        relevant_insights.append((insight.action_preferences, weight))
            
            if not relevant_insights:
                return None
            
            # Weighted average of action preferences
            preferences = np.zeros_like(relevant_insights[0][0])
            total_weight = 0.0
            
            for action_pref, weight in relevant_insights:
                preferences += action_pref * weight
                total_weight += weight
            
            if total_weight > 0:
                preferences /= total_weight
                return preferences
            
            return None
    
    def update_performance(self, agent_type: str, performance: float):
        """Update performance history for an agent"""
        with self._lock:
            self.performance_history[agent_type].append(performance)
            # Limit history
            if len(self.performance_history[agent_type]) > 1000:
                self.performance_history[agent_type] = \
                    self.performance_history[agent_type][-1000:]
    
    def get_relative_performance(self, agent_type: str) -> Dict[str, float]:
        """Get performance comparison across agents"""
        with self._lock:
            if agent_type not in self.performance_history:
                return {}
            
            my_performance = np.mean(self.performance_history[agent_type][-50:])
            comparisons = {}
            
            for other_agent, history in self.performance_history.items():
                if other_agent != agent_type and len(history) >= 10:
                    other_performance = np.mean(history[-50:])
                    comparisons[other_agent] = float(other_performance - my_performance)
            
            return comparisons
    
    def _calculate_state_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate similarity between two states"""
        if len(state1) != len(state2):
            return 0.0
        
        # Normalize states
        state1_norm = state1 / (np.linalg.norm(state1) + 1e-8)
        state2_norm = state2 / (np.linalg.norm(state2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(state1_norm, state2_norm)
        return max(0.0, float(similarity))
    
    def _update_uncertainty_map(self, experience: Experience):
        """Update uncertainty map based on new experience"""
        state_key = self._state_to_key(experience.state)
        
        if state_key in self.uncertainty_map:
            # Reduce uncertainty for visited states
            self.uncertainty_map[state_key] *= self.uncertainty_decay
        else:
            # New state starts with high uncertainty
            self.uncertainty_map[state_key] = 1.0
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to string key for uncertainty map"""
        # Discretize continuous state
        discretized = np.round(state * 10) / 10
        return str(discretized.tolist())
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        with self._lock:
            return {
                'total_experiences': len(self.experiences),
                'total_value_estimates': len(self.value_estimates),
                'policy_insights_count': {k: len(v) for k, v in self.policy_insights.items()},
                'message_counts': dict(self.message_counts),
                'uncertainty_states': len(self.uncertainty_map),
                'active_agents': list(self.performance_history.keys())
            }
    
    def export_knowledge(self, filepath: str):
        """Export knowledge base to file"""
        with self._lock:
            knowledge_data = {
                'experiences': [asdict(exp) for exp in self.experiences],
                'value_estimates': [asdict(est) for est in self.value_estimates],
                'policy_insights': {k: [asdict(insight) for insight in v] 
                                  for k, v in self.policy_insights.items()},
                'uncertainty_map': self.uncertainty_map,
                'performance_history': dict(self.performance_history),
                'message_counts': dict(self.message_counts)
            }
            
            with open(filepath, 'w') as f:
                json.dump(knowledge_data, f, indent=2, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class AgentCommunicationInterface:
    """
    Interface for agents to communicate with the shared knowledge base
    """
    
    def __init__(self, agent_type: str, knowledge_base: SharedKnowledgeBase):
        """
        Initialize communication interface
        
        Args:
            agent_type: Type/name of the agent
            knowledge_base: Shared knowledge base instance
        """
        self.agent_type = agent_type
        self.kb = knowledge_base
        self.local_insights_cache = []
        self.last_sync_time = time.time()
    
    def share_transition(self, 
                        state: np.ndarray, 
                        action: Union[int, np.ndarray], 
                        reward: float,
                        next_state: np.ndarray, 
                        done: bool,
                        confidence: float = 1.0):
        """Share a transition experience"""
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            agent_type=self.agent_type,
            timestamp=time.time(),
            confidence=confidence
        )
        self.kb.share_experience(experience)
    
    def share_value_prediction(self, state: np.ndarray, value: float, uncertainty: float = 0.1):
        """Share a value prediction"""
        estimate = ValueEstimate(
            state=state.copy(),
            value=value,
            uncertainty=uncertainty,
            agent_type=self.agent_type,
            timestamp=time.time()
        )
        self.kb.share_value_estimate(estimate)
    
    def share_policy_knowledge(self, 
                             state: np.ndarray, 
                             action_probs: np.ndarray,
                             confidence: float,
                             success_rate: float):
        """Share policy knowledge"""
        insight = PolicyInsight(
            state_features=state.copy(),
            action_preferences=action_probs.copy(),
            confidence=confidence,
            success_rate=success_rate,
            agent_type=self.agent_type
        )
        self.kb.share_policy_insight(insight)
    
    def get_shared_experiences(self, state: np.ndarray, max_count: int = 5) -> List[Experience]:
        """Get relevant shared experiences"""
        return self.kb.get_relevant_experiences(self.agent_type, state, max_count=max_count)
    
    def get_value_consensus(self, state: np.ndarray) -> Tuple[float, float]:
        """Get consensus value estimate"""
        return self.kb.get_consensus_value(state, self.agent_type)
    
    def get_policy_advice(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Get policy advice from other agents"""
        return self.kb.get_policy_guidance(state, self.agent_type)
    
    def update_my_performance(self, performance: float):
        """Update own performance metric"""
        self.kb.update_performance(self.agent_type, performance)
    
    def get_performance_comparison(self) -> Dict[str, float]:
        """Get performance comparison with other agents"""
        return self.kb.get_relative_performance(self.agent_type)
    
    def sync_with_knowledge_base(self):
        """Periodic synchronization with knowledge base"""
        self.last_sync_time = time.time()
        # Could implement additional sync logic here
        pass
