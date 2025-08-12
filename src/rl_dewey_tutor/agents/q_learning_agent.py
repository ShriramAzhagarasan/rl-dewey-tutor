"""
Q-Learning Agent with Function Approximation for RL-Dewey-Tutor

This module implements a Q-Learning agent using neural network function approximation
to handle continuous state spaces in the tutor environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import random
from collections import deque
import os

class QNetwork(nn.Module):
    """Neural network for Q-value function approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for Q-Learning"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    """
    Q-Learning Agent with Neural Network Function Approximation
    
    Features:
    - Experience replay for stable learning
    - Target network for stable Q-value updates
    - Epsilon-greedy exploration strategy
    - Adaptive learning rate
    - Gradient clipping for stability
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dims: List[int],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 target_update_freq: int = 100,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 hidden_dims: List[int] = [128, 64]):
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.total_actions = int(np.prod(action_dims))
        
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Network parameters
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.update_count = 0
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, self.total_actions, hidden_dims)
        self.target_network = QNetwork(state_dim, self.total_actions, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def _action_to_index(self, action: np.ndarray) -> int:
        """Convert multi-dimensional action to single index"""
        index = 0
        for i, a in enumerate(action):
            index += int(a) * int(np.prod(self.action_dims[i+1:])) if i < len(self.action_dims) - 1 else int(a)
        return int(index)
    
    def _index_to_action(self, index: int) -> np.ndarray:
        """Convert single index to multi-dimensional action"""
        action = []
        remaining = int(index)
        
        for i, dim in enumerate(self.action_dims):
            if i < len(self.action_dims) - 1:
                divisor = int(np.prod(self.action_dims[i+1:]))
                action.append(remaining // divisor)
                remaining = remaining % divisor
            else:
                action.append(remaining)
        
        return np.array(action)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy strategy"""
        if training and random.random() < self.epsilon:
            # Random action
            return np.array([random.randint(0, int(dim) - 1) for dim in self.action_dims])
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_index = int(q_values.argmax().item())
            return self._index_to_action(action_index)
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones.astype(np.float32)).to(self.device)  # 1.0 if done else 0.0
        
        # Convert actions to indices
        action_indices = torch.LongTensor([
            self._action_to_index(action) for action in actions
        ]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1.0 - dones_tensor) * (self.gamma * next_q_values)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record loss
        self.training_losses.append(float(loss.item()))
        
        return float(loss.item())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_losses = checkpoint.get('training_losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
    
    def get_training_stats(self) -> Dict[str, List]:
        """Get training statistics"""
        return {
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'epsilon': [self.epsilon]
        } 