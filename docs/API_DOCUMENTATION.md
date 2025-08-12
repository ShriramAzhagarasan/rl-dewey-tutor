# RL-Dewey-Tutor API Documentation

## Overview

This document provides comprehensive API documentation for all custom tools and components in the RL-Dewey-Tutor system. The system is designed with modular, reusable components that enable sophisticated reinforcement learning for adaptive tutoring.

## Table of Contents

1. [Core Environment](#core-environment)
2. [RL Agents](#rl-agents)
3. [Exploration Strategies](#exploration-strategies)
4. [Communication System](#communication-system)
5. [Adaptive Controller](#adaptive-controller)
6. [Baseline Tutors](#baseline-tutors)
7. [Analysis Tools](#analysis-tools)
8. [Error Handling](#error-handling)
9. [Utilities](#utilities)

---

## Core Environment

### TutorEnv

The main environment class implementing a sophisticated student-tutor interaction model.

#### Class Definition

```python
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
```

#### Constructor

```python
def __init__(self, 
             n_topics: int = 3,
             n_difficulty_levels: int = 5,
             max_steps: int = 50,
             skill_decay_rate: float = 0.02,
             learning_rate_variance: float = 0.1,
             reward_shaping: bool = True,
             reward_shaping_scale: float = 1.0)
```

**Parameters:**
- `n_topics` (int): Number of learning topics (default: 3)
- `n_difficulty_levels` (int): Number of difficulty levels per topic (default: 5)
- `max_steps` (int): Maximum steps per episode (default: 50)
- `skill_decay_rate` (float): Rate of skill decay over time (default: 0.02)
- `learning_rate_variance` (float): Variance in learning rates across topics (default: 0.1)
- `reward_shaping` (bool): Enable adaptive reward shaping (default: True)
- `reward_shaping_scale` (float): Scale factor for reward shaping (default: 1.0)

#### Key Methods

##### step(action)

Execute one step in the environment.

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
```

**Parameters:**
- `action` (np.ndarray): Difficulty levels for each topic

**Returns:**
- `observation` (np.ndarray): New state vector
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode is finished
- `truncated` (bool): Whether episode was truncated
- `info` (dict): Additional information

##### reset()

Reset the environment to initial state.

```python
def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Returns:**
- `observation` (np.ndarray): Initial state vector
- `info` (dict): Additional information

#### State Space

The observation space is a Box with dimensions `n_topics * 3 + 2`:
- **Skill levels** (n_topics): Current skill level for each topic [0, 1]
- **Learning rates** (n_topics): Adaptive learning rate for each topic [0, 1]
- **Topic mastery** (n_topics): Mastery level for each topic [0, 1]
- **Global confidence** (1): Student's overall confidence [0, 1]
- **Last performance** (1): Performance in the last interaction [0, 1]

#### Action Space

MultiDiscrete space with `n_difficulty_levels` choices for each topic.

#### Reward Function

The reward function balances multiple objectives:
- **Base performance**: Mean performance across topics
- **Difficulty appropriateness**: Bonus for well-matched difficulty
- **Consistency bonus**: Reward for consistent performance
- **Mastery progression**: Bonus for advancing mastery
- **Global confidence**: Bonus for maintaining confidence

---

## RL Agents

### QLearningAgent

Deep Q-Learning agent with neural network function approximation.

#### Class Definition

```python
class QLearningAgent:
    """
    Q-Learning agent with neural network function approximation
    
    Features:
    - Deep Q-Network with target network
    - Experience replay buffer
    - Epsilon-greedy exploration
    - Gradient clipping for stability
    """
```

#### Constructor

```python
def __init__(self,
             state_dim: int,
             action_dim: int,
             hidden_dim: int = 128,
             learning_rate: float = 0.001,
             gamma: float = 0.99,
             epsilon: float = 1.0,
             epsilon_min: float = 0.01,
             epsilon_decay: float = 0.995,
             buffer_size: int = 10000,
             batch_size: int = 32,
             target_update_freq: int = 100)
```

**Parameters:**
- `state_dim` (int): Dimension of state space
- `action_dim` (int): Dimension of action space
- `hidden_dim` (int): Hidden layer dimension (default: 128)
- `learning_rate` (float): Learning rate for optimizer (default: 0.001)
- `gamma` (float): Discount factor (default: 0.99)
- `epsilon` (float): Initial exploration rate (default: 1.0)
- `epsilon_min` (float): Minimum exploration rate (default: 0.01)
- `epsilon_decay` (float): Exploration decay rate (default: 0.995)
- `buffer_size` (int): Replay buffer size (default: 10000)
- `batch_size` (int): Training batch size (default: 32)
- `target_update_freq` (int): Target network update frequency (default: 100)

#### Key Methods

##### act(state, exploration_prob)

Select action using epsilon-greedy policy.

```python
def act(self, state: np.ndarray, exploration_prob: Optional[float] = None) -> int
```

##### update(state, action, reward, next_state, done)

Update Q-network using experience.

```python
def update(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: bool) -> float
```

##### save_model(filepath) / load_model(filepath)

Save and load model checkpoints.

```python
def save_model(self, filepath: str)
def load_model(self, filepath: str)
```

---

## Exploration Strategies

### ThompsonSamplingExplorer

Bayesian exploration strategy using Thompson Sampling.

#### Class Definition

```python
class ThompsonSamplingExplorer:
    """
    Thompson Sampling exploration strategy for RL agents
    
    Features:
    - Bayesian uncertainty quantification
    - State-dependent exploration
    - Beta distribution priors
    - Adaptive exploration based on visit counts
    """
```

#### Constructor

```python
def __init__(self,
             n_actions: int,
             alpha_prior: float = 1.0,
             beta_prior: float = 1.0,
             uncertainty_threshold: float = 0.1)
```

#### Key Methods

##### get_exploration_probability(state, visit_count)

Calculate exploration probability for given state.

```python
def get_exploration_probability(self, state: np.ndarray, visit_count: int = 0) -> float
```

##### update_uncertainty(state, action, reward)

Update uncertainty estimates based on experience.

```python
def update_uncertainty(self, state: np.ndarray, action: int, reward: float)
```

---

## Communication System

### SharedKnowledgeBase

Centralized knowledge sharing system for multi-agent communication.

#### Class Definition

```python
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
```

#### Key Methods

##### share_experience(experience)

Share experience with other agents.

```python
def share_experience(self, experience: Experience)
```

##### get_consensus_value(state, agent_type)

Get consensus value estimate from all agents.

```python
def get_consensus_value(self, state: np.ndarray, agent_type: str) -> Tuple[float, float]
```

### AgentCommunicationInterface

Interface for agents to communicate with the knowledge base.

#### Constructor

```python
def __init__(self, agent_type: str, knowledge_base: SharedKnowledgeBase)
```

#### Key Methods

##### share_transition(state, action, reward, next_state, done)

Share a transition experience.

```python
def share_transition(self, state: np.ndarray, action: Union[int, np.ndarray], 
                    reward: float, next_state: np.ndarray, done: bool,
                    confidence: float = 1.0)
```

---

## Adaptive Controller

### AdaptiveController

Sophisticated controller for dynamic agent orchestration.

#### Class Definition

```python
class AdaptiveController:
    """
    Sophisticated controller that orchestrates multiple RL agents with:
    - Dynamic agent selection based on performance
    - Fallback strategies for failed training
    - Error recovery mechanisms
    - Performance monitoring and adaptation
    """
```

#### Constructor

```python
def __init__(self, 
             performance_window: int = 100,
             stability_threshold: float = 0.1,
             convergence_threshold: float = 0.05,
             fallback_timeout: float = 300.0)
```

#### Key Methods

##### select_agent(current_performance)

Dynamically select the best agent based on performance.

```python
def select_agent(self, current_performance: Dict[str, float]) -> AgentType
```

##### handle_training_error(agent_type, error)

Handle training errors with fallback strategies.

```python
def handle_training_error(self, agent_type: AgentType, error: Exception) -> Dict[str, Any]
```

---

## Baseline Tutors

### BaselineTutor (Abstract)

Abstract base class for baseline tutoring strategies.

```python
class BaselineTutor(ABC):
    @abstractmethod
    def select_difficulty(self, student_state: Dict[str, Any]) -> np.ndarray
    
    @abstractmethod
    def update_strategy(self, student_state: Dict[str, Any], performance: np.ndarray)
    
    @abstractmethod
    def get_name(self) -> str
```

### Available Baseline Strategies

1. **RandomTutor**: Random difficulty selection
2. **FixedProgressionTutor**: Fixed linear progression
3. **PerformanceBasedTutor**: Simple performance-based adjustment
4. **ZPDTutor**: Zone of Proximal Development inspired
5. **AdaptiveDifficultyTutor**: UCB-based difficulty selection
6. **MasteryBasedTutor**: Mastery-based progression

### BaselineTutorFactory

Factory for creating baseline tutors.

```python
@staticmethod
def create_tutor(strategy: BaselineStrategy, 
                n_topics: int, 
                n_difficulty_levels: int,
                **kwargs) -> BaselineTutor
```

---

## Analysis Tools

### LearningStabilityAnalyzer

Comprehensive analyzer for learning stability and convergence.

#### Constructor

```python
def __init__(self, 
             window_size: int = 100,
             convergence_threshold: float = 0.05,
             outlier_threshold: float = 2.5,
             min_convergence_length: int = 50)
```

#### Key Methods

##### analyze_learning_curve(rewards)

Comprehensive stability analysis of learning curve.

```python
def analyze_learning_curve(self, 
                          rewards: List[float],
                          additional_metrics: Optional[Dict[str, List[float]]] = None) -> StabilityMetrics
```

##### visualize_stability_analysis(rewards, metrics)

Create comprehensive visualization of stability analysis.

```python
def visualize_stability_analysis(self, 
                               rewards: List[float],
                               metrics: StabilityMetrics,
                               save_path: Optional[str] = None) -> plt.Figure
```

### StabilityMetrics

Dataclass containing comprehensive stability metrics:

- **Convergence metrics**: convergence_rate, convergence_point, plateau_stability
- **Variance metrics**: reward_variance, variance_trend, coefficient_of_variation
- **Trend analysis**: learning_trend, trend_consistency, monotonicity_score
- **Robustness metrics**: outlier_frequency, shock_recovery_time, performance_volatility
- **Statistical measures**: autocorrelation, stationarity_p_value, regime_changes
- **Composite scores**: overall_stability_score, learning_efficiency_score, robustness_score

---

## Error Handling

### RobustErrorHandler

Comprehensive error handler with recovery strategies.

#### Constructor

```python
def __init__(self, 
             max_retries: int = 3,
             retry_delay: float = 1.0,
             log_file: Optional[str] = None,
             emergency_save_dir: str = "emergency_saves")
```

#### Key Methods

##### robust_execute(func, *args, **kwargs)

Execute function with robust error handling.

```python
def robust_execute(self, 
                  func: Callable, 
                  *args, 
                  error_context: Optional[Dict[str, Any]] = None,
                  recovery_strategy: Optional[RecoveryStrategy] = None,
                  fallback_func: Optional[Callable] = None,
                  **kwargs) -> Any
```

### Decorator Usage

```python
@robust_execution(max_retries=3, recovery_strategy=RecoveryStrategy.FALLBACK)
def my_training_function():
    # Training code here
    pass
```

---

## Usage Examples

### Basic Training Setup

```python
from envs.tutor_env import TutorEnv
from rl_dewey_tutor.agents.q_learning_agent import QLearningAgent
from rl_dewey_tutor.agents.thompson_sampling import ThompsonSamplingExplorer

# Create environment
env = TutorEnv(n_topics=3, n_difficulty_levels=5, max_steps=50)

# Create agent
agent = QLearningAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.nvec[0],
    learning_rate=0.001
)

# Create exploration strategy
explorer = ThompsonSamplingExplorer(n_actions=env.action_space.nvec[0])

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(env.max_steps):
        # Get exploration probability
        exploration_prob = explorer.get_exploration_probability(state)
        
        # Select action
        action = agent.act(state, exploration_prob)
        
        # Take step
        next_state, reward, done, _, info = env.step([action] * env.n_topics)
        
        # Update agent
        loss = agent.update(state, action, reward, next_state, done)
        
        # Update explorer
        explorer.update_uncertainty(state, action, reward)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
```

### Advanced Multi-Agent Setup

```python
from rl_dewey_tutor.controllers.adaptive_controller import AdaptiveController
from rl_dewey_tutor.communication.knowledge_sharing import SharedKnowledgeBase, AgentCommunicationInterface

# Create shared knowledge base
knowledge_base = SharedKnowledgeBase()

# Create adaptive controller
controller = AdaptiveController()

# Create communication interfaces
ppo_comm = AgentCommunicationInterface("ppo", knowledge_base)
ql_comm = AgentCommunicationInterface("qlearning", knowledge_base)

# Training with communication
for episode in range(1000):
    # Select agent based on performance
    selected_agent = controller.select_agent(current_performance)
    
    # Share experiences
    ppo_comm.share_transition(state, action, reward, next_state, done)
    
    # Get consensus values
    consensus_value, uncertainty = ql_comm.get_value_consensus(state)
    
    # Update performance
    ppo_comm.update_my_performance(episode_reward)
```

### Stability Analysis

```python
from rl_dewey_tutor.analysis.stability_metrics import LearningStabilityAnalyzer

# Create analyzer
analyzer = LearningStabilityAnalyzer()

# Analyze learning curve
rewards = [...]  # Your reward history
metrics = analyzer.analyze_learning_curve(rewards)

# Visualize results
fig = analyzer.visualize_stability_analysis(rewards, metrics, save_path="stability_analysis.png")

# Print key metrics
print(f"Overall Stability: {metrics.overall_stability_score:.3f}")
print(f"Learning Efficiency: {metrics.learning_efficiency_score:.3f}")
print(f"Robustness Score: {metrics.robustness_score:.3f}")
```

### Baseline Comparison

```python
from rl_dewey_tutor.agents.baseline_tutor import BaselineTutorFactory, BaselineStrategy, BaselineEvaluator

# Create evaluator
evaluator = BaselineEvaluator(env_config)

# Evaluate all baselines
results = evaluator.compare_all_baselines(n_episodes=100)

# Compare with RL agent
best_baseline = results['comparison_summary']['best_mean_reward']
print(f"Best baseline: {best_baseline}")
print(f"Best baseline performance: {results[best_baseline]['mean_reward']:.3f}")
```

---

## Configuration Management

### Environment Configuration

```json
{
    "n_topics": 3,
    "n_difficulty_levels": 5,
    "max_steps": 50,
    "skill_decay_rate": 0.02,
    "learning_rate_variance": 0.1,
    "reward_shaping": true,
    "reward_shaping_scale": 1.0
}
```

### Training Configuration

```json
{
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "batch_size": 32,
    "buffer_size": 10000,
    "target_update_freq": 100
}
```

---

## Best Practices

### 1. Environment Setup
- Start with default parameters for initial experiments
- Use reward shaping for faster learning, disable for ablation studies
- Adjust `max_steps` based on your learning objectives

### 2. Agent Configuration
- Use larger hidden dimensions for complex environments
- Tune learning rates based on convergence behavior
- Monitor exploration-exploitation balance

### 3. Stability Analysis
- Always analyze learning stability for research publications
- Use multiple seeds for statistical significance
- Compare against baseline tutors for context

### 4. Error Handling
- Wrap training loops with robust error handling
- Implement fallback strategies for production deployment
- Monitor error patterns for system reliability

### 5. Communication Systems
- Enable knowledge sharing for multi-agent setups
- Monitor communication overhead in large-scale deployments
- Use consensus mechanisms for value function alignment

---

## Troubleshooting

### Common Issues

1. **Slow Convergence**: Increase learning rate, reduce environment complexity
2. **Unstable Learning**: Enable gradient clipping, reduce learning rate
3. **Poor Exploration**: Increase exploration parameters, check Thompson Sampling configuration
4. **Memory Issues**: Reduce batch size, buffer size, or model complexity
5. **Communication Bottlenecks**: Limit knowledge sharing frequency, optimize data structures

### Performance Optimization

1. **GPU Usage**: Ensure PyTorch tensors are on GPU for neural networks
2. **Vectorization**: Use vectorized environments for parallel training
3. **Batch Processing**: Optimize batch sizes for your hardware
4. **Memory Management**: Monitor memory usage in long training runs

---

## Contributing

When extending the system:

1. Follow the existing API patterns
2. Add comprehensive docstrings
3. Include type hints for all function parameters
4. Write unit tests for new components
5. Update this documentation for new features

For questions or contributions, refer to the main repository documentation.
