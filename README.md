# RL-Dewey-Tutor: Adaptive Tutorial Agent with Reinforcement Learning

## Overview

RL-Dewey-Tutor is an intelligent tutorial system that uses reinforcement learning to adaptively optimize question difficulty for students. The system learns from student interactions to provide personalized learning experiences that maximize skill development and engagement.

## 🎯 Key Features

- **Dual RL Approaches**: Implements both PPO (Proximal Policy Optimization) and Q-Learning with function approximation
- **Thompson Sampling Exploration**: Advanced exploration strategy for adaptive difficulty selection
- **Rich Environment Model**: Multi-topic learning with realistic student progression, forgetting curves, and skill decay
- **Comprehensive Experimentation**: Automated experiment suite with statistical analysis and reproducibility
- **Professional-Grade Logging**: Detailed training curves, performance metrics, and evaluation reports

## 🏗️ Architecture

### Core Components

```
rl-dewey-tutor/
├── src/
│   ├── envs/
│   │   └── tutor_env.py          # Enhanced tutor environment
│   ├── rl_dewey_tutor/
│   │   └── agents/
│   │       ├── q_learning_agent.py    # Q-Learning with neural networks
│   │       └── thompson_sampling.py   # Thompson Sampling explorer
│   ├── train.py                  # Comprehensive training script
│   ├── evaluate.py               # Model evaluation and comparison
│   ├── run_experiments.py        # Automated experiment suite
│   └── main.py                   # Entry point
├── results/                      # Experiment results and models
├── experiments/                  # Experiment configurations and reports
└── requirements.txt              # Dependencies
```

### Environment Design

The `TutorEnv` implements a sophisticated student-tutor interaction model:

- **State Space**: Multi-dimensional representation including skill levels, learning rates, topic mastery, and confidence
- **Action Space**: Difficulty selection for multiple topics (MultiDiscrete)
- **Reward Function**: Multi-objective optimization balancing performance, difficulty appropriateness, and learning progression
- **Student Model**: Realistic progression with forgetting curves, adaptive learning rates, and topic transfer

### RL Methods

#### 1. PPO (Proximal Policy Optimization)
- **Implementation**: Stable-Baselines3 integration
- **Features**: Policy gradient optimization with clipping, GAE advantage estimation
- **Use Case**: Stable, sample-efficient learning for continuous control

#### 2. Q-Learning with Function Approximation
- **Implementation**: Custom neural network architecture with experience replay
- **Features**: Target networks, epsilon-greedy exploration, gradient clipping
- **Use Case**: Value-based learning with continuous state spaces

#### 3. Thompson Sampling Exploration
- **Implementation**: Bayesian uncertainty quantification
- **Features**: Adaptive exploration based on state uncertainty, Beta distribution priors
- **Integration**: Works with both RL methods for enhanced exploration

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rl-dewey-tutor
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Single Training Run
```bash
# Train both PPO and Q-Learning
python src/train.py --method both --experiment my_experiment

# Train only PPO
python src/train.py --method ppo --experiment ppo_only

# Train only Q-Learning
python src/train.py --method qlearning --experiment ql_only
```

#### Evaluation
```bash
# Evaluate trained models
python src/evaluate.py --experiment results/my_experiment --episodes 100
```

#### Automated Experiments
```bash
# Run baseline configuration with multiple seeds
python src/run_experiments.py --configs baseline --seeds 42 123 456

# Run multiple configurations
python src/run_experiments.py --configs baseline high_complexity --seeds 42 123
```

## 📊 Experiment Configurations

### Baseline Configuration
- **Topics**: 3
- **Difficulty Levels**: 5
- **Max Steps**: 50
- **Training**: 100K timesteps, 500 episodes

### High Complexity Configuration
- **Topics**: 5
- **Difficulty Levels**: 7
- **Max Steps**: 75
- **Training**: 150K timesteps, 750 episodes

### Fast Learning Configuration
- **Topics**: 2
- **Difficulty Levels**: 4
- **Max Steps**: 30
- **Training**: 75K timesteps, 300 episodes

## 🔬 Research Features

### Reproducibility
- **Seed Management**: Global seed setting for all random processes
- **Configuration Files**: JSON-based experiment configurations
- **Version Control**: Comprehensive logging of hyperparameters and results

### Statistical Analysis
- **Multi-Seed Experiments**: Statistical significance testing across seeds
- **Cross-Configuration Comparison**: Performance analysis across different setups
- **Comprehensive Metrics**: Reward distributions, skill progression, mastery levels

### Visualization
- **Training Curves**: Real-time reward, loss, and exploration tracking
- **Evaluation Plots**: Performance comparisons, skill vs mastery scatter plots
- **Statistical Reports**: Automated report generation with significance testing

## 📈 Performance Metrics

The system tracks comprehensive metrics including:

- **Learning Performance**: Episode rewards, training loss, convergence speed
- **Student Outcomes**: Final skill levels, topic mastery, confidence progression
- **Adaptive Behavior**: Difficulty selection patterns, exploration vs exploitation
- **Statistical Significance**: T-tests, confidence intervals, effect sizes

## 🎓 Educational Applications

### Real-World Use Cases
- **Adaptive Testing**: Dynamic difficulty adjustment based on performance
- **Personalized Learning**: Individualized curriculum pacing
- **Skill Assessment**: Continuous evaluation of learning progress
- **Engagement Optimization**: Balancing challenge and success for motivation

### Pedagogical Principles
- **Zone of Proximal Development**: Questions at optimal difficulty level
- **Spaced Repetition**: Strategic review timing for retention
- **Adaptive Scaffolding**: Support that adjusts to student needs
- **Growth Mindset**: Encouraging persistence through appropriate challenges

## 🔧 Advanced Configuration

### Custom Environment Parameters
```json
{
  "env_config": {
    "n_topics": 4,
    "n_difficulty_levels": 6,
    "max_steps": 60,
    "skill_decay_rate": 0.025,
    "learning_rate_variance": 0.12
  }
}
```

### Training Hyperparameters
```json
{
  "training_config": {
    "gamma": 0.99,
    "learning_rate": 0.001,
    "total_timesteps": 200000,
    "total_episodes": 1000,
    "exploration_strength": 1.1
  }
}
```

## 📝 Technical Report

### Mathematical Formulation

#### PPO Objective
```
L(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```

#### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

#### Thompson Sampling
```
P(a|s) = ∫ P(a|θ,s) P(θ|D) dθ
```

### Experimental Design
- **Control Variables**: Environment complexity, training duration, exploration strategy
- **Dependent Variables**: Final performance, learning speed, adaptation quality
- **Statistical Tests**: Paired t-tests, ANOVA for multi-group comparisons

## 🚀 Future Enhancements

### Planned Features
- **Multi-Student Environments**: Classroom-scale optimization
- **Curriculum Planning**: Long-term learning path optimization
- **Real-Time Adaptation**: Online learning with continuous updates
- **Multi-Modal Input**: Text, audio, and visual question types

### Research Directions
- **Meta-Learning**: Learning to learn across different subjects
- **Transfer Learning**: Knowledge transfer between related topics
- **Human-in-the-Loop**: Incorporating teacher feedback and expertise
- **Explainable AI**: Interpretable decision-making for educational transparency

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Standards
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Full type annotation for maintainability
- **Testing**: Unit tests for all components
- **Formatting**: Black code formatting, flake8 linting

## 📚 References

### Key Papers
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
- Thompson, W. R. (1933). "On the likelihood that one unknown probability exceeds another"

### Educational Theory
- Vygotsky, L. S. (1978). "Mind in Society: Development of Higher Psychological Processes"
- Dweck, C. S. (2006). "Mindset: The New Psychology of Success"
- Hattie, J. (2008). "Visible Learning: A Synthesis of Over 800 Meta-Analyses"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Northeastern University CS 5800: Reinforcement Learning for Agentic AI Systems
- Stable-Baselines3 team for the excellent RL framework
- Gymnasium team for the environment interface
- PyTorch team for the deep learning framework

---

**Note**: This is a research implementation for educational purposes. For production use in educational settings, additional safety measures, validation, and human oversight are recommended.
