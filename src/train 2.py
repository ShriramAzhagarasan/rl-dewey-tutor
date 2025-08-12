"""
Enhanced Training Script for RL-Dewey-Tutor

Supports multiple RL methods (PPO, Q-Learning) with Thompson Sampling exploration
and comprehensive experiment logging for reproducible research.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import our custom implementations
from src.envs.tutor_env import TutorEnv
from src.rl_dewey_tutor.agents.q_learning_agent import QLearningAgent
from src.rl_dewey_tutor.agents.thompson_sampling import ThompsonSamplingExplorer

class ExperimentLogger:
    """Comprehensive experiment logging and visualization"""
    
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        
        # Create directories
        for dir_path in [self.experiment_dir, self.logs_dir, self.models_dir, self.figures_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize logging
        self.training_log = []
        self.evaluation_log = []
        self.experiment_config = {}
        
        # Create experiment metadata
        self.experiment_config['timestamp'] = datetime.now().isoformat()
        self.experiment_config['experiment_name'] = experiment_name
    
    def log_training_step(self, step: int, reward: float, loss: Optional[float] = None, 
                         exploration_prob: Optional[float] = None, **kwargs):
        """Log training step information"""
        log_entry = {
            'step': step,
            'reward': reward,
            'timestamp': time.time()
        }
        
        if loss is not None:
            log_entry['loss'] = loss
        if exploration_prob is not None:
            log_entry['exploration_probability'] = exploration_prob
        
        log_entry.update(kwargs)
        self.training_log.append(log_entry)
    
    def log_evaluation(self, episode: int, total_reward: float, final_skill: float, 
                      final_mastery: float, **kwargs):
        """Log evaluation episode results"""
        log_entry = {
            'episode': episode,
            'total_reward': total_reward,
            'final_skill': final_skill,
            'final_mastery': final_mastery,
            'timestamp': time.time()
        }
        log_entry.update(kwargs)
        self.evaluation_log.append(log_entry)
    
    def save_experiment_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        self.experiment_config.update(config)
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.experiment_config, f, indent=2)
    
    def save_training_log(self):
        """Save training log to CSV"""
        if self.training_log:
            df = pd.DataFrame(self.training_log)
            log_path = os.path.join(self.logs_dir, "training_log.csv")
            df.to_csv(log_path, index=False)
    
    def save_evaluation_log(self):
        """Save evaluation log to CSV"""
        if self.evaluation_log:
            df = pd.DataFrame(self.evaluation_log)
            log_path = os.path.join(self.logs_dir, "evaluation_log.csv")
            df.to_csv(log_path, index=False)
    
    def plot_training_curves(self):
        """Generate and save training visualization plots"""
        if not self.training_log:
            return
        
        df = pd.DataFrame(self.training_log)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {self.experiment_name}', fontsize=16)
        
        # Reward over time
        if 'reward' in df.columns:
            axes[0, 0].plot(df['step'], df['reward'], alpha=0.6)
            axes[0, 0].set_title('Reward per Step')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Reward')
            
            # Rolling average
            if len(df) > 50:
                rolling_avg = df['reward'].rolling(50).mean()
                axes[0, 0].plot(df['step'], rolling_avg, 'r-', linewidth=2, label='50-step MA')
                axes[0, 0].legend()
        
        # Loss over time
        if 'loss' in df.columns:
            axes[0, 1].plot(df['step'], df['loss'], alpha=0.6)
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
        
        # Exploration probability
        if 'exploration_probability' in df.columns:
            axes[1, 0].plot(df['step'], df['exploration_probability'], alpha=0.6)
            axes[1, 0].set_title('Exploration Probability')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Exploration Probability')
        
        # Evaluation results
        if self.evaluation_log:
            eval_df = pd.DataFrame(self.evaluation_log)
            axes[1, 1].plot(eval_df['episode'], eval_df['total_reward'], 'g-', alpha=0.7)
            axes[1, 1].set_title('Evaluation Performance')
            axes[1, 1].set_xlabel('Evaluation Episode')
            axes[1, 1].set_ylabel('Total Reward')
        
        plt.tight_layout()
        plot_path = os.path.join(self.figures_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, logger: ExperimentLogger, verbose: int = 0):
        super().__init__(verbose)
        self.logger = logger
        self.step_count = 0
    
    def _on_step(self) -> bool:
        """Called after each step"""
        if self.training_env is not None:
            # Extract information from the environment
            infos = self.training_env.get_attr('get_student_progress')[0]
            if infos:
                reward = infos.get('reward_history', [0])[-1] if infos.get('reward_history') else 0
                self.logger.log_training_step(
                    step=self.step_count,
                    reward=reward
                )
        
        self.step_count += 1
        return True

def make_env(env_config: Dict[str, Any] = None):
    """Create and wrap environment"""
    if env_config is None:
        env_config = {}
    
    env = TutorEnv(**env_config)
    return Monitor(env, filename=None)  # We'll handle logging ourselves

def train_ppo(env_config: Dict[str, Any], training_config: Dict[str, Any], 
              logger: ExperimentLogger) -> PPO:
    """Train PPO agent"""
    print("Training PPO agent...")
    
    # Create environment
    env = DummyVecEnv([lambda: make_env(env_config)])
    
    # PPO configuration
    ppo_config = (
        "MlpPolicy", env, verbose=1,
        gamma=training_config.get('gamma', 0.99),
        gae_lambda=training_config.get('gae_lambda', 0.95),
        n_steps=training_config.get('n_steps', 2048),
        batch_size=training_config.get('batch_size', 256),
        learning_rate=training_config.get('learning_rate', 3e-4),
        ent_coef=training_config.get('ent_coef', 0.01),
        clip_range=training_config.get('clip_range', 0.2),
        tensorboard_log=logger.logs_dir
    )
    
    model = PPO(*ppo_config)
    
    # Training callback
    callback = TrainingCallback(logger)
    
    # Train
    model.learn(
        total_timesteps=training_config.get('total_timesteps', 200_000),
        callback=callback
    )
    
    # Save model
    model_path = os.path.join(logger.models_dir, "ppo_tutor")
    model.save(model_path)
    print(f"PPO model saved to: {model_path}")
    
    return model

def train_q_learning(env_config: Dict[str, Any], training_config: Dict[str, Any], 
                     logger: ExperimentLogger) -> QLearningAgent:
    """Train Q-Learning agent"""
    print("Training Q-Learning agent...")
    
    # Create environment
    env = make_env(env_config)
    
    # Initialize Q-Learning agent
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    
    agent = QLearningAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        learning_rate=training_config.get('learning_rate', 0.001),
        gamma=training_config.get('gamma', 0.99),
        epsilon=training_config.get('epsilon', 1.0),
        epsilon_decay=training_config.get('epsilon_decay', 0.995),
        epsilon_min=training_config.get('epsilon_min', 0.01),
        batch_size=training_config.get('batch_size', 64),
        buffer_size=training_config.get('buffer_size', 10000)
    )
    
    # Initialize Thompson Sampling explorer
    explorer = ThompsonSamplingExplorer(
        state_dim=state_dim,
        action_dims=action_dims,
        exploration_strength=training_config.get('exploration_strength', 1.0)
    )
    
    # Training loop
    total_episodes = training_config.get('total_episodes', 1000)
    eval_freq = training_config.get('eval_freq', 50)
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(env.max_steps):
            # Get exploration probability
            exploration_prob = explorer.get_exploration_probability(obs)
            
            # Select action
            if np.random.random() < exploration_prob:
                action = explorer.sample_action_thompson(obs)
            else:
                action = agent.select_action(obs, training=True)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            explorer.update_uncertainty(obs, action, reward, next_obs)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            # Log training step
            logger.log_training_step(
                step=episode * env.max_steps + step,
                reward=reward,
                loss=loss,
                exploration_probability=exploration_prob
            )
            
            episode_reward += reward
            obs = next_obs
            
            if done or truncated:
                break
        
        # Log episode
        final_skill = np.mean(env.skill_levels) if hasattr(env, 'skill_levels') else 0.5
        final_mastery = np.mean(env.topic_mastery) if hasattr(env, 'topic_mastery') else 0.5
        
        logger.log_evaluation(
            episode=episode,
            total_reward=episode_reward,
            final_skill=final_skill,
            final_mastery=final_mastery
        )
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"Episode {episode + 1}/{total_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    model_path = os.path.join(logger.models_dir, "qlearning_tutor.pth")
    agent.save_model(model_path)
    print(f"Q-Learning model saved to: {model_path}")
    
    return agent

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL-Dewey-Tutor Training")
    parser.add_argument("--method", choices=["ppo", "qlearning", "both"], 
                       default="both", help="RL method to use")
    parser.add_argument("--experiment", type=str, default=None, 
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Generate experiment name if not provided
    if args.experiment is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment = f"tutor_experiment_{timestamp}"
    
    # Initialize logger
    logger = ExperimentLogger(args.experiment)
    
    # Default configurations
    env_config = {
        'n_topics': 3,
        'n_difficulty_levels': 5,
        'max_steps': 50,
        'skill_decay_rate': 0.02,
        'learning_rate_variance': 0.1
    }
    
    training_config = {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'total_timesteps': 200_000,
        'total_episodes': 1000,
        'eval_freq': 50,
        'exploration_strength': 1.0
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            env_config.update(custom_config.get('env', {}))
            training_config.update(custom_config.get('training', {}))
    
    # Save experiment configuration
    logger.save_experiment_config({
        'method': args.method,
        'seed': args.seed,
        'env_config': env_config,
        'training_config': training_config
    })
    
    # Training
    if args.method in ["ppo", "both"]:
        ppo_model = train_ppo(env_config, training_config, logger)
    
    if args.method in ["qlearning", "both"]:
        qlearning_model = train_q_learning(env_config, training_config, logger)
    
    # Save logs and generate plots
    logger.save_training_log()
    logger.save_evaluation_log()
    logger.plot_training_curves()
    
    print(f"\nTraining completed! Results saved to: {logger.experiment_dir}")
    print("Generated files:")
    print(f"  - Training log: {logger.logs_dir}/training_log.csv")
    print(f"  - Evaluation log: {logger.logs_dir}/evaluation_log.csv")
    print(f"  - Training curves: {logger.figures_dir}/training_curves.png")
    print(f"  - Models: {logger.models_dir}/")

if __name__ == "__main__":
    main()

