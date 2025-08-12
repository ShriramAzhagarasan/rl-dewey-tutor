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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.callbacks import EvalCallback

# Import our custom implementations
from envs.tutor_env import TutorEnv
from rl_dewey_tutor.agents.q_learning_agent import QLearningAgent
from rl_dewey_tutor.agents.thompson_sampling import ThompsonSamplingExplorer

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
                         exploration_prob: Optional[float] = None, method: Optional[str] = None,
                         metric: Optional[str] = None, **kwargs):
        """Log training step information"""
        log_entry = {
            'step': int(step),
            'reward': float(reward),
            'timestamp': time.time()
        }
        
        if loss is not None:
            log_entry['loss'] = float(loss)
        if exploration_prob is not None:
            log_entry['exploration_probability'] = float(exploration_prob)
        if method is not None:
            log_entry['method'] = method
        if metric is not None:
            log_entry['metric'] = metric
        
        log_entry.update(kwargs)
        self.training_log.append(log_entry)
    
    def log_evaluation(self, episode: int, total_reward: float, final_skill: float, 
                      final_mastery: float, method: Optional[str] = None, **kwargs):
        """Log evaluation episode results"""
        log_entry = {
            'episode': int(episode),
            'total_reward': float(total_reward),
            'final_skill': float(final_skill),
            'final_mastery': float(final_mastery),
            'timestamp': time.time()
        }
        if method is not None:
            log_entry['method'] = method
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
            # PPO eval rewards
            if 'method' in df.columns and 'metric' in df.columns:
                dppo = df[(df['method'] == 'ppo') & (df['metric'] == 'eval_reward')]
                if not dppo.empty:
                    axes[0, 0].plot(dppo['step'], dppo['reward'], label='PPO eval reward', color='tab:blue')
                # QL step rewards (rolling)
                dql = df[(df['method'] == 'qlearning') & (df['metric'] == 'step_reward')]
                if not dql.empty:
                    dql_sorted = dql.sort_values('step')
                    ma = dql_sorted['reward'].rolling(200).mean()
                    axes[0, 0].plot(dql_sorted['step'], ma, label='QL reward (200-step MA)', color='tab:orange')
            else:
                axes[0, 0].plot(df['step'], df['reward'], alpha=0.3, label='reward')
                if len(df) > 50:
                    axes[0, 0].plot(df['step'], df['reward'].rolling(50).mean(), 'r-', linewidth=2, label='50-step MA')
            axes[0, 0].set_title('Reward per Step (separated)')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
        
        # Loss over time
        if 'loss' in df.columns:
            if 'method' in df.columns:
                dql_loss = df[(df['method'] == 'qlearning') & df['loss'].notna()].sort_values('step')
                if not dql_loss.empty:
                    axes[0, 1].plot(dql_loss['step'], dql_loss['loss'], alpha=0.3, label='QL loss')
            else:
                axes[0, 1].plot(df['step'], df['loss'], alpha=0.3)
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_title('Training Loss (log scale)')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
        
        # Exploration probability
        if 'exploration_probability' in df.columns:
            if 'method' in df.columns:
                dql_exp = df[(df['method'] == 'qlearning') & df['exploration_probability'].notna()].sort_values('step')
                if not dql_exp.empty:
                    axes[1, 0].plot(dql_exp['step'], dql_exp['exploration_probability'], alpha=0.6)
            else:
                axes[1, 0].plot(df['step'], df['exploration_probability'], alpha=0.6)
            axes[1, 0].set_title('Exploration Probability (QL)')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Exploration Probability')
        
        # Evaluation results (if populated)
        if self.evaluation_log:
            eval_df = pd.DataFrame(self.evaluation_log)
            if 'method' in eval_df.columns:
                ppo_eval = eval_df[eval_df['method'] == 'ppo']
                ql_eval = eval_df[eval_df['method'] == 'qlearning']
                if not ppo_eval.empty:
                    axes[1, 1].plot(ppo_eval['episode'], ppo_eval['total_reward'], 'g-', alpha=0.7, label='PPO eval')
                if not ql_eval.empty:
                    axes[1, 1].plot(ql_eval['episode'], ql_eval['total_reward'], 'b-', alpha=0.5, label='QL episode')
                axes[1, 1].legend()
            else:
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
    
    def __init__(self, logger: 'ExperimentLogger', verbose: int = 0):
        super().__init__(verbose)
        self._logger = logger
        self.step_count = 0
    
    @property
    def logger(self):
        return self._logger
    
    def _on_step(self) -> bool:
        # Keep simple; PPO logging handled by EvalCallback below
        self.step_count += 1
        return True

class LoggingEvalCallback(EvalCallback):
    """EvalCallback that also mirrors eval results into our ExperimentLogger."""
    def __init__(self, logger: 'ExperimentLogger', eval_env, n_eval_episodes=5, eval_freq=5000,
                 deterministic=True, render=False):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                         deterministic=deterministic, render=render)
        self._logger = logger
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        # If an evaluation just happened (every eval_freq calls), record mean reward
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            mean_r = float(self.last_mean_reward) if hasattr(self, 'last_mean_reward') else None
            if mean_r is not None:
                self._logger.log_training_step(step=self.model.num_timesteps, reward=mean_r, method='ppo', metric='eval_reward')
                # also mirror into evaluation log for the eval pane
                self._logger.log_evaluation(episode=self.model.num_timesteps, total_reward=mean_r, final_skill=0.0, final_mastery=0.0, method='ppo')
        return result

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
    
    # Create train and eval environments
    env = DummyVecEnv([lambda: make_env(env_config)])
    eval_env = DummyVecEnv([lambda: make_env(env_config)])
    
    # PPO configuration
    model = PPO(
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
    
    # Callbacks: periodic evaluation that logs to ExperimentLogger
    eval_callback = LoggingEvalCallback(
        logger=logger,
        eval_env=eval_env,
        n_eval_episodes=training_config.get('eval_episodes', 5),
        eval_freq=training_config.get('eval_freq_timesteps', 5000),
        deterministic=True,
        render=False,
    )
    callback_list = CallbackList([TrainingCallback(logger), eval_callback])
    
    # Train
    model.learn(
        total_timesteps=training_config.get('total_timesteps', 200_000),
        callback=callback_list
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
        epsilon_decay=training_config.get('epsilon_decay', 0.997),
        epsilon_min=training_config.get('epsilon_min', 0.05),
        target_update_freq=training_config.get('target_update_freq', 500),
        batch_size=training_config.get('batch_size', 128),
        buffer_size=training_config.get('buffer_size', 50000)
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
    
    # Get max_steps from the underlying environment
    max_steps = env.unwrapped.max_steps if hasattr(env, 'unwrapped') else 50
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
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
                step=episode * max_steps + step,
                reward=reward,
                loss=loss,
                exploration_prob=exploration_prob,
                method='qlearning',
                metric='step_reward'
            )
            
            # Also log loss separately when available
            if loss is not None:
                logger.log_training_step(
                    step=episode * max_steps + step,
                    reward=0,  # Use 0 for loss entries
                    loss=loss,
                    exploration_prob=exploration_prob,
                    method='qlearning',
                    metric='loss'
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
            final_mastery=final_mastery,
            method='qlearning'
        )
        
        # Also log evaluation data at regular intervals for better tracking
        if episode % eval_freq == 0:
            # Run a few evaluation episodes
            eval_rewards = []
            for _ in range(3):  # Quick evaluation
                eval_obs, _ = env.reset()
                eval_reward = 0
                for _ in range(max_steps):
                    eval_action = agent.select_action(eval_obs, training=False)
                    eval_obs, eval_r, eval_done, eval_truncated, _ = env.step(eval_action)
                    eval_reward += eval_r
                    if eval_done or eval_truncated:
                        break
                eval_rewards.append(eval_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            logger.log_training_step(
                step=episode * max_steps,
                reward=avg_eval_reward,
                loss=None,
                exploration_prob=explorer.get_exploration_probability(obs),
                method='qlearning',
                metric='eval_reward'
            )
        
        # Print progress
        if (episode + 1) % 50 == 0:
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
        'eval_freq_timesteps': 5000,
        'eval_episodes': 5,
        'exploration_strength': 1.0
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            env_config.update(custom_config.get('env_config', {}))
            training_config.update(custom_config.get('training_config', {}))
    
    # Save experiment configuration
    logger.save_experiment_config({
        'method': args.method,
        'seed': args.seed,
        'env_config': env_config,
        'training_config': training_config
    })
    
    # Training
    if args.method in ["ppo", "both"]:
        _ = train_ppo(env_config, training_config, logger)
    
    if args.method in ["qlearning", "both"]:
        _ = train_q_learning(env_config, training_config, logger)
    
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