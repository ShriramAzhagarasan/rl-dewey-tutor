"""
Enhanced Evaluation Script for RL-Dewey-Tutor

Evaluates trained models (PPO and Q-Learning) with comprehensive analysis
including performance metrics, learning curves, and student progress tracking.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import argparse
from pathlib import Path

# Import our implementations
from envs.tutor_env import TutorEnv
from rl_dewey_tutor.agents.q_learning_agent import QLearningAgent
from stable_baselines3 import PPO

class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results_dir = self.experiment_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load experiment configuration
        config_path = self.experiment_dir / "experiment_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Initialize environment
        env_config = self.config.get('env_config', {})
        self.env = TutorEnv(**env_config)
        
        # Evaluation results
        self.evaluation_results = {}
    
    def _resolve_model_path(self, base_path: Path) -> Optional[Path]:
        """Return an existing path for the model, handling optional .zip for SB3 models"""
        if base_path.exists():
            return base_path
        # Try with .zip suffix (SB3 saves as .zip)
        zip_path = base_path.with_suffix('.zip')
        if zip_path.exists():
            return zip_path
        return None
    
    def evaluate_ppo(self, model_path: str, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate PPO model"""
        print(f"Evaluating PPO model: {model_path}")
        
        # Resolve possible .zip
        path = self._resolve_model_path(Path(model_path))
        if path is None:
            raise FileNotFoundError(f"PPO model not found at {model_path} or {model_path}.zip")
        
        # Load model
        model = PPO.load(str(path), env=self.env)
        
        # Evaluation metrics
        episode_rewards = []
        final_skills = []
        final_masteries = []
        difficulty_distributions = []
        performance_trajectories = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_actions = []
            episode_performances = []
            
            for step in range(self.env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_actions.append(action)
                episode_performances.append(info.get('performance', [0]))
                
                if done or truncated:
                    break
            
            # Record episode results
            episode_rewards.append(episode_reward)
            final_skills.append(np.mean(self.env.skill_levels))
            final_masteries.append(np.mean(self.env.topic_mastery))
            difficulty_distributions.append(episode_actions)
            performance_trajectories.append(episode_performances)
        
        # Calculate statistics
        results = {
            'model_type': 'PPO',
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_final_skill': np.mean(final_skills),
            'mean_final_mastery': np.mean(final_masteries),
            'episode_rewards': episode_rewards,
            'final_skills': final_skills,
            'final_masteries': final_masteries,
            'difficulty_distributions': difficulty_distributions,
            'performance_trajectories': performance_trajectories
        }
        
        self.evaluation_results['ppo'] = results
        return results
    
    def evaluate_qlearning(self, model_path: str, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate Q-Learning model"""
        print(f"Evaluating Q-Learning model: {model_path}")
        
        # Load model
        agent = QLearningAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dims=self.env.action_space.nvec
        )
        agent.load_model(model_path)
        
        # Evaluation metrics
        episode_rewards = []
        final_skills = []
        final_masteries = []
        difficulty_distributions = []
        performance_trajectories = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_actions = []
            episode_performances = []
            
            for step in range(self.env.max_steps):
                action = agent.select_action(obs, training=False)
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_actions.append(action)
                episode_performances.append(info.get('performance', [0]))
                
                if done or truncated:
                    break
            
            # Record episode results
            episode_rewards.append(episode_reward)
            final_skills.append(np.mean(self.env.skill_levels))
            final_masteries.append(np.mean(self.env.topic_mastery))
            difficulty_distributions.append(episode_actions)
            performance_trajectories.append(episode_performances)
        
        # Calculate statistics
        results = {
            'model_type': 'Q-Learning',
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_final_skill': np.mean(final_skills),
            'mean_final_mastery': np.mean(final_masteries),
            'episode_rewards': episode_rewards,
            'final_skills': final_skills,
            'final_masteries': final_masteries,
            'difficulty_distributions': difficulty_distributions,
            'performance_trajectories': performance_trajectories
        }
        
        self.evaluation_results['qlearning'] = results
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare performance between different models"""
        if len(self.evaluation_results) < 2:
            print("Need at least 2 models for comparison")
            return {}
        
        comparison = {}
        
        # Extract model names
        model_names = list(self.evaluation_results.keys())
        
        # Compare key metrics
        for metric in ['mean_reward', 'mean_final_skill', 'mean_final_mastery']:
            values = [self.evaluation_results[name][metric] for name in model_names]
            comparison[metric] = dict(zip(model_names, values))
            
            # Find best model
            best_idx = int(np.argmax(values))
            comparison[f'{metric}_best'] = model_names[best_idx]
        
        # Statistical significance test (simple t-test)
        if 'ppo' in self.evaluation_results and 'qlearning' in self.evaluation_results:
            ppo_rewards = self.evaluation_results['ppo']['episode_rewards']
            ql_rewards = self.evaluation_results['qlearning']['episode_rewards']
            
            # Perform t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(ppo_rewards, ql_rewards)
            
            comparison['statistical_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        return comparison
    
    def generate_plots(self):
        """Generate comprehensive evaluation plots"""
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # 1. Reward distributions
        ax = axes[0, 0]
        for model_name, results in self.evaluation_results.items():
            ax.hist(results['episode_rewards'], alpha=0.7, label=model_name, bins=20)
        ax.set_title('Episode Reward Distributions')
        ax.set_xlabel('Total Episode Reward')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 2. Final skill vs mastery scatter
        ax = axes[0, 1]
        colors = ['blue', 'red', 'green', 'orange']
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            ax.scatter(results['final_skills'], results['final_masteries'], 
                      alpha=0.6, label=model_name, color=colors[i % len(colors)])
        ax.set_title('Final Skill vs Mastery')
        ax.set_xlabel('Final Skill Level')
        ax.set_ylabel('Final Topic Mastery')
        ax.legend()
        
        # 3. Learning curves from training log (if available)
        ax = axes[1, 0]
        train_log_path = self.experiment_dir / 'logs' / 'training_log.csv'
        if train_log_path.exists():
            df = pd.read_csv(train_log_path)
            if {'method','metric','step','reward'}.issubset(df.columns):
                dppo = df[(df['method']=='ppo') & (df['metric']=='eval_reward')]
                dql = df[(df['method']=='qlearning') & (df['metric']=='step_reward')]
                if not dppo.empty:
                    ax.plot(dppo['step'], dppo['reward'], label='PPO eval reward', color='tab:blue')
                if not dql.empty:
                    dql_sorted = dql.sort_values('step')
                    ma = dql_sorted['reward'].rolling(200).mean()
                    ax.plot(dql_sorted['step'], ma, label='QL reward (200-step MA)', color='tab:orange')
                ax.legend()
            else:
                if 'step' in df.columns and 'reward' in df.columns:
                    ax.plot(df['step'], df['reward'], alpha=0.3, label='reward (step)')
                    if len(df) > 50:
                        ax.plot(df['step'], df['reward'].rolling(50).mean(), label='reward (50-step MA)')
            if 'loss' in df.columns:
                ax2 = ax.twinx()
                ax2.plot(df['step'], df['loss'], color='tab:red', alpha=0.2, label='loss')
                ax2.set_yscale('log')
                ax2.set_ylabel('Loss (log)')
            ax.set_title('Training Curves (separated & log-loss)')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
        else:
            ax.text(0.5, 0.5, 'No training_log.csv found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves')
        
        # 4. Performance comparison (bars)
        ax = axes[1, 1]
        if len(self.evaluation_results) >= 1:
            # Normalize metrics to [0,1] for mixed scales
            models = list(self.evaluation_results.keys())
            metrics = ['mean_reward', 'mean_final_skill', 'mean_final_mastery']
            vals = np.array([[self.evaluation_results[m][k] for k in metrics] for m in models], dtype=float)
            # min-max per metric
            vmin = vals.min(axis=0)
            vmax = vals.max(axis=0)
            denom = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
            norm_vals = (vals - vmin) / denom
            x = np.arange(len(metrics))
            width = 0.35
            for i, m in enumerate(models):
                ax.bar(x + i*width, norm_vals[i], width, label=m, alpha=0.8)
            ax.set_title('Performance Comparison (normalized)')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(['Reward', 'Skill', 'Mastery'])
            ax.set_ylim(0,1)
            ax.legend()
        
        plt.tight_layout()
        plot_path = self.results_dir / "evaluation_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to: {plot_path}")
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RL-DEWEY-TUTOR EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Experiment: {self.experiment_dir.name}")
        report_lines.append(f"Configuration: {json.dumps(self.config, indent=2)}")
        report_lines.append("")
        
        # Individual model results
        for model_name, results in self.evaluation_results.items():
            report_lines.append(f"--- {model_name.upper()} RESULTS ---")
            report_lines.append(f"Episodes evaluated: {results['n_episodes']}")
            report_lines.append(f"Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            report_lines.append(f"Reward range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
            report_lines.append(f"Final skill: {results['mean_final_skill']:.3f}")
            report_lines.append(f"Final mastery: {results['mean_final_mastery']:.3f}")
            report_lines.append("")
        
        # Model comparison
        if len(self.evaluation_results) > 1:
            comparison = self.compare_models()
            report_lines.append("--- MODEL COMPARISON ---")
            
            for metric in ['mean_reward', 'mean_final_skill', 'mean_final_mastery']:
                if metric in comparison:
                    report_lines.append(f"{metric}:")
                    for model_name, value in comparison[metric].items():
                        if isinstance(value, dict):
                            continue
                        report_lines.append(f"  {model_name}: {value:.3f}")
                    report_lines.append(f"  Best: {comparison.get(f'{metric}_best', 'n/a')}")
                    report_lines.append("")
            
            if 'statistical_test' in comparison:
                stats = comparison['statistical_test']
                report_lines.append("--- STATISTICAL ANALYSIS ---")
                report_lines.append(f"T-statistic: {stats['t_statistic']:.3f}")
                report_lines.append(f"P-value: {stats['p_value']:.4f}")
                report_lines.append(f"Significant difference: {'Yes' if stats['significant'] else 'No'}")
                report_lines.append("")
        
        report_lines.append("=" * 60)
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to: {report_path}")
        return report
    
    def save_results(self):
        """Save evaluation results to files"""
        # Save raw results
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'model': model_name,
                'mean_reward': results['mean_reward'],
                'std_reward': results['std_reward'],
                'mean_final_skill': results['mean_final_skill'],
                'mean_final_mastery': results['mean_final_mastery'],
                'n_episodes': results['n_episodes']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Evaluation results saved to: {results_path}")
        print(f"Summary saved to: {summary_path}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="RL-Dewey-Tutor Evaluation")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--models", nargs='+', default=['ppo', 'qlearning'],
                       help="Models to evaluate")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.experiment)
    
    # Evaluate models
    models_dir = evaluator.experiment_dir / "models"
    
    if 'ppo' in args.models:
        ppo_base = models_dir / "ppo_tutor"
        ppo_path = evaluator._resolve_model_path(ppo_base)
        if ppo_path is not None:
            evaluator.evaluate_ppo(str(ppo_base), args.episodes)
        else:
            print(f"PPO model not found at: {ppo_base}(.zip)")
    
    if 'qlearning' in args.models:
        ql_path = models_dir / "qlearning_tutor.pth"
        if ql_path.exists():
            evaluator.evaluate_qlearning(str(ql_path), args.episodes)
        else:
            print(f"Q-Learning model not found at: {ql_path}")
    
    # Generate analysis
    evaluator.generate_plots()
    evaluator.generate_report()
    evaluator.save_results()
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {evaluator.results_dir}")

if __name__ == "__main__":
    main()
