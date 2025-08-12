"""
Enhanced Evaluation Script for RL-Dewey-Tutor

Evaluates trained models (PPO and Q-Learning) with comprehensive analysis
including performance metrics, learning curves, and student progress tracking.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

# Project imports
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

        self.evaluation_results: Dict[str, Dict[str, Any]] = {}

    def _resolve_model_path(self, base_path: Path) -> Optional[Path]:
        if base_path.exists():
            return base_path
        zip_path = base_path.with_suffix('.zip')
        if zip_path.exists():
            return zip_path
        return None

    def evaluate_ppo(self, model_path: str, n_episodes: int = 100) -> Dict[str, Any]:
        print(f"Evaluating PPO model: {model_path}")
        path = self._resolve_model_path(Path(model_path))
        if path is None:
            raise FileNotFoundError(f"PPO model not found at {model_path} or {model_path}.zip")

        model = PPO.load(str(path), env=self.env)

        episode_rewards: List[float] = []
        final_skills: List[float] = []
        final_masteries: List[float] = []
        difficulty_distributions: List[List[Any]] = []
        performance_trajectories: List[List[Any]] = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_actions: List[Any] = []
            episode_performances: List[Any] = []

            for _ in range(self.env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += float(reward)
                episode_actions.append(action)
                episode_performances.append(info.get('performance', []))
                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            final_skills.append(float(np.mean(self.env.skill_levels)))
            final_masteries.append(float(np.mean(self.env.topic_mastery)))
            difficulty_distributions.append(episode_actions)
            performance_trajectories.append(episode_performances)

        ep_rewards = np.asarray(episode_rewards, dtype=float)
        results = {
            'model_type': 'PPO',
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(ep_rewards)),
            'std_reward': float(np.std(ep_rewards)),
            'min_reward': float(np.min(ep_rewards)),
            'max_reward': float(np.max(ep_rewards)),
            'mean_final_skill': float(np.mean(final_skills)) if final_skills else 0.0,
            'mean_final_mastery': float(np.mean(final_masteries)) if final_masteries else 0.0,
            'episode_rewards': episode_rewards,
            'final_skills': final_skills,
            'final_masteries': final_masteries,
            'difficulty_distributions': difficulty_distributions,
            'performance_trajectories': performance_trajectories
        }

        self.evaluation_results['ppo'] = results
        return results

    def evaluate_qlearning(self, model_path: str, n_episodes: int = 100) -> Dict[str, Any]:
        print(f"Evaluating Q-Learning model: {model_path}")

        agent = QLearningAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dims=self.env.action_space.nvec.tolist()
        )
        agent.load_model(model_path)

        episode_rewards: List[float] = []
        final_skills: List[float] = []
        final_masteries: List[float] = []
        difficulty_distributions: List[List[Any]] = []
        performance_trajectories: List[List[Any]] = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_actions: List[Any] = []
            episode_performances: List[Any] = []

            for _ in range(self.env.max_steps):
                # Robust action conversion: handle scalar or array-like from agent
                a = agent.select_action(obs, training=False)
                if isinstance(a, (int, np.integer)):
                    action = np.full(self.env.n_topics, int(a), dtype=int)
                else:
                    a_arr = np.array(a).ravel()
                    if a_arr.size == self.env.n_topics:
                        action = a_arr.astype(int)
                    elif a_arr.size >= 1:
                        action = np.full(self.env.n_topics, int(a_arr[0]), dtype=int)
                    else:
                        action = np.zeros(self.env.n_topics, dtype=int)

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += float(reward)
                episode_actions.append(action.tolist())
                episode_performances.append(info.get('performance', []))
                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            final_skills.append(float(np.mean(self.env.skill_levels)))
            final_masteries.append(float(np.mean(self.env.topic_mastery)))
            difficulty_distributions.append(episode_actions)
            performance_trajectories.append(episode_performances)

        ep_rewards = np.asarray(episode_rewards, dtype=float)
        results = {
            'model_type': 'Q-Learning',
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(ep_rewards)),
            'std_reward': float(np.std(ep_rewards)),
            'min_reward': float(np.min(ep_rewards)),
            'max_reward': float(np.max(ep_rewards)),
            'mean_final_skill': float(np.mean(final_skills)) if final_skills else 0.0,
            'mean_final_mastery': float(np.mean(final_masteries)) if final_masteries else 0.0,
            'episode_rewards': episode_rewards,
            'final_skills': final_skills,
            'final_masteries': final_masteries,
            'difficulty_distributions': difficulty_distributions,
            'performance_trajectories': performance_trajectories
        }

        self.evaluation_results['qlearning'] = results
        return results

    def compare_models(self) -> Dict[str, Any]:
        if len(self.evaluation_results) < 2:
            print("Need at least 2 models for comparison")
            return {}

        comparison: Dict[str, Any] = {}
        model_names = list(self.evaluation_results.keys())

        for metric in ['mean_reward', 'mean_final_skill', 'mean_final_mastery']:
            values = [self.evaluation_results[name][metric] for name in model_names]
            comparison[metric] = dict(zip(model_names, values))
            best_idx = int(np.argmax(values))
            comparison[f'{metric}_best'] = model_names[best_idx]

        if 'ppo' in self.evaluation_results and 'qlearning' in self.evaluation_results:
            from scipy import stats
            ppo_rewards = self.evaluation_results['ppo']['episode_rewards']
            ql_rewards = self.evaluation_results['qlearning']['episode_rewards']
            t_stat, p_value = stats.ttest_ind(ppo_rewards, ql_rewards, equal_var=False)
            comparison['statistical_test'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }

        return comparison

    def generate_plots(self):
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)

        # 1) Reward distributions - PROPERLY FIXED
        ax = axes[0, 0]
        for model_name, results in self.evaluation_results.items():
            rewards = results['episode_rewards']
            if len(rewards) > 0:
                # FIXED: Use fewer bins for small datasets to avoid clustering
                if len(rewards) <= 10:
                    n_bins = 3  # Force 3 bins for small datasets
                else:
                    n_bins = max(3, min(8, len(rewards) // 2))  # Cap at 8 bins max
                ax.hist(rewards, alpha=0.7, label=model_name, bins=n_bins, edgecolor='black', linewidth=0.5)
        ax.set_title('Episode Reward Distributions (fixed binning)')
        ax.set_xlabel('Total Episode Reward')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2) Final skill vs mastery
        ax = axes[0, 1]
        colors = ['blue', 'red', 'green', 'orange']
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            ax.scatter(results['final_skills'], results['final_masteries'],
                       alpha=0.6, label=model_name, color=colors[i % len(colors)])
        ax.set_title('Final Skill vs Mastery')
        ax.set_xlabel('Final Skill Level')
        ax.set_ylabel('Final Topic Mastery')
        ax.legend()

        # 3) Training curves - CLEARLY SEPARATED AND NON-OVERLAPPING
        ax = axes[1, 0]
        train_log_path = self.experiment_dir / 'logs' / 'training_log.csv'
        if train_log_path.exists():
            df = pd.read_csv(train_log_path)
            if {'method', 'metric', 'step', 'reward'}.issubset(df.columns):
                # FIXED: Separate PPO and Q-Learning into clear, non-overlapping sections
                
                # PPO evaluation rewards (main plot - left y-axis)
                dppo = df[(df['method'] == 'ppo') & (df['metric'] == 'eval_reward')]
                if not dppo.empty:
                    ax.plot(dppo['step'], dppo['reward'], label='PPO eval reward', color='tab:blue', alpha=0.9, linewidth=3)

                # Q-Learning evaluation rewards (main plot - left y-axis)
                dql_eval = df[(df['method'] == 'qlearning') & (df['metric'] == 'eval_reward')]
                if not dql_eval.empty:
                    ax.plot(dql_eval['step'], dql_eval['reward'], label='QL eval reward', color='tab:orange', alpha=0.9, linewidth=3)

                # Q-Learning step sum (episode conversion) - on separate y-axis to avoid overlap
                dql = df[(df['method'] == 'qlearning') & (df['metric'] == 'step_reward')]
                if not dql.empty:
                    dql_sorted = dql.sort_values('step')
                    # Convert step rewards to episode rewards (50 steps per episode)
                    episode_size = 50
                    episode_rewards = []
                    episode_steps = []

                    # Group steps into episodes and sum rewards
                    for i in range(0, len(dql_sorted), episode_size):
                        episode_data = dql_sorted.iloc[i:i+episode_size]
                        if len(episode_data) >= episode_size // 2:  # At least half episode
                            episode_rewards.append(episode_data['reward'].sum())
                            episode_steps.append(episode_data['step'].iloc[-1])

                    if episode_rewards:
                        # Create separate y-axis for step sum to avoid overlap
                        ax2 = ax.twinx()
                        ax2.plot(episode_steps, episode_rewards, label='QL step sum (50 steps)', color='tab:green', alpha=0.8, linewidth=2)
                        ax2.set_ylabel('QL Step Sum (50 steps)', color='tab:green')
                        ax2.tick_params(axis='y', labelcolor='tab:green')
                        # FIXED: Use completely different y-axis range to prevent overlap
                        # Since step sum is 150-300, use 0-1000 to give it its own visual space
                        ax2.set_ylim(0, 1000)
                        # Also offset the right spine to visually separate the axes
                        ax2.spines['right'].set_position(('outward', 60))

                # Q-Learning loss - on third y-axis (log scale)
                dql_loss = df[(df['method'] == 'qlearning') & (df['metric'] == 'loss')]
                if not dql_loss.empty:
                    ax3 = ax.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.plot(dql_loss['step'], dql_loss['reward'], color='tab:purple', alpha=0.9, label='QL loss', linewidth=2)
                    ax3.set_ylabel('QL Loss (log)', color='tab:purple')
                    ax3.set_yscale('log')
                    ax3.tick_params(axis='y', labelcolor='tab:purple')

                # FIXED: Clear legend handling - separate legends for each y-axis
                ax.legend(loc='upper left')
                if 'ax2' in locals():
                    ax2.legend(loc='upper center')
                if 'ax3' in locals():
                    ax3.legend(loc='upper right')

                ax.set_title('Training Curves (PPO vs Q-Learning) - CLEARLY SEPARATED')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Evaluation Reward', color='tab:blue')
                ax.tick_params(axis='y', labelcolor='tab:blue')
                # FIXED: Set main y-axis range to focus on eval rewards and prevent overlap
                ax.set_ylim(100, 300)  # Focus on eval reward range, avoid overlap with step sum
                ax.grid(True, alpha=0.3)
            else:
                if 'step' in df.columns and 'reward' in df.columns:
                    ax.plot(df['step'], df['reward'], alpha=0.3, label='reward (step)')
                    if len(df) > 50:
                        ax.plot(df['step'], df['reward'].rolling(50).mean(), label='reward (50-step MA)')
                ax.set_title('Training Curves (PPO vs Q-Learning)')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Reward')
        else:
            ax.text(0.5, 0.5, 'No training_log.csv found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves')

        # 4) Performance comparison - PROPERLY FIXED
        ax = axes[1, 1]
        if len(self.evaluation_results) >= 1:
            models = list(self.evaluation_results.keys())
            metrics = ['mean_reward', 'mean_final_skill', 'mean_final_mastery']
            vals = np.array([[self.evaluation_results[m][k] for k in metrics] for m in models], dtype=float)
            
            # FIXED normalization: normalize to meaningful baselines
            norm_vals = np.zeros_like(vals)
            # Reward: normalize to max 300 (reasonable max for this environment)
            norm_vals[:, 0] = np.clip(vals[:, 0] / 300.0, 0, 1)
            # Skill: already 0-1, just use as is
            norm_vals[:, 1] = vals[:, 1]
            # Mastery: already 0-1, just use as is
            norm_vals[:, 2] = vals[:, 2]
            
            x = np.arange(len(metrics))
            width = 0.35
            for i, m in enumerate(models):
                ax.bar(x + i * width, norm_vals[i], width, label=m, alpha=0.8)
            ax.set_title('Performance Comparison (fixed normalization)')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(['Reward/300', 'Skill', 'Mastery'])
            ax.set_ylim(0, 1)
            ax.legend()

        plt.tight_layout()
        plot_path = self.results_dir / "evaluation_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Evaluation plots saved to: {plot_path}")

    def generate_report(self) -> str:
        if not self.evaluation_results:
            return "No evaluation results available"

        report_lines: List[str] = []
        report_lines.append("=" * 60)
        report_lines.append("RL-DEWEY-TUTOR EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Experiment: {self.experiment_dir.name}")
        report_lines.append(f"Configuration: {json.dumps(self.config, indent=2)}")
        report_lines.append("")

        for model_name, results in self.evaluation_results.items():
            report_lines.append(f"--- {model_name.upper()} RESULTS ---")
            report_lines.append(f"Episodes evaluated: {results['n_episodes']}")
            report_lines.append(f"Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            report_lines.append(f"Reward range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
            report_lines.append(f"Final skill: {results['mean_final_skill']:.3f}")
            report_lines.append(f"Final mastery: {results['mean_final_mastery']:.3f}")
            
            # Add data quality indicators
            rewards = results['episode_rewards']
            if len(rewards) > 1:
                cv = results['std_reward'] / results['mean_reward']  # Coefficient of variation
                report_lines.append(f"Reward CV: {cv:.3f} ({'Low' if cv < 0.3 else 'Medium' if cv < 0.7 else 'High'} variance)")
                report_lines.append(f"Data points: {len(rewards)}")
            report_lines.append("")

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
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Evaluation report saved to: {report_path}")
        return report

    def save_results(self):
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)

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
    parser = argparse.ArgumentParser(description="RL-Dewey-Tutor Evaluation")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--models", nargs='+', default=['ppo', 'qlearning'], help="Models to evaluate")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.experiment)

    models_dir = evaluator.experiment_dir / "models"

    # Handle 'both' models case
    models_to_evaluate = args.models
    if 'both' in models_to_evaluate:
        models_to_evaluate = ['ppo', 'qlearning']
    
    if 'ppo' in models_to_evaluate:
        ppo_base = models_dir / "ppo_tutor"
        if evaluator._resolve_model_path(ppo_base) is not None:
            evaluator.evaluate_ppo(str(ppo_base), args.episodes)
        else:
            print(f"PPO model not found at: {ppo_base}(.zip)")

    if 'qlearning' in models_to_evaluate:
        ql_path = models_dir / "qlearning_tutor.pth"
        if ql_path.exists():
            evaluator.evaluate_qlearning(str(ql_path), args.episodes)
        else:
            print(f"Q-Learning model not found at: {ql_path}")

    evaluator.generate_plots()
    evaluator.generate_report()
    evaluator.save_results()

    print("\nEvaluation completed!")
    print(f"Results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()