"""
Experiment Automation Script for RL-Dewey-Tutor

Runs comprehensive experiments with multiple seeds, configurations, and statistical analysis
for reproducible reinforcement learning research.
"""

import os
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class ExperimentRunner:
    """Automated experiment runner with statistical analysis"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Experiment configurations
        self.experiment_configs = self._get_default_configs()
        
        # Results storage
        self.experiment_results = {}
        self.comparison_results = {}
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default experiment configurations"""
        return {
            'baseline': {
                'env_config': {
                    'n_topics': 3,
                    'n_difficulty_levels': 5,
                    'max_steps': 50,
                    'skill_decay_rate': 0.02,
                    'learning_rate_variance': 0.1
                },
                'training_config': {
                    'gamma': 0.99,
                    'learning_rate': 0.001,
                    'total_timesteps': 100_000,
                    'total_episodes': 500,
                    'eval_freq': 25,
                    'exploration_strength': 1.0
                }
            },
            'high_complexity': {
                'env_config': {
                    'n_topics': 5,
                    'n_difficulty_levels': 7,
                    'max_steps': 75,
                    'skill_decay_rate': 0.03,
                    'learning_rate_variance': 0.15
                },
                'training_config': {
                    'gamma': 0.99,
                    'learning_rate': 0.0005,
                    'total_timesteps': 150_000,
                    'total_episodes': 750,
                    'eval_freq': 25,
                    'exploration_strength': 1.2
                }
            },
            'fast_learning': {
                'env_config': {
                    'n_topics': 2,
                    'n_difficulty_levels': 4,
                    'max_steps': 30,
                    'skill_decay_rate': 0.01,
                    'learning_rate_variance': 0.05
                },
                'training_config': {
                    'gamma': 0.95,
                    'learning_rate': 0.002,
                    'total_timesteps': 75_000,
                    'total_episodes': 300,
                    'eval_freq': 20,
                    'exploration_strength': 0.8
                }
            },
            'no_reward_shaping': {
                'env_config': {
                    'n_topics': 3,
                    'n_difficulty_levels': 5,
                    'max_steps': 50,
                    'skill_decay_rate': 0.02,
                    'learning_rate_variance': 0.1,
                    'reward_shaping': False
                },
                'training_config': {
                    'gamma': 0.99,
                    'learning_rate': 0.001,
                    'total_timesteps': 100_000,
                    'total_episodes': 500,
                    'eval_freq': 25,
                    'exploration_strength': 1.0
                }
            }
        }
    
    def run_single_experiment(self, config_name: str, seed: int, 
                             method: str = "both") -> Dict[str, Any]:
        """Run a single experiment with given configuration and seed"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config_name}_seed{seed}_{timestamp}"
        
        # Create configuration file
        config = self.experiment_configs[config_name].copy()
        config['seed'] = seed
        
        config_path = self.base_dir / f"{experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training
        cmd = [
            "python", "src/train.py",
            "--method", method,
            "--experiment", experiment_name,
            "--seed", str(seed),
            "--config", str(config_path)
        ]
        
        print(f"Running experiment: {experiment_name}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run training
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            training_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"Training failed for {experiment_name}: {result.stderr}")
                return {
                    'experiment_name': experiment_name,
                    'config_name': config_name,
                    'seed': seed,
                    'method': method,
                    'status': 'failed',
                    'error': result.stderr,
                    'training_time': training_time
                }
            
            # Run evaluation
            eval_cmd = [
                "python", "src/evaluate.py",
                "--experiment", f"results/{experiment_name}",
                "--episodes", "50",
                "--models", "ppo", "qlearning"
            ]
            
            eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=".")
            
            if eval_result.returncode != 0:
                print(f"Evaluation failed for {experiment_name}: {eval_result.stderr}")
            
            # Load results
            results_dir = Path(f"results/{experiment_name}")
            eval_results_dir = results_dir / "evaluation_results"
            
            if eval_results_dir.exists():
                with open(eval_results_dir / "evaluation_results.json", 'r') as f:
                    eval_results = json.load(f)
            else:
                eval_results = {}
            
            return {
                'experiment_name': experiment_name,
                'config_name': config_name,
                'seed': seed,
                'method': method,
                'status': 'completed',
                'training_time': training_time,
                'evaluation_results': eval_results,
                'config_path': str(config_path)
            }
            
        except Exception as e:
            print(f"Error running experiment {experiment_name}: {str(e)}")
            return {
                'experiment_name': experiment_name,
                'config_name': config_name,
                'seed': seed,
                'method': method,
                'status': 'error',
                'error': str(e),
                'training_time': 0
            }
    
    def run_experiment_suite(self, config_names: List[str], seeds: List[int], 
                            method: str = "both", max_workers: int = None) -> Dict[str, Any]:
        """Run a suite of experiments with parallel processing"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 4)  # Limit to avoid overwhelming system
        
        print(f"Running experiment suite with {max_workers} workers")
        print(f"Configurations: {config_names}")
        print(f"Seeds: {seeds}")
        print(f"Method: {method}")
        
        # Generate all experiment combinations
        experiments = []
        for config_name in config_names:
            for seed in seeds:
                experiments.append((config_name, seed))
        
        print(f"Total experiments: {len(experiments)}")
        
        # Run experiments
        results = {}
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(self.run_single_experiment, config, seed, method): (config, seed)
                for config, seed in experiments
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_exp):
                config, seed = future_to_exp[future]
                try:
                    result = future.result()
                    results[f"{config}_seed{seed}"] = result
                    
                    # Print progress
                    completed = len(results)
                    total = len(experiments)
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                    
                    print(f"Completed {completed}/{total} experiments "
                          f"({completed/total*100:.1f}%) - ETA: {eta/60:.1f} minutes")
                    
                except Exception as e:
                    print(f"Experiment {config}_seed{seed} failed: {str(e)}")
                    results[f"{config}_seed{seed}"] = {
                        'experiment_name': f"{config}_seed{seed}",
                        'config_name': config,
                        'seed': seed,
                        'method': method,
                        'status': 'error',
                        'error': str(e),
                        'training_time': 0
                    }
        
        total_time = time.time() - start_time
        print(f"\nExperiment suite completed in {total_time/60:.1f} minutes")
        
        # Store results
        self.experiment_results.update(results)
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results with statistical comparisons"""
        if not self.experiment_results:
            print("No experiment results to analyze")
            return {}
        
        analysis = {}
        
        # Group results by configuration
        config_groups = {}
        for exp_name, result in self.experiment_results.items():
            if result['status'] != 'completed':
                continue
            
            config_name = result['config_name']
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(result)
        
        # Analyze each configuration
        for config_name, results in config_groups.items():
            if not results:
                continue
            
            config_analysis = {
                'n_experiments': len(results),
                'success_rate': len([r for r in results if r['status'] == 'completed']) / len(results),
                'avg_training_time': np.mean([r['training_time'] for r in results]),
                'performance_metrics': {}
            }
            
            # Extract performance metrics
            ppo_rewards = []
            ql_rewards = []
            ppo_skills = []
            ql_skills = []
            ppo_masteries = []
            ql_masteries = []
            
            for result in results:
                if 'evaluation_results' not in result:
                    continue
                
                eval_results = result['evaluation_results']
                
                if 'ppo' in eval_results:
                    ppo_rewards.append(eval_results['ppo']['mean_reward'])
                    ppo_skills.append(eval_results['ppo']['mean_final_skill'])
                    ppo_masteries.append(eval_results['ppo']['mean_final_mastery'])
                
                if 'qlearning' in eval_results:
                    ql_rewards.append(eval_results['qlearning']['mean_reward'])
                    ql_skills.append(eval_results['qlearning']['mean_final_skill'])
                    ql_masteries.append(eval_results['qlearning']['mean_final_mastery'])
            
            # Calculate statistics
            if ppo_rewards:
                config_analysis['performance_metrics']['ppo'] = {
                    'mean_reward': np.mean(ppo_rewards),
                    'std_reward': np.std(ppo_rewards),
                    'mean_skill': np.mean(ppo_skills),
                    'std_skill': np.std(ppo_skills),
                    'mean_mastery': np.mean(ppo_masteries),
                    'std_mastery': np.std(ppo_masteries)
                }
            
            if ql_rewards:
                config_analysis['performance_metrics']['qlearning'] = {
                    'mean_reward': np.mean(ql_rewards),
                    'std_reward': np.std(ql_rewards),
                    'mean_skill': np.mean(ql_skills),
                    'std_skill': np.std(ql_skills),
                    'mean_mastery': np.mean(ql_masteries),
                    'std_mastery': np.std(ql_masteries)
                }
            
            analysis[config_name] = config_analysis
        
        # Cross-configuration comparison
        if len(config_groups) > 1:
            analysis['cross_config_comparison'] = self._compare_configurations(config_groups)
        
        self.comparison_results = analysis
        return analysis
    
    def _compare_configurations(self, config_groups: Dict[str, List]) -> Dict[str, Any]:
        """Compare performance across different configurations"""
        comparison = {}
        
        # Compare PPO performance across configurations
        ppo_comparison = {}
        for config_name, results in config_groups.items():
            ppo_rewards = []
            for result in results:
                if ('evaluation_results' in result and 
                    'ppo' in result['evaluation_results']):
                    ppo_rewards.append(result['evaluation_results']['ppo']['mean_reward'])
            
            if ppo_rewards:
                ppo_comparison[config_name] = {
                    'mean': np.mean(ppo_rewards),
                    'std': np.std(ppo_rewards),
                    'n_samples': len(ppo_rewards)
                }
        
        if len(ppo_comparison) > 1:
            comparison['ppo'] = ppo_comparison
        
        # Compare Q-Learning performance across configurations
        ql_comparison = {}
        for config_name, results in config_groups.items():
            ql_rewards = []
            for result in results:
                if ('evaluation_results' in result and 
                    'qlearning' in result['evaluation_results']):
                    ql_rewards.append(result['evaluation_results']['qlearning']['mean_reward'])
            
            if ql_rewards:
                ql_comparison[config_name] = {
                    'mean': np.mean(ql_rewards),
                    'std': np.std(ql_rewards),
                    'n_samples': len(ql_rewards)
                }
        
        if len(ql_comparison) > 1:
            comparison['qlearning'] = ql_comparison
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate comprehensive experiment report"""
        if not self.comparison_results:
            self.analyze_results()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RL-DEWEY-TUTOR COMPREHENSIVE EXPERIMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Total experiments: {len(self.experiment_results)}")
        report_lines.append("")
        
        # Summary statistics
        successful_experiments = [r for r in self.experiment_results.values() if r['status'] == 'completed']
        failed_experiments = [r for r in self.experiment_results.values() if r['status'] != 'completed']
        
        report_lines.append("--- EXPERIMENT SUMMARY ---")
        report_lines.append(f"Successful experiments: {len(successful_experiments)}")
        report_lines.append(f"Failed experiments: {len(failed_experiments)}")
        report_lines.append(f"Success rate: {len(successful_experiments)/len(self.experiment_results)*100:.1f}%")
        report_lines.append("")
        
        # Configuration analysis
        for config_name, analysis in self.comparison_results.items():
            if config_name == 'cross_config_comparison':
                continue
                
            report_lines.append(f"--- {config_name.upper()} CONFIGURATION ---")
            report_lines.append(f"Experiments: {analysis['n_experiments']}")
            report_lines.append(f"Success rate: {analysis['success_rate']*100:.1f}%")
            report_lines.append(f"Average training time: {analysis['avg_training_time']/60:.1f} minutes")
            report_lines.append("")
            
            # Performance metrics
            if 'performance_metrics' in analysis:
                for method, metrics in analysis['performance_metrics'].items():
                    report_lines.append(f"  {method.upper()}:")
                    report_lines.append(f"    Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
                    report_lines.append(f"    Skill: {metrics['mean_skill']:.3f} ± {metrics['std_skill']:.3f}")
                    report_lines.append(f"    Mastery: {metrics['mean_mastery']:.3f} ± {metrics['std_mastery']:.3f}")
                report_lines.append("")
        
        # Cross-configuration comparison
        if 'cross_config_comparison' in self.comparison_results:
            report_lines.append("--- CROSS-CONFIGURATION COMPARISON ---")
            comparison = self.comparison_results['cross_config_comparison']
            
            for method, configs in comparison.items():
                report_lines.append(f"{method.upper()}:")
                for config_name, stats in configs.items():
                    report_lines.append(f"  {config_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.base_dir / "experiment_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Experiment report saved to: {report_path}")
        return report
    
    def save_results(self):
        """Save all experiment results and analysis"""
        # Save raw results
        results_path = self.base_dir / "experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)
        
        # Save analysis
        analysis_path = self.base_dir / "experiment_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        # Compact CSV summary across seeds/configs (mean±std where available)
        rows = []
        for name, res in self.experiment_results.items():
            row = {
                'experiment': name,
                'config': res.get('config_name'),
                'seed': res.get('seed'),
                'status': res.get('status')
            }
            eval_res = res.get('evaluation_results', {})
            for m in ['ppo','qlearning']:
                if m in eval_res:
                    row[f'{m}_mean_reward'] = eval_res[m].get('mean_reward')
                    row[f'{m}_std_reward'] = eval_res[m].get('std_reward')
                    row[f'{m}_final_skill'] = eval_res[m].get('mean_final_skill')
                    row[f'{m}_final_mastery'] = eval_res[m].get('mean_final_mastery')
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.base_dir / 'experiment_summary.csv', index=False)
        
        print(f"Experiment results saved to: {results_path}")
        print(f"Experiment analysis saved to: {analysis_path}")

def main():
    """Main experiment runner function"""
    parser = argparse.ArgumentParser(description="RL-Dewey-Tutor Experiment Runner")
    parser.add_argument("--configs", nargs='+', default=['baseline'],
                       help="Configuration names to run")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 456],
                       help="Random seeds to use")
    parser.add_argument("--method", choices=["ppo", "qlearning", "both"], 
                       default="both", help="RL method to use")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner()
    
    # Run experiments
    results = runner.run_experiment_suite(
        config_names=args.configs,
        seeds=args.seeds,
        method=args.method,
        max_workers=args.workers
    )
    
    # Analyze and report
    runner.analyze_results()
    runner.generate_report()
    runner.save_results()
    
    print("\nExperiment suite completed!")
    print(f"Results saved to: {runner.base_dir}")

if __name__ == "__main__":
    main() 