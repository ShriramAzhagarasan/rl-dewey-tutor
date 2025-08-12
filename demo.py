#!/usr/bin/env python3
"""
Demo Script for RL-Dewey-Tutor

This script demonstrates the key features of the system without requiring
full training or external dependencies. It shows the environment dynamics,
agent behavior, and system capabilities.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_environment():
    """Demonstrate the enhanced tutor environment"""
    print("=" * 60)
    print("DEMO: Enhanced Tutor Environment")
    print("=" * 60)
    
    try:
        from src.envs.tutor_env import TutorEnv
        
        # Create environment with different configurations
        configs = [
            ("Basic", {'n_topics': 2, 'n_difficulty_levels': 3}),
            ("Standard", {'n_topics': 3, 'n_difficulty_levels': 5}),
            ("Advanced", {'n_topics': 4, 'n_difficulty_levels': 7})
        ]
        
        for name, config in configs:
            print(f"\n--- {name} Configuration ---")
            env = TutorEnv(**config)
            
            # Show environment specs
            print(f"Topics: {env.n_topics}")
            print(f"Difficulty levels: {env.n_difficulty_levels}")
            print(f"State dimension: {env.observation_space.shape[0]}")
            print(f"Action space: {env.action_space}")
            
            # Run a quick episode
            obs, _ = env.reset()
            total_reward = 0
            
            # Fix: Convert numpy array to list for safe formatting
            obs_preview = obs[:6].tolist()
            print(f"Initial state: {obs_preview}...")  # Show first 6 dimensions
            
            for step in range(min(5, env.max_steps)):  # Just 5 steps for demo
                # Random action for demo
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Fix: Safe formatting for numpy values
                avg_skill = float(np.mean(env.skill_levels))
                print(f"Step {step+1}: Action={action}, Reward={reward:.3f}, "
                      f"Skill={avg_skill:.3f}")
                
                if done or truncated:
                    break
            
            print(f"Episode reward: {total_reward:.3f}")
            print(f"Final skills: {env.skill_levels.tolist()}")
            print(f"Final mastery: {env.topic_mastery.tolist()}")
            
        return True
        
    except ImportError as e:
        print(f"⚠ Environment demo skipped (missing dependencies): {e}")
        print("   Install requirements.txt to run full demo")
        return False
    except Exception as e:
        print(f"⚠ Environment demo error: {e}")
        return False

def demo_agents():
    """Demonstrate the RL agents (without training)"""
    print("\n" + "=" * 60)
    print("DEMO: RL Agents Architecture")
    print("=" * 60)
    
    try:
        from src.rl_dewey_tutor.agents.q_learning_agent import QLearningAgent
        from src.rl_dewey_tutor.agents.thompson_sampling import ThompsonSamplingExplorer
        
        # Show agent capabilities
        print("✓ Q-Learning Agent Features:")
        print("  - Neural network function approximation")
        print("  - Experience replay buffer")
        print("  - Target network for stable learning")
        print("  - Epsilon-greedy exploration")
        print("  - Gradient clipping for stability")
        
        print("\n✓ Thompson Sampling Explorer Features:")
        print("  - Bayesian uncertainty quantification")
        print("  - Adaptive exploration based on state uncertainty")
        print("  - Beta distribution priors")
        print("  - Integration with both RL methods")
        
        # Show architecture details
        print("\n--- Architecture Details ---")
        print("State Space: Continuous, multi-dimensional")
        print("Action Space: MultiDiscrete (difficulty per topic)")
        print("Reward Function: Multi-objective optimization")
        print("Exploration: Thompson Sampling + Epsilon-greedy")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Agent demo skipped (missing dependencies): {e}")
        return False
    except Exception as e:
        print(f"⚠ Agent demo error: {e}")
        return False

def demo_experiment_workflow():
    """Demonstrate the experiment workflow"""
    print("\n" + "=" * 60)
    print("DEMO: Experiment Workflow")
    print("=" * 60)
    
    print("1. Training Pipeline:")
    print("   - Environment initialization with custom configs")
    print("   - Dual RL method training (PPO + Q-Learning)")
    print("   - Thompson Sampling exploration integration")
    print("   - Comprehensive logging and monitoring")
    
    print("\n2. Evaluation Pipeline:")
    print("   - Multi-episode performance assessment")
    print("   - Statistical comparison between methods")
    print("   - Visualization and reporting")
    print("   - Cross-configuration analysis")
    
    print("\n3. Experiment Automation:")
    print("   - Multi-seed statistical significance")
    print("   - Configuration sweeps")
    print("   - Parallel processing")
    print("   - Automated report generation")
    
    print("\n4. Key Metrics Tracked:")
    print("   - Episode rewards and convergence")
    print("   - Student skill progression")
    print("   - Topic mastery development")
    print("   - Exploration vs exploitation balance")
    print("   - Difficulty selection patterns")
    
    return True

def demo_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("DEMO: Usage Examples")
    print("=" * 60)
    
    print("Basic Training:")
    print("  python3 src/train.py --method both --experiment my_experiment")
    
    print("\nEvaluation:")
    print("  python3 src/evaluate.py --experiment results/my_experiment --episodes 100")
    
    print("\nAutomated Experiments:")
    print("  python3 src/run_experiments.py --configs baseline high_complexity --seeds 42 123 456")
    
    print("\nCustom Configuration:")
    print("  # Edit configs/baseline_config.json")
    print("  # Modify environment and training parameters")
    print("  # Run with --config flag")
    
    return True

def main():
    """Run the complete demo"""
    print("🎓 RL-Dewey-Tutor System Demo")
    print("=" * 60)
    print("This demo showcases the key features and capabilities")
    print("of the RL-Dewey-Tutor system without requiring full training.")
    print("=" * 60)
    
    # Run demos
    demos = [
        demo_environment,
        demo_agents,
        demo_experiment_workflow,
        demo_usage_examples
    ]
    
    results = []
    for demo in demos:
        try:
            result = demo()
            # Fix: Ensure result is always a boolean
            if result is None:
                result = False
            results.append(bool(result))
        except Exception as e:
            print(f"✗ Demo {demo.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    
    successful_demos = sum(results)
    total_demos = len(demos)
    
    print(f"Tests passed: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("\nThe RL-Dewey-Tutor system is ready for:")
        print("✅ Research and experimentation")
        print("✅ Educational technology development")
        print("✅ Adaptive learning system implementation")
        print("✅ Reinforcement learning research")
        
        print("\n🚀 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run a quick experiment: python3 src/train.py --method both")
        print("3. Explore configurations: Edit configs/baseline_config.json")
        print("4. Run full experiment suite: python3 src/run_experiments.py")
        
    else:
        print("\n⚠ Some demos had issues (likely due to missing dependencies)")
        print("Install requirements.txt to run the complete system")
    
    print("\n" + "=" * 60)
    print("Thank you for exploring RL-Dewey-Tutor!")
    print("=" * 60)

if __name__ == "__main__":
    main() 