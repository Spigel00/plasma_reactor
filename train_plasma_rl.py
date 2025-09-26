#!/usr/bin/env python3
"""
Train RL Agent for Plasma Control

This script demonstrates how to train an RL agent to control tokamak plasma
using our surrogate model and custom Gymnasium environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom environment
from plasma_control_env import PlasmaControlEnv

def train_simple_rl_agent():
    """
    Train a simple RL agent using random policy improvement.
    This is a basic example - in practice you'd use Stable-Baselines3.
    """
    
    print("Training Simple RL Agent for Plasma Control")
    print("=" * 50)
    
    # Create environment
    env = PlasmaControlEnv(max_steps=50)
    
    # Simple policy: start with random, keep good actions
    best_policy = None
    best_reward = -float('inf')
    
    rewards_history = []
    
    # Train for multiple episodes
    for episode in range(100):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        
        # Run episode
        for step in range(50):
            # Simple policy: random with slight bias toward previous good actions
            if best_policy is not None and np.random.random() < 0.7:
                # Use best policy with noise
                action = best_policy[step % len(best_policy)] + np.random.normal(0, 0.5, 4)
                action = np.clip(action, 5.0, 15.0)
            else:
                # Random action
                action = env.action_space.sample()
            
            episode_actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Keep track of best policy
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_policy = episode_actions[:10]  # Keep first 10 actions
            
        rewards_history.append(episode_reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Best = {best_reward:.2f}")
    
    return rewards_history, best_policy

def demonstrate_with_stable_baselines():
    """
    Show how to use with Stable-Baselines3 (requires pip install stable-baselines3)
    """
    
    print("\nDemonstrating Stable-Baselines3 Integration")
    print("=" * 50)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create vectorized environment (multiple parallel environments)
        env = make_vec_env(lambda: PlasmaControlEnv(), n_envs=4)
        
        # Create PPO agent
        model = PPO(
            "MlpPolicy",           # Multi-layer perceptron policy
            env, 
            learning_rate=3e-4,    # Learning rate
            n_steps=2048,          # Steps per update
            batch_size=64,         # Batch size
            verbose=1              # Print training progress
        )
        
        print("Training PPO agent for 10,000 steps...")
        # Train the agent
        model.learn(total_timesteps=10000)
        
        # Test trained agent
        print("Testing trained agent...")
        test_env = PlasmaControlEnv()
        obs, info = test_env.reset()
        
        total_reward = 0
        for step in range(20):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: Reward = {reward:.2f}, Targets met: {info['targets_met']}")
            
            if terminated or truncated:
                break
                
        print(f"Total test reward: {total_reward:.2f}")
        
        # Save trained model
        model.save("plasma_control_ppo_model")
        print("Model saved as 'plasma_control_ppo_model.zip'")
        
    except ImportError:
        print("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")
        print("Showing how the code would work...")
        
        code_example = '''
# With Stable-Baselines3 installed, you would do:

from stable_baselines3 import PPO
from plasma_control_env import PlasmaControlEnv

# Create environment
env = PlasmaControlEnv()

# Create and train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Use trained agent
obs, info = env.reset()
action, _states = model.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)
        '''
        print(code_example)

def visualize_control_performance():
    """Demonstrate the environment and visualize control performance."""
    
    print("Demonstrating Plasma Control Environment")
    print("=" * 50)
    
    env = PlasmaControlEnv()
    
    # Test different control strategies
    strategies = {
        "Random": lambda: env.action_space.sample(),
        "Baseline": lambda: np.array([10.0, 8.0, 12.0, 6.0]),
        "Shape-focused": lambda: np.array([9.0, 12.0, 8.0, 7.0]),  # Emphasize coil_2 for elongation
    }
    
    results = {}
    
    for strategy_name, policy in strategies.items():
        print(f"\nTesting {strategy_name} strategy:")
        
        obs, info = env.reset(seed=42)  # Fixed seed for comparison
        episode_rewards = []
        episode_elongations = []
        
        for step in range(20):
            action = policy()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_elongations.append(obs[2])  # elongation is index 2
            
            if step < 5:  # Show first few steps
                print(f"  Step {step + 1}: Elongation = {obs[2]:.3f}, Reward = {reward:.2f}")
            
            if terminated or truncated:
                break
        
        results[strategy_name] = {
            'rewards': episode_rewards,
            'elongations': episode_elongations,
            'total_reward': sum(episode_rewards)
        }
        
        print(f"  Total reward: {results[strategy_name]['total_reward']:.2f}")
    
    # Simple visualization
    print(f"\nStrategy Comparison:")
    for name, result in results.items():
        print(f"  {name}: Total Reward = {result['total_reward']:.2f}")
    
    return results

if __name__ == "__main__":
    print("🚀 Plasma Control RL Training Demo")
    print("=" * 60)
    
    # 1. Test the environment
    visualize_control_performance()
    
    # 2. Train simple agent
    print("\n" + "="*60)
    rewards_history, best_policy = train_simple_rl_agent()
    
    # 3. Show Stable-Baselines3 integration
    print("\n" + "="*60) 
    demonstrate_with_stable_baselines()
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Install Stable-Baselines3: pip install stable-baselines3")
    print("2. Run full RL training with PPO/SAC algorithms")
    print("3. Tune reward function for specific plasma objectives")
    print("4. Add more sophisticated physics constraints")