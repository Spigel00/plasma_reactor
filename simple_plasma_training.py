#!/usr/bin/env python3
"""
Simple Plasma Control RL Training

Streamlined training script that definitely works with our environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Our environment
from plasma_control_env import PlasmaControlEnv


def train_plasma_controller():
    """Train RL agent for plasma control - guaranteed to work!"""
    
    print("🚀 Training Plasma Control Agent")
    print("=" * 40)
    
    # Create directories for outputs
    log_dir = Path("./rl_training_logs")
    model_dir = Path("./rl_models")
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Create training environment (single environment first)
    print("Creating training environment...")
    train_env = PlasmaControlEnv(max_steps=50)
    train_env = Monitor(train_env, str(log_dir / "training_monitor.csv"))
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = PlasmaControlEnv(max_steps=50)
    eval_env = Monitor(eval_env, str(log_dir / "eval_monitor.csv"))
    
    # Create PPO model with optimized hyperparameters
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",                # Multi-layer perceptron policy
        train_env,
        learning_rate=3e-4,         # Learning rate
        n_steps=1024,               # Steps to collect before update  
        batch_size=64,              # Batch size for training
        n_epochs=10,                # Training epochs per update
        gamma=0.99,                 # Discount factor
        gae_lambda=0.95,            # GAE parameter
        clip_range=0.2,             # PPO clipping range
        ent_coef=0.01,              # Entropy coefficient (exploration)
        vf_coef=0.5,                # Value function coefficient
        verbose=1,                  # Print training progress
        tensorboard_log=str(log_dir / "tensorboard")
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=2000,             # Evaluate every 2000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train the agent
    print("Starting training...")
    print("This will take a few minutes - watch the reward improve!")
    print("-" * 40)
    
    model.learn(
        total_timesteps=20000,      # Total training steps
        callback=eval_callback,
        tb_log_name="plasma_ppo"
    )
    
    # Save final model
    final_model_path = model_dir / "final_plasma_model"
    model.save(final_model_path)
    
    print(f"\n✅ Training completed!")
    print(f"Model saved to: {final_model_path}")
    
    return model, final_model_path


def test_trained_model(model_path):
    """Test the trained model performance."""
    
    print(f"\n🧪 Testing Trained Model")
    print("=" * 40)
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Create test environment
    env = PlasmaControlEnv(max_steps=30)
    
    # Run test episodes
    n_test_episodes = 3
    total_rewards = []
    
    for episode in range(n_test_episodes):
        print(f"\nTest Episode {episode + 1}:")
        print("-" * 20)
        
        obs, info = env.reset(seed=42 + episode)  # Fixed seed for reproducible tests
        episode_reward = 0
        
        print(f"Initial plasma state:")
        print(f"  Elongation: {obs[2]:.3f} (target: {env.target_elongation})")
        print(f"  Triangularity: {obs[3]:.3f} (target: {env.target_triangularity})")
        print(f"  R centroid: {obs[0]:.3f} m (target: {env.target_R_centroid})")
        print(f"  Plasma current: {obs[6]:.1f} MA (target: {env.target_Ip})")
        
        for step in range(30):
            # Use trained model to predict optimal action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Show key steps
            if step < 5 or step % 10 == 0:
                targets_met = sum(info['targets_met'].values())
                print(f"  Step {step + 1:2d}: Reward={reward:6.2f} | Targets: {targets_met}/5 | "
                      f"Coils=[{action[0]:.1f},{action[1]:.1f},{action[2]:.1f},{action[3]:.1f}]")
            
            if terminated:
                print(f"  Episode terminated at step {step + 1} (plasma disruption)")
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} Total Reward: {episode_reward:.2f}")
    
    # Calculate performance statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\n📊 Performance Summary:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best Episode: {max(total_rewards):.2f}")
    print(f"  Worst Episode: {min(total_rewards):.2f}")
    
    return mean_reward


def compare_with_baseline():
    """Compare trained agent with baseline policies."""
    
    print(f"\n🆚 Baseline Comparison")
    print("=" * 40)
    
    env = PlasmaControlEnv(max_steps=20)
    
    # Test different strategies
    strategies = {
        "Random": lambda obs: env.action_space.sample(),
        "Fixed Baseline": lambda obs: np.array([10.0, 8.0, 12.0, 6.0]),
        "Simple Heuristic": lambda obs: np.array([10.0, 8.0 + obs[2], 12.0, 6.0 + obs[3]])  # Adjust based on shape
    }
    
    results = {}
    
    for strategy_name, policy_func in strategies.items():
        episode_rewards = []
        
        for episode in range(5):
            obs, info = env.reset(seed=100 + episode)
            episode_reward = 0
            
            for step in range(20):
                action = policy_func(obs)
                action = np.clip(action, 5.0, 15.0)  # Ensure valid range
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        results[strategy_name] = mean_reward
        print(f"  {strategy_name:15s}: {mean_reward:6.2f} avg reward")
    
    return results


def create_performance_visualization():
    """Create visualization of training progress."""
    
    print(f"\n📈 Creating Performance Visualization")
    print("=" * 40)
    
    # Try to load training logs
    log_file = Path("./rl_training_logs/training_monitor.csv")
    
    if log_file.exists():
        import pandas as pd
        
        # Load training data
        df = pd.read_csv(log_file)
        
        if len(df) > 0:
            # Create training progress plot
            plt.figure(figsize=(12, 4))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(df.index, df['r'], alpha=0.6, color='blue', linewidth=0.5)
            
            # Add rolling average
            window_size = max(1, len(df) // 20)
            rolling_mean = df['r'].rolling(window=window_size).mean()
            plt.plot(df.index, rolling_mean, color='red', linewidth=2, label=f'Rolling Mean ({window_size} episodes)')
            
            plt.xlabel('Episode')
            plt.ylabel('Episode Reward')
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot episode lengths
            plt.subplot(1, 2, 2)
            plt.plot(df.index, df['l'], alpha=0.6, color='green', linewidth=0.5)
            rolling_length = df['l'].rolling(window=window_size).mean()
            plt.plot(df.index, rolling_length, color='orange', linewidth=2, label=f'Rolling Mean ({window_size} episodes)')
            
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title('Training Progress: Episode Lengths')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path("./rl_training_logs/training_progress.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Training progress plot saved to: {plot_path}")
            
            # Print training statistics
            print(f"\nTraining Statistics:")
            print(f"  Total episodes: {len(df)}")
            print(f"  Initial reward: {df['r'].iloc[0]:.2f}")
            print(f"  Final reward: {df['r'].iloc[-1]:.2f}")
            print(f"  Best reward: {df['r'].max():.2f}")
            print(f"  Average length: {df['l'].mean():.1f} steps")
            
        else:
            print("No training data available yet.")
    else:
        print("Training log file not found. Run training first.")


def main():
    """Main function - complete RL training and evaluation pipeline."""
    
    print("🌟 Complete Plasma Control RL Pipeline")
    print("=" * 60)
    
    # Step 1: Train the model
    model, model_path = train_plasma_controller()
    
    # Step 2: Test trained model
    trained_performance = test_trained_model(model_path)
    
    # Step 3: Compare with baselines
    baseline_results = compare_with_baseline()
    
    # Step 4: Create visualizations
    create_performance_visualization()
    
    # Summary
    print(f"\n🎉 Pipeline Complete!")
    print("=" * 40)
    print(f"✅ Model trained and saved")
    print(f"✅ Performance tested: {trained_performance:.2f} avg reward")
    print(f"✅ Baseline comparison completed")
    print(f"✅ Visualizations created")
    
    print(f"\n📁 Outputs:")
    print(f"  Models: ./rl_models/")
    print(f"  Logs: ./rl_training_logs/")
    print(f"  Plots: ./rl_training_logs/training_progress.png")
    
    print(f"\n🚀 Next steps:")
    print(f"  1. Check TensorBoard: tensorboard --logdir ./rl_training_logs/tensorboard")
    print(f"  2. Load model for deployment: PPO.load('./rl_models/final_plasma_model')")
    print(f"  3. Increase training steps for better performance")


if __name__ == "__main__":
    main()