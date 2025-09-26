#!/usr/bin/env python3
"""
Advanced Plasma Control RL Training

Complete training pipeline with hyperparameter tuning, monitoring, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import gymnasium as gym

# RL imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Our environment
from plasma_control_env import PlasmaControlEnv


class PlasmaRLTrainer:
    """Advanced RL trainer for plasma control with monitoring and evaluation."""
    
    def __init__(self, log_dir="./rl_logs", model_dir="./rl_models"):
        """Initialize trainer with logging directories."""
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        
        # Create directories
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            "algorithm": "PPO",
            "total_timesteps": 100000,
            "n_envs": 4,  # Parallel environments
            "eval_freq": 5000,
            "save_freq": 10000,
            
            # PPO hyperparameters (optimized for continuous control)
            "ppo_params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,  # Encourage exploration
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            
            # SAC hyperparameters (alternative algorithm)
            "sac_params": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1
            }
        }
        
        # Save config
        with open(self.log_dir / "training_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def create_training_environment(self):
        """Create vectorized training environment with monitoring."""
        
        def make_env(rank=0):
            """Create single environment instance."""
            def _init():
                env = PlasmaControlEnv(max_steps=100)
                env = Monitor(env, self.log_dir / f"monitor_{rank}.csv")
                return env
            return _init
        
        # Create vectorized environment (multiple parallel environments)
        vec_env = make_vec_env(make_env, n_envs=self.config["n_envs"])
        
        return vec_env
    
    def create_eval_environment(self):
        """Create evaluation environment."""
        eval_env = PlasmaControlEnv(max_steps=100)
        eval_env = Monitor(eval_env, self.log_dir / "eval_monitor.csv")
        return eval_env
    
    def setup_callbacks(self, eval_env):
        """Setup training callbacks for monitoring and saving."""
        
        # Evaluation callback - tests agent performance during training
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=self.config["eval_freq"],
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        
        # Checkpoint callback - saves model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config["save_freq"],
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="plasma_model"
        )
        
        return [eval_callback, checkpoint_callback]
    
    def train_ppo_agent(self):
        """Train PPO agent for plasma control."""
        
        print("🚀 Training PPO Agent for Plasma Control")
        print("=" * 50)
        
        # Create environments
        train_env = self.create_training_environment()
        eval_env = self.create_eval_environment()
        
        # Setup callbacks
        callbacks = self.setup_callbacks(eval_env)
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(self.log_dir / "tensorboard"),
            **self.config["ppo_params"]
        )
        
        # Configure logger
        model.set_logger(configure(str(self.log_dir), ["stdout", "csv", "tensorboard"]))
        
        print(f"Training for {self.config['total_timesteps']} timesteps...")
        print(f"Using {self.config['n_envs']} parallel environments")
        print(f"Evaluation every {self.config['eval_freq']} steps")
        
        # Train the agent
        model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=callbacks,
            tb_log_name="PPO_plasma_control"
        )
        
        # Save final model
        final_model_path = self.model_dir / "final_ppo_model"
        model.save(final_model_path)
        
        print(f"✅ Training completed!")
        print(f"Final model saved to: {final_model_path}")
        
        return model
    
    def train_sac_agent(self):
        """Train SAC agent for plasma control (alternative algorithm)."""
        
        print("🚀 Training SAC Agent for Plasma Control")
        print("=" * 50)
        
        # Create environments
        train_env = self.create_training_environment()
        eval_env = self.create_eval_environment()
        
        # Setup callbacks
        callbacks = self.setup_callbacks(eval_env)
        
        # Create SAC model
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(self.log_dir / "tensorboard"),
            **self.config["sac_params"]
        )
        
        print(f"Training SAC for {self.config['total_timesteps']} timesteps...")
        
        # Train the agent
        model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=callbacks,
            tb_log_name="SAC_plasma_control"
        )
        
        # Save final model
        final_model_path = self.model_dir / "final_sac_model"
        model.save(final_model_path)
        
        print(f"✅ SAC training completed!")
        print(f"Final model saved to: {final_model_path}")
        
        return model
    
    def evaluate_trained_model(self, model_path, n_episodes=10):
        """Evaluate trained model performance."""
        
        print(f"🔍 Evaluating Model: {model_path}")
        print("=" * 50)
        
        # Load model
        if "sac" in str(model_path).lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
        
        # Create test environment
        env = PlasmaControlEnv(max_steps=100)
        
        episode_rewards = []
        episode_lengths = []
        success_episodes = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            targets_met_history = []
            
            for step in range(100):
                # Use trained model to predict action
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                targets_met_history.append(info['targets_met'])
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check if episode was successful (most targets met most of the time)
            success_rate = np.mean([np.sum(list(targets.values())) for targets in targets_met_history])
            if success_rate > 2.5:  # At least 2.5/5 targets met on average
                success_episodes += 1
                
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}, Success Rate = {success_rate:.2f}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_percentage = (success_episodes / n_episodes) * 100
        
        print(f"\n📊 Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Episode Length: {mean_length:.1f}")
        print(f"  Success Rate: {success_percentage:.1f}%")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "success_rate": success_percentage,
            "episode_rewards": episode_rewards
        }
    
    def demonstrate_control(self, model_path, n_steps=20):
        """Demonstrate trained agent controlling plasma in real-time."""
        
        print(f"🎮 Live Plasma Control Demonstration")
        print("=" * 50)
        
        # Load model
        if "sac" in str(model_path).lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
        
        # Create environment
        env = PlasmaControlEnv(max_steps=n_steps)
        
        # Reset environment
        obs, info = env.reset()
        
        print("Initial State:")
        print(f"  Elongation: {obs[2]:.3f} (target: {env.target_elongation})")
        print(f"  Triangularity: {obs[3]:.3f} (target: {env.target_triangularity})")
        print(f"  R centroid: {obs[0]:.3f} m (target: {env.target_R_centroid})")
        print(f"  Plasma current: {obs[6]:.1f} MA (target: {env.target_Ip})")
        print()
        
        total_reward = 0
        control_history = []
        
        for step in range(n_steps):
            # Get AI control action
            action, _states = model.predict(obs, deterministic=True)
            
            # Apply control
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            control_history.append({
                'step': step + 1,
                'coils': action.copy(),
                'elongation': obs[2],
                'triangularity': obs[3],
                'R_centroid': obs[0],
                'Ip': obs[6],
                'reward': reward,
                'targets_met': info['targets_met']
            })
            
            # Print progress
            targets_met = sum(info['targets_met'].values())
            print(f"Step {step + 1:2d}: Reward={reward:6.2f} | Targets met: {targets_met}/5 | "
                  f"κ={obs[2]:.3f} δ={obs[3]:.3f} R={obs[0]:.3f} Ip={obs[6]:.1f}")
            
            if terminated:
                print("⚠️  Episode terminated - plasma disruption detected!")
                break
        
        print(f"\n🎯 Total Reward: {total_reward:.2f}")
        
        return control_history


def main():
    """Main training pipeline."""
    
    print("🌟 Plasma Control RL Training Pipeline")
    print("=" * 60)
    
    # Create trainer
    trainer = PlasmaRLTrainer()
    
    print("Configuration:")
    print(f"  Algorithm: {trainer.config['algorithm']}")
    print(f"  Total timesteps: {trainer.config['total_timesteps']:,}")
    print(f"  Parallel environments: {trainer.config['n_envs']}")
    print(f"  Evaluation frequency: {trainer.config['eval_freq']:,}")
    print()
    
    # Train PPO agent
    model = trainer.train_ppo_agent()
    
    # Evaluate final model
    best_model_path = trainer.model_dir / "best_model.zip"
    if best_model_path.exists():
        print("\n" + "="*60)
        results = trainer.evaluate_trained_model(best_model_path)
        
        # Demonstrate control
        print("\n" + "="*60)
        control_history = trainer.demonstrate_control(best_model_path)
        
    print(f"\n🎉 Training Pipeline Complete!")
    print(f"📁 Logs saved to: {trainer.log_dir}")
    print(f"🤖 Models saved to: {trainer.model_dir}")
    print(f"📊 TensorBoard: tensorboard --logdir {trainer.log_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()