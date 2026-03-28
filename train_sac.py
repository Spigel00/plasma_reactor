#!/usr/bin/env python3
"""
Plasma Control RL Training with SAC

SAC (Soft Actor-Critic) training variant for plasma control.
SAC is particularly good for continuous control and may learn faster than PPO
by automatically balancing exploration through entropy regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# RL imports
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Our environment
from plasma_control_env import PlasmaControlEnv


class ActionLoggingCallback(BaseCallback):
    """Custom callback to log action statistics and reward components during training."""
    
    def __init__(self, log_dir, n_eval_episodes=5):
        super(ActionLoggingCallback, self).__init__()
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "sac_action_logging.txt"
        self.n_eval_episodes = n_eval_episodes
        self.last_log_step = 0
        self.probe_env = None
        
    def _on_step(self) -> bool:
        """Called after every environment step."""
        return True
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Use a standalone gym environment for logging rollouts.
        # self.training_env is a VecEnv and has different reset/step signatures.
        self.probe_env = PlasmaControlEnv(max_steps=50)
        print(f"SAC training logging to: {self.log_file}")
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        if self.num_timesteps - self.last_log_step >= 5000:  # Log every 5k steps
            self.last_log_step = self.num_timesteps
            
            # Log current policy statistics
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Timestep: {self.num_timesteps}\n")
                f.write(f"{'='*60}\n")
                
                # Get action statistics from a sample rollout
                env = self.probe_env
                obs, info = env.reset()
                episode_actions = []
                episode_rewards = []
                reward_components = {'shape': [], 'position': [], 'current': [], 
                                   'stability': [], 'control': [], 'success': []}
                
                for step in range(50):
                    action, _states = self.model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    
                    # Extract reward components
                    for key in reward_components:
                        if f'reward_{key}' in info:
                            reward_components[key].append(info[f'reward_{key}'])
                    
                    if terminated or truncated:
                        break
                
                episode_actions = np.array(episode_actions)
                episode_rewards = np.array(episode_rewards)
                
                # Write statistics
                f.write(f"\nAction Statistics (normalized [-1, 1]):\n")
                f.write(f"  Mean: {episode_actions.mean(axis=0)}\n")
                f.write(f"  Std:  {episode_actions.std(axis=0)}\n")
                f.write(f"  Min:  {episode_actions.min(axis=0)}\n")
                f.write(f"  Max:  {episode_actions.max(axis=0)}\n")
                f.write(f"\nReward Statistics:\n")
                f.write(f"  Episode Total: {episode_rewards.sum():.2f}\n")
                f.write(f"  Episode Mean: {episode_rewards.mean():.2f}\n")
                f.write(f"  Episode Std: {episode_rewards.std():.2f}\n")
                
                for key, values in reward_components.items():
                    if values:
                        f.write(f"  {key:12s}: mean={np.mean(values):7.3f} std={np.std(values):7.3f}\n")
                
                print(f"  [Step {self.num_timesteps}] Logged SAC action/reward statistics")

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.probe_env is not None:
            self.probe_env.close()
            self.probe_env = None


def train_plasma_controller_sac():
    """Train RL agent for plasma control using SAC algorithm."""
    
    print("🚀 Training Plasma Control Agent (SAC)")
    print("=" * 50)
    print("Using Soft Actor-Critic for continuous control")
    print("=" * 50)
    
    # Create directories for outputs
    log_dir = Path("./rl_training_logs_sac")
    model_dir = Path("./rl_models_sac")
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = PlasmaControlEnv(max_steps=50)
    train_env = Monitor(train_env, str(log_dir / "training_monitor.csv"))
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = PlasmaControlEnv(max_steps=50)
    eval_env = Monitor(eval_env, str(log_dir / "eval_monitor.csv"))
    
    # Create SAC model with optimized hyperparameters
    print("\nInitializing SAC agent...")
    print("SAC hyperparameters:")
    print(f"  learning_rate: 3e-4")
    print(f"  buffer_size: 1_000_000")
    print(f"  batch_size: 256")
    print(f"  tau: 0.005 (target network update)")
    print(f"  gamma: 0.99")
    print(f"  ent_coef: auto (automatic entropy adjustment)")
    
    model = SAC(
        "MlpPolicy",                # Multi-layer perceptron policy
        train_env,
        learning_rate=3e-4,         # Learning rate for actor and critic
        buffer_size=1_000_000,      # Replay buffer size
        batch_size=256,             # Batch size
        tau=0.005,                  # Target network update coefficient
        gamma=0.99,                 # Discount factor
        ent_coef='auto',            # Automatic entropy coefficient tuning
        target_entropy='auto',      # Automatic target entropy
        use_sde=True,               # Use State Dependent Exploration for better exploration
        sde_sample_freq=4,          # Sample exploration noise every N steps
        verbose=1,
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
    
    # Setup action logging callback
    action_callback = ActionLoggingCallback(log_dir)
    
    # Train the agent
    print("\nStarting SAC training...")
    print("SAC uses replay buffers and should converge faster for continuous control.")
    print("-" * 60)
    
    model.learn(
        total_timesteps=100_000,    # Same as PPO for fair comparison
        callback=[eval_callback, action_callback],
        tb_log_name="plasma_sac"
    )
    
    # Save final model
    final_model_path = model_dir / "final_plasma_model_sac"
    model.save(final_model_path)
    
    print(f"\n✅ SAC Training completed!")
    print(f"Model saved to: {final_model_path}")
    
    return model, final_model_path


def test_trained_model_sac(model_path):
    """Test the trained SAC model performance."""
    
    print(f"\n🧪 Testing Trained SAC Model")
    print("=" * 50)
    
    # Load trained model
    model = SAC.load(model_path)
    
    # Create test environment
    env = PlasmaControlEnv(max_steps=30)
    
    # Run test episodes
    n_test_episodes = 3
    total_rewards = []
    
    for episode in range(n_test_episodes):
        print(f"\nTest Episode {episode + 1}:")
        print("-" * 30)
        
        obs, info = env.reset(seed=42 + episode)
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
                      f"Coils (kA)=[{info['coil_currents'][0]:.1f},{info['coil_currents'][1]:.1f},"
                      f"{info['coil_currents'][2]:.1f},{info['coil_currents'][3]:.1f}]")
            
            if terminated:
                print(f"  Episode terminated at step {step + 1} (plasma disruption)")
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} Total Reward: {episode_reward:.2f}")
    
    # Calculate performance statistics
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\n📊 SAC Performance Summary:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best Episode: {max(total_rewards):.2f}")
    print(f"  Worst Episode: {min(total_rewards):.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train a new model
        model, model_path = train_plasma_controller_sac()
        
        # Test the trained model
        test_trained_model_sac(model_path)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test an existing model
        if len(sys.argv) > 2:
            test_trained_model_sac(sys.argv[2])
        else:
            print("Usage: python train_sac.py test <model_path>")
    
    else:
        print("Usage:")
        print("  Train new model:     python train_sac.py train")
        print("  Test existing model: python train_sac.py test <model_path>")
        print("\nRunning default: training new SAC model...")
        model, model_path = train_plasma_controller_sac()
