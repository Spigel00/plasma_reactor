#!/usr/bin/env python3
"""
Direct Plasma Control Deployment - Tests the trained model in actual plasma control
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from plasma_control_env import PlasmaControlEnv


def deploy_trained_model():
    """Deploy and test trained model on plasma control task."""
    
    print("\n" + "="*70)
    print("PLASMA CONTROL DEPLOYMENT TEST")
    print("Using Trained RL Model with Corrected Reward System")
    print("="*70 + "\n")
    
    # Use the best trained model
    model_path = Path.cwd() / "rl_models" / "best_model.zip"
    if not model_path.exists():
        model_path = Path.cwd() / "rl_models" / "final_plasma_model.zip"
    
    print(f"Loading model: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    model = PPO.load(str(model_path))
    print("✅ Model loaded successfully\n")
    
    # Create plasma environment
    print("Creating plasma control environment (max_steps=150)...")
    env = PlasmaControlEnv(max_steps=150)
    print("✅ Environment initialized\n")
    
    # Run deployment scenario
    print("INITIATING PLASMA CONTROL SEQUENCE")
    print("-" * 70)
    
    obs, _ = env.reset(seed=100)
    
    # Store history
    history = {
        'time': [],
        'elongation': [],
        'triangularity': [],
        'R_centroid': [],
        'Z_centroid': [],
        'Ip': [],
        'coil_1': [],
        'coil_2': [],
        'coil_3': [],
        'coil_4': [],
        'reward': [],
        'targets_met': [],
        'q95': []
    }
    
    total_reward = 0
    max_targets_met = 0
    step_achieved_targets = 0
    
    print(f"\nInitial State:")
    print(f"  Elongation: {obs[2]:.3f} (target: 1.800)")
    print(f"  Triangularity: {obs[3]:.3f} (target: 0.400)")
    print(f"  R centroid: {obs[0]:.3f} m (target: 1.650)")
    print(f"  Plasma current: {obs[6]:.1f} MA (target: 15.000)")
    print(f"  q95: {obs[7]:.2f}")
    print("\nControl Steps:")
    print("-" * 70)
    
    for step in range(150):
        # Get optimal action from trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute control action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        targets_met = sum(info['targets_met'].values())
        if targets_met > max_targets_met:
            max_targets_met = targets_met
            step_achieved_targets = step + 1
        
        # Record history
        history['time'].append(step)
        history['elongation'].append(obs[2])
        history['triangularity'].append(obs[3])
        history['R_centroid'].append(obs[0])
        history['Z_centroid'].append(obs[1])
        history['Ip'].append(obs[6])
        history['coil_1'].append(action[0])
        history['coil_2'].append(action[1])
        history['coil_3'].append(action[2])
        history['coil_4'].append(action[3])
        history['reward'].append(reward)
        history['targets_met'].append(targets_met)
        history['q95'].append(obs[7])
        
        # Print key steps
        if step < 5 or step % 30 == 0:
            status = "✓ CONTROLLED" if targets_met >= 3 else "• Adjusting"
            print(f"  Step {step+1:3d} [{status}]: Targets={targets_met}/5 | "
                  f"κ={obs[2]:5.2f} δ={obs[3]:4.2f} R={obs[0]:.2f} | "
                  f"I_p={obs[6]:5.1f} MA | Reward={reward:+6.2f}")
        
        if terminated or truncated:
            if terminated:
                print(f"  ⚠️ Plasma disruption at step {step+1}")
            break
    
    print("-" * 70)
    
    # Results summary
    print(f"\n✅ DEPLOYMENT RESULTS:")
    print(f"  Total control steps executed: {step+1}")
    print(f"  Total accumulated reward: {total_reward:+.2f}")
    print(f"  Max targets met simultaneously: {max_targets_met}/5 (at step {step_achieved_targets})")
    print(f"\nFinal Plasma State:")
    print(f"  Elongation: {obs[2]:.3f} (target: 1.800, error: {abs(obs[2]-1.8):.3f})")
    print(f"  Triangularity: {obs[3]:.3f} (target: 0.400, error: {abs(obs[3]-0.4):.3f})")
    print(f"  R centroid: {obs[0]:.3f} m (target: 1.650, error: {abs(obs[0]-1.65):.3f})")
    print(f"  Z centroid: {obs[1]:.3f} m (target: 0.000, error: {abs(obs[1]-0.0):.3f})")
    print(f"  Plasma current: {obs[6]:.1f} MA (target: 15.000, error: {abs(obs[6]-15.0):.2f})")
    print(f"  q95 (stability): {obs[7]:.2f} (safe if > 2.0)")
    
    # Create visualization
    print(f"\nGenerating performance visualization...")
    create_deployment_plot(history)
    print("✅ Saved deployment performance visualization")
    
    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETE")
    print("="*70)
    
    return 0


def create_deployment_plot(history):
    """Create visualization of deployment performance."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Plasma Control Deployment - RL Agent with Corrected Reward System", 
                 fontsize=14, fontweight='bold')
    
    time = np.array(history['time'])
    
    # Elongation
    axes[0, 0].plot(time, history['elongation'], 'b-', linewidth=2, label='Actual')
    axes[0, 0].axhline(1.8, color='r', linestyle='--', linewidth=2, label='Target')
    axes[0, 0].fill_between(time, 1.7, 1.9, alpha=0.2, color='g', label='Good range')
    axes[0, 0].set_ylabel('Elongation (κ)', fontweight='bold')
    axes[0, 0].set_ylim([1.5, 2.1])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title('Shape Control: Elongation')
    
    # Triangularity
    axes[0, 1].plot(time, history['triangularity'], 'g-', linewidth=2, label='Actual')
    axes[0, 1].axhline(0.4, color='r', linestyle='--', linewidth=2, label='Target')
    axes[0, 1].fill_between(time, 0.3, 0.5, alpha=0.2, color='g', label='Good range')
    axes[0, 1].set_ylabel('Triangularity (δ)', fontweight='bold')
    axes[0, 1].set_ylim([0.0, 0.7])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_title('Shape Control: Triangularity')
    
    # Plasma current
    axes[0, 2].plot(time, history['Ip'], 'm-', linewidth=2, label='Actual')
    axes[0, 2].axhline(15.0, color='r', linestyle='--', linewidth=2, label='Target')
    axes[0, 2].fill_between(time, 14.0, 16.0, alpha=0.2, color='g', label='Good range')
    axes[0, 2].set_ylabel('Plasma Current (MA)', fontweight='bold')
    axes[0, 2].set_ylim([10.0, 20.0])
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_title('Current Control')
    
    # R centroid
    axes[1, 0].plot(time, history['R_centroid'], 'c-', linewidth=2, label='Actual')
    axes[1, 0].axhline(1.65, color='r', linestyle='--', linewidth=2, label='Target')
    axes[1, 0].set_ylabel('R Centroid (m)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_title('Position Control: R')
    axes[1, 0].set_xlabel('Time Step', fontweight='bold')
    
    # Coil currents
    axes[1, 1].plot(time, history['coil_1'], label='Coil 1', linewidth=1.5, alpha=0.8)
    axes[1, 1].plot(time, history['coil_2'], label='Coil 2', linewidth=1.5, alpha=0.8)
    axes[1, 1].plot(time, history['coil_3'], label='Coil 3', linewidth=1.5, alpha=0.8)
    axes[1, 1].plot(time, history['coil_4'], label='Coil 4', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_ylabel('Coil Current (kA)', fontweight='bold')
    axes[1, 1].set_xlabel('Time Step', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].set_title('Control Inputs: Coil Currents')
    axes[1, 1].set_ylim([4.0, 16.0])
    
    # Reward and targets
    ax_reward = axes[1, 2]
    ax_targets = ax_reward.twinx()
    
    ax_reward.bar(time, history['reward'], color='steelblue', alpha=0.7, label='Step Reward')
    ax_targets.plot(time, history['targets_met'], 'r*-', markersize=8, linewidth=1.5, label='Targets Met')
    ax_targets.set_ylim([-0.5, 5.5])
    ax_targets.set_yticks([0, 1, 2, 3, 4, 5])
    
    ax_reward.set_ylabel('Reward per Step', fontweight='bold')
    ax_reward.set_xlabel('Time Step', fontweight='bold')
    ax_targets.set_ylabel('# of Targets Met', fontweight='bold', color='r')
    ax_reward.grid(True, alpha=0.3, axis='y')
    ax_reward.set_title('Control Performance Metrics')
    
    plt.tight_layout()
    plt.savefig('plasma_deployment_results.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    sys.exit(deploy_trained_model())
