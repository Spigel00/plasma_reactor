#!/usr/bin/env python3
"""
Plasma Control RL Deployment Interface

Simple interface to load and use trained RL models for plasma control.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from stable_baselines3 import PPO
from plasma_control_env import PlasmaControlEnv


class PlasmaControlDeployment:
    """Deployment interface for trained plasma control RL models."""
    
    def __init__(self, model_path="./rl_models/final_plasma_model.zip"):
        """
        Initialize deployment interface.
        
        Args:
            model_path: Path to trained RL model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.env = None
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load trained RL model."""
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        print(f"Loading trained model from: {self.model_path}")
        self.model = PPO.load(self.model_path)
        
        # Create environment for testing
        self.env = PlasmaControlEnv(max_steps=100)
        
        print("✅ Model loaded successfully!")
        
    def predict_optimal_control(self, plasma_state):
        """
        Predict optimal coil currents for given plasma state.
        
        Args:
            plasma_state: Array of 8 plasma observables
            
        Returns:
            Array of 4 optimal coil currents [kA]
        """
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Ensure proper format
        plasma_state = np.array(plasma_state, dtype=np.float32)
        
        # Use trained model to predict action
        action, _states = self.model.predict(plasma_state, deterministic=True)
        
        return action
    
    def run_control_simulation(self, n_steps=30, target_config=None):
        """
        Run complete control simulation with trained agent.
        
        Args:
            n_steps: Number of control steps to simulate
            target_config: Optional dict with target values
            
        Returns:
            Dictionary with simulation results
        """
        
        print(f"🎮 Running Plasma Control Simulation")
        print("=" * 40)
        
        # Update targets if specified
        if target_config:
            for key, value in target_config.items():
                if hasattr(self.env, f"target_{key}"):
                    setattr(self.env, f"target_{key}", value)
                    print(f"Updated target_{key} = {value}")
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Store simulation history
        history = {
            'steps': [],
            'observations': [],
            'actions': [],
            'rewards': [],
            'targets_met': [],
            'plasma_states': []
        }
        
        print(f"\nInitial Plasma State:")
        print(f"  Elongation: {obs[2]:.3f} (target: {self.env.target_elongation})")
        print(f"  Triangularity: {obs[3]:.3f} (target: {self.env.target_triangularity})")  
        print(f"  R centroid: {obs[0]:.3f} m (target: {self.env.target_R_centroid})")
        print(f"  Plasma current: {obs[6]:.1f} MA (target: {self.env.target_Ip})")
        print(f"\nControl Sequence:")
        
        total_reward = 0
        
        for step in range(n_steps):
            # Get AI control action
            action = self.predict_optimal_control(obs)
            
            # Apply control
            new_obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            
            # Store history
            history['steps'].append(step + 1)
            history['observations'].append(obs.copy())
            history['actions'].append(action.copy())
            history['rewards'].append(reward)
            history['targets_met'].append(step_info['targets_met'].copy())
            
            # Create plasma state summary
            plasma_state = {
                'elongation': new_obs[2],
                'triangularity': new_obs[3], 
                'R_centroid': new_obs[0],
                'Z_centroid': new_obs[1],
                'Ip': new_obs[6],
                'Te_avg': new_obs[4],
                'ne_avg': new_obs[5],
                'q95': new_obs[7]
            }
            history['plasma_states'].append(plasma_state)
            
            # Print key steps
            if step < 5 or step % 10 == 0:
                targets_met = sum(step_info['targets_met'].values())
                print(f"  Step {step + 1:2d}: Reward={reward:6.2f} | Targets: {targets_met}/5 | "
                      f"κ={new_obs[2]:.3f} δ={new_obs[3]:.3f} | "
                      f"Coils=[{action[0]:.1f},{action[1]:.1f},{action[2]:.1f},{action[3]:.1f}]")
            
            obs = new_obs
            
            if terminated:
                print(f"  ⚠️  Simulation terminated at step {step + 1} (plasma disruption)")
                break
        
        # Calculate final performance
        final_targets_met = sum(history['targets_met'][-1].values()) if history['targets_met'] else 0
        avg_targets_met = np.mean([sum(targets.values()) for targets in history['targets_met']])
        
        print(f"\n📊 Simulation Results:")
        print(f"  Total Steps: {len(history['steps'])}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final Targets Met: {final_targets_met}/5")
        print(f"  Average Targets Met: {avg_targets_met:.1f}/5")
        
        return {
            'history': history,
            'total_reward': total_reward,
            'final_targets_met': final_targets_met,
            'avg_targets_met': avg_targets_met,
            'success': final_targets_met >= 3  # Success if 3+ targets met
        }
    
    def create_control_visualization(self, simulation_results, save_path="plasma_control_results.png"):
        """Create visualization of control simulation results."""
        
        print(f"\n📈 Creating Control Visualization")
        print("=" * 40)
        
        history = simulation_results['history']
        
        if not history['steps']:
            print("No simulation data available")
            return
            
        steps = history['steps']
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('AI Plasma Control Performance', fontsize=14, fontweight='bold')
        
        # 1. Rewards over time
        axes[0, 0].plot(steps, history['rewards'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Control Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Control Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Plasma shape evolution
        elongations = [state['elongation'] for state in history['plasma_states']]
        triangularities = [state['triangularity'] for state in history['plasma_states']]
        
        axes[0, 1].plot(steps, elongations, 'r-', linewidth=2, label='Elongation κ')
        axes[0, 1].axhline(self.env.target_elongation, color='r', linestyle='--', alpha=0.5, label='Target κ')
        axes[0, 1].plot(steps, triangularities, 'g-', linewidth=2, label='Triangularity δ')
        axes[0, 1].axhline(self.env.target_triangularity, color='g', linestyle='--', alpha=0.5, label='Target δ')
        axes[0, 1].set_xlabel('Control Step')
        axes[0, 1].set_ylabel('Shape Parameters')
        axes[0, 1].set_title('Plasma Shape Control')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Position control
        R_centroids = [state['R_centroid'] for state in history['plasma_states']]
        Z_centroids = [state['Z_centroid'] for state in history['plasma_states']]
        
        axes[0, 2].plot(steps, R_centroids, 'purple', linewidth=2, label='R centroid')
        axes[0, 2].axhline(self.env.target_R_centroid, color='purple', linestyle='--', alpha=0.5, label='Target R')
        axes[0, 2].plot(steps, Z_centroids, 'orange', linewidth=2, label='Z centroid')  
        axes[0, 2].axhline(self.env.target_Z_centroid, color='orange', linestyle='--', alpha=0.5, label='Target Z')
        axes[0, 2].set_xlabel('Control Step')
        axes[0, 2].set_ylabel('Position [m]')
        axes[0, 2].set_title('Plasma Position Control')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Coil currents
        coil_currents = np.array(history['actions'])
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(4):
            axes[1, 0].plot(steps, coil_currents[:, i], colors[i], linewidth=2, label=f'Coil {i+1}')
        axes[1, 0].set_xlabel('Control Step')
        axes[1, 0].set_ylabel('Current [kA]')
        axes[1, 0].set_title('Coil Current Commands')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Plasma current and temperature
        plasma_currents = [state['Ip'] for state in history['plasma_states']]
        temperatures = [state['Te_avg'] for state in history['plasma_states']]
        
        ax_ip = axes[1, 1]
        ax_temp = ax_ip.twinx()
        
        line1 = ax_ip.plot(steps, plasma_currents, 'b-', linewidth=2, label='Plasma Current Ip')
        ax_ip.axhline(self.env.target_Ip, color='b', linestyle='--', alpha=0.5)
        
        line2 = ax_temp.plot(steps, temperatures, 'r-', linewidth=2, label='Temperature Te')
        
        ax_ip.set_xlabel('Control Step')
        ax_ip.set_ylabel('Plasma Current [MA]', color='b')
        ax_temp.set_ylabel('Temperature [keV]', color='r')
        ax_ip.set_title('Performance Parameters')
        
        # Combine legends
        lines1, labels1 = ax_ip.get_legend_handles_labels()
        lines2, labels2 = ax_temp.get_legend_handles_labels()
        ax_ip.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax_ip.grid(True, alpha=0.3)
        
        # 6. Targets met over time
        targets_met_count = [sum(targets.values()) for targets in history['targets_met']]
        axes[1, 2].plot(steps, targets_met_count, 'green', linewidth=3, marker='o', markersize=4)
        axes[1, 2].axhline(3, color='orange', linestyle='--', alpha=0.7, label='Success Threshold')
        axes[1, 2].set_xlabel('Control Step')
        axes[1, 2].set_ylabel('Targets Met (out of 5)')
        axes[1, 2].set_title('Control Success Rate')
        axes[1, 2].set_ylim(-0.5, 5.5)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Control visualization saved to: {save_path}")
        
        return save_path
    
    def interactive_control_demo(self):
        """Interactive demonstration of plasma control."""
        
        print(f"\n🎮 Interactive Plasma Control Demo")
        print("=" * 40)
        print("Choose a control scenario:")
        print("1. Standard operation (κ=1.8, δ=0.4)")
        print("2. High elongation (κ=2.2, δ=0.3)")  
        print("3. High triangularity (κ=1.6, δ=0.6)")
        print("4. Custom targets")
        
        choice = input("Enter choice (1-4): ").strip()
        
        target_configs = {
            "1": {"elongation": 1.8, "triangularity": 0.4},
            "2": {"elongation": 2.2, "triangularity": 0.3},
            "3": {"elongation": 1.6, "triangularity": 0.6},
        }
        
        if choice in target_configs:
            target_config = target_configs[choice]
            print(f"Selected scenario: {choice}")
        elif choice == "4":
            try:
                elongation = float(input("Enter target elongation (1.0-2.5): "))
                triangularity = float(input("Enter target triangularity (0.0-0.8): "))
                target_config = {"elongation": elongation, "triangularity": triangularity}
            except ValueError:
                print("Invalid input, using default targets")
                target_config = None
        else:
            print("Invalid choice, using default targets")
            target_config = None
        
        # Run simulation
        results = self.run_control_simulation(n_steps=25, target_config=target_config)
        
        # Create visualization
        self.create_control_visualization(results)
        
        return results


def main():
    """Main deployment demonstration."""
    
    print("🚀 Plasma Control RL Deployment")
    print("=" * 50)
    
    try:
        # Initialize deployment interface
        deployment = PlasmaControlDeployment()
        
        # Run demonstration
        results = deployment.run_control_simulation(n_steps=20)
        
        # Create visualization
        deployment.create_control_visualization(results)
        
        # Performance summary
        print(f"\n🎯 Deployment Summary:")
        print(f"✅ Model loaded and tested")
        print(f"✅ Control simulation completed")
        print(f"✅ Performance visualization created")
        
        if results['success']:
            print(f"🎉 SUCCESS: AI achieved good plasma control!")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Consider more training or reward tuning")
        
        print(f"\n📁 Files created:")
        print(f"  🤖 Model: ./rl_models/final_plasma_model.zip")
        print(f"  📊 Visualization: ./plasma_control_results.png")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Solution: Run training first with 'python simple_plasma_training.py'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()