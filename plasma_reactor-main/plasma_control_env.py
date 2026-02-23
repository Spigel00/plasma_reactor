#!/usr/bin/env python3
"""
Plasma Control Gymnasium Environment

This environment wraps our linear surrogate model to create a Gym environment
for training RL agents to control tokamak plasma.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
from pathlib import Path

# Add the linear_surrogate directory to Python path
sys.path.append(str(Path(__file__).parent / "linear_surrogate"))
from linear_plasma_surrogate import LinearPlasmaSurrogate


class PlasmaControlEnv(gym.Env):
    """
    Gymnasium environment for plasma control using linear surrogate model.
    
    Action Space: 4 coil currents [kA] (continuous control)
    Observation Space: 8 plasma observables (position, shape, current, etc.)
    
    Goal: Learn to control plasma shape and position while maintaining stability
    """
    
    def __init__(self, max_steps=100, target_elongation=1.8, target_triangularity=0.4):
        """
        Initialize plasma control environment.
        
        Args:
            max_steps: Maximum steps per episode
            target_elongation: Desired plasma elongation (κ)
            target_triangularity: Desired plasma triangularity (δ)
        """
        super(PlasmaControlEnv, self).__init__()
        
        # Load our trained surrogate model
        model_path = Path(__file__).parent / "linear_surrogate" / "linear_surrogate_model.pkl"
        self.surrogate = LinearPlasmaSurrogate(str(model_path))
        
        # Episode parameters
        self.max_steps = max_steps
        self.current_step = 0
        
        # Control targets (what we want the plasma to achieve)
        self.target_elongation = target_elongation
        self.target_triangularity = target_triangularity
        self.target_R_centroid = 1.65  # meters
        self.target_Z_centroid = 0.0   # meters (centered)
        self.target_Ip = 15.0          # MA
        
        # Define action space: 4 coil currents [5-15 kA]
        self.action_space = spaces.Box(
            low=np.array([5.0, 5.0, 5.0, 5.0]),    # Minimum coil currents
            high=np.array([15.0, 15.0, 15.0, 15.0]), # Maximum coil currents
            dtype=np.float32
        )
        
        # Define observation space: 8 plasma observables
        # ['R_centroid', 'Z_centroid', 'elongation', 'triangularity', 'Te_avg', 'ne_avg', 'Ip', 'q95']
        obs_low = np.array([1.4, -0.5, 1.0, 0.0, 5.0, 2.0, 10.0, 2.0])   # Reasonable minimums
        obs_high = np.array([1.9, 0.5, 3.0, 1.0, 25.0, 8.0, 20.0, 6.0])  # Reasonable maximums
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high, 
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset episode counter
        self.current_step = 0
        
        # Start with baseline coil currents (with small random perturbation)
        if seed is not None:
            np.random.seed(seed)
            
        initial_coils = np.array([10.0, 8.0, 12.0, 6.0]) + np.random.normal(0, 0.5, 4)
        initial_coils = np.clip(initial_coils, 5.0, 15.0)  # Keep within bounds
        
        # Get initial plasma state from surrogate
        plasma_responses = self.surrogate.predict(initial_coils)
        
        # Convert to observation vector
        self.state = self._responses_to_observation(plasma_responses)
        self.current_coils = initial_coils
        
        return self.state.astype(np.float32), {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Array of 4 coil currents [kA]
            
        Returns:
            observation: New plasma state
            reward: Reward for this action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, 5.0, 15.0)
        self.current_coils = action
        
        # Use surrogate model to predict new plasma state
        plasma_responses = self.surrogate.predict(action)
        
        # Convert to observation
        self.state = self._responses_to_observation(plasma_responses)
        
        # Calculate reward
        reward = self._calculate_reward(plasma_responses, action)
        
        # Check if episode is done
        self.current_step += 1
        terminated = self._is_terminated(plasma_responses)
        truncated = self.current_step >= self.max_steps
        
        # Additional info for debugging
        info = {
            'coil_currents': action.copy(),
            'plasma_responses': plasma_responses.copy(),
            'step': self.current_step,
            'targets_met': self._check_targets(plasma_responses)
        }
        
        return self.state.astype(np.float32), reward, terminated, truncated, info
    
    def _responses_to_observation(self, responses):
        """Convert surrogate model responses to observation vector."""
        # Order must match observation_space definition
        obs = np.array([
            responses['R_centroid'],
            responses['Z_centroid'], 
            responses['elongation'],
            responses['triangularity'],
            responses['Te_avg'],
            responses['ne_avg'],
            responses['Ip'],
            responses['q95']
        ])
        return obs
        
    def _calculate_reward(self, plasma_responses, action):
        """
        Calculate reward based on plasma performance using proper reward shaping.
        
        Reward components (all normalized to contribution range):
        1. Shape control (elongation, triangularity) - Primary objective
        2. Position control (R, Z centroids) - Secondary objective
        3. Plasma current control - Tertiary objective
        4. Stability (q95) - Safety constraint
        5. Control smoothness - Efficiency objective
        6. Cumulative target bonuses - Achievement rewards
        
        All rewards are bounded to prevent training instability.
        """
        reward = 0.0
        
        # ===== 1. SHAPE CONTROL REWARD (Primary: -3 to +3) =====
        # Elongation: target 1.8, acceptable range [1.5, 2.1]
        elongation_error = abs(plasma_responses['elongation'] - self.target_elongation)
        elongation_normalized = min(elongation_error / 1.0, 1.0)  # Normalize to 0-1
        shape_elongation_reward = 3.0 * (1.0 - elongation_normalized)
        
        # Triangularity: target 0.4, acceptable range [0.2, 0.6]
        triangularity_error = abs(plasma_responses['triangularity'] - self.target_triangularity)
        triangularity_normalized = min(triangularity_error / 0.5, 1.0)  # Normalize to 0-1
        shape_triangularity_reward = 3.0 * (1.0 - triangularity_normalized)
        
        shape_reward = (shape_elongation_reward + shape_triangularity_reward) / 2.0
        reward += shape_reward
        
        # ===== 2. POSITION CONTROL REWARD (Secondary: -2 to +2) =====
        # R centroid: target 1.65 m, acceptable range [1.6, 1.7]
        R_error = abs(plasma_responses['R_centroid'] - self.target_R_centroid)
        R_normalized = min(R_error / 0.15, 1.0)
        position_R_reward = 2.0 * (1.0 - R_normalized)
        
        # Z centroid: target 0.0 m, acceptable range [-0.1, 0.1]
        Z_error = abs(plasma_responses['Z_centroid'] - self.target_Z_centroid)
        Z_normalized = min(Z_error / 0.2, 1.0)
        position_Z_reward = 2.0 * (1.0 - Z_normalized)
        
        position_reward = (position_R_reward + position_Z_reward) / 2.0
        reward += position_reward
        
        # ===== 3. PLASMA CURRENT CONTROL (Tertiary: -1 to +1) =====
        # Plasma current: target 15 MA, acceptable range [12, 18]
        Ip_error = abs(plasma_responses['Ip'] - self.target_Ip)
        Ip_normalized = min(Ip_error / 6.0, 1.0)
        current_reward = 1.0 * (1.0 - Ip_normalized)
        reward += current_reward
        
        # ===== 4. STABILITY REWARD (Safety: -5 to +1) =====
        # q95 should be > 2 for safe operation
        q95 = plasma_responses['q95']
        if q95 > 2.5:
            stability_reward = 1.0  # Excellent stability
        elif q95 > 2.0:
            stability_reward = 0.5 * (q95 - 1.5)  # Good stability, scaled
        elif q95 > 1.5:
            stability_reward = -2.0 * (2.0 - q95)  # Degraded but recoverable
        else:
            stability_reward = -5.0  # Critical instability - strong penalty
        
        reward += stability_reward
        
        # ===== 5. CONTROL SMOOTHNESS (Efficiency: -0.5 to 0) =====
        # Penalize extreme coil currents, but gently
        control_deviation = np.sum(np.abs(action - 10.0)) / 4.0  # Average deviation from 10 kA
        control_penalty = 0.5 * min(control_deviation / 5.0, 1.0)  # Normalized penalty
        reward -= control_penalty
        
        # ===== 6. CUMULATIVE BONUSES (Achievement rewards) =====
        # Check individual target metrics
        elongation_met = elongation_error < 0.1
        triangularity_met = triangularity_error < 0.05
        R_met = R_error < 0.05
        Z_met = Z_error < 0.05  
        Ip_met = Ip_error < 1.5
        q95_met = q95 > 2.0
        
        targets_met_count = sum([elongation_met, triangularity_met, R_met, Z_met, Ip_met, q95_met])
        
        # Bonus for meeting 3+ targets (encourages multi-objective optimization)
        if targets_met_count >= 3:
            reward += 1.0 * (targets_met_count / 6.0)  # Up to +1 bonus
        
        # Major bonus for meeting all targets (equilibrium bonus)
        if targets_met_count == 6:
            reward += 5.0
        
        # Ensure reward stays within reasonable bounds for training stability
        reward = np.clip(reward, -10.0, 20.0)
        
        return float(reward)
    
    def _is_terminated(self, plasma_responses):
        """Check if episode should terminate (plasma disruption)."""
        # Terminate if plasma goes outside safe operating limits
        
        # Safety limits
        if plasma_responses['elongation'] > 2.5:  # Too elongated
            return True
        if plasma_responses['elongation'] < 1.0:  # Not elongated enough  
            return True
        if abs(plasma_responses['Z_centroid']) > 0.3:  # Too far vertically
            return True
        if plasma_responses['q95'] < 1.5:  # MHD unstable
            return True
        if plasma_responses['Ip'] < 5.0 or plasma_responses['Ip'] > 25.0:  # Current limits
            return True
            
        return False
    
    def _check_targets(self, plasma_responses):
        """Check which control targets are being met."""
        targets_met = {
            'elongation': abs(plasma_responses['elongation'] - self.target_elongation) < 0.1,
            'triangularity': abs(plasma_responses['triangularity'] - self.target_triangularity) < 0.05,
            'R_centroid': abs(plasma_responses['R_centroid'] - self.target_R_centroid) < 0.02,
            'Z_centroid': abs(plasma_responses['Z_centroid'] - self.target_Z_centroid) < 0.02,
            'Ip': abs(plasma_responses['Ip'] - self.target_Ip) / self.target_Ip < 0.05
        }
        return targets_met
    
    def render(self, mode='human'):
        """Render current plasma state (optional for visualization)."""
        if mode == 'human':
            print(f"Step {self.current_step}:")
            print(f"  Coil currents: {self.current_coils}")
            print(f"  Elongation: {self.state[2]:.3f} (target: {self.target_elongation})")
            print(f"  Triangularity: {self.state[3]:.3f} (target: {self.target_triangularity})")
            print(f"  R centroid: {self.state[0]:.3f} m (target: {self.target_R_centroid})")
            print(f"  Plasma current: {self.state[6]:.1f} MA (target: {self.target_Ip})")
            print()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Plasma Control Environment")
    print("=" * 40)
    
    # Create environment
    env = PlasmaControlEnv()
    
    # Test random policy
    obs, info = env.reset()
    print("Initial observation:", obs)
    
    total_reward = 0
    for step in range(5):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}:")
        print(f"  Action (coil currents): {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Targets met: {info['targets_met']}")
        print(f"  Terminated: {terminated}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Environment test completed!")