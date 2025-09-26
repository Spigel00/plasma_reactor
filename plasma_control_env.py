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
        self.surrogate = LinearPlasmaSurrogate()
        
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
        Calculate reward based on plasma performance.
        
        Reward components:
        1. Shape control (elongation, triangularity)
        2. Position control (R, Z centroids)  
        3. Performance (current, temperature)
        4. Stability (q95, avoid disruptions)
        5. Control efficiency (penalize extreme coil currents)
        """
        reward = 0.0
        
        # 1. Shape control rewards (primary objective)
        elongation_error = abs(plasma_responses['elongation'] - self.target_elongation)
        triangularity_error = abs(plasma_responses['triangularity'] - self.target_triangularity)
        
        shape_reward = 10.0 * (2.0 - elongation_error - triangularity_error)  # Max +20 points
        reward += shape_reward
        
        # 2. Position control rewards  
        R_error = abs(plasma_responses['R_centroid'] - self.target_R_centroid)
        Z_error = abs(plasma_responses['Z_centroid'] - self.target_Z_centroid)
        
        position_reward = 5.0 * (1.0 - R_error - 2.0 * Z_error)  # Max +5 points
        reward += position_reward
        
        # 3. Performance rewards
        Ip_error = abs(plasma_responses['Ip'] - self.target_Ip) / self.target_Ip
        performance_reward = 5.0 * (1.0 - Ip_error)  # Max +5 points
        reward += performance_reward
        
        # 4. Stability rewards (q95 should be > 2 for stability)
        q95 = plasma_responses['q95']
        if q95 > 2.0:
            stability_reward = 2.0
        else:
            stability_reward = -10.0 * (2.0 - q95)  # Penalty for low q95
        reward += stability_reward
        
        # 5. Control efficiency (penalize extreme coil currents)
        control_penalty = 0.1 * np.sum((action - 10.0)**2)  # Prefer moderate currents
        reward -= control_penalty
        
        # 6. Bonus for meeting all targets simultaneously
        if (elongation_error < 0.1 and triangularity_error < 0.05 and 
            R_error < 0.02 and Z_error < 0.02 and Ip_error < 0.05):
            reward += 20.0  # Big bonus for excellent control
            
        return reward
    
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