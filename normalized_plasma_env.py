#!/usr/bin/env python3
"""
Normalized Plasma Control Environment Wrapper

Wraps PlasmaControlEnv with observation normalization using hardcoded statistics.
This helps the neural network learn faster by ensuring observations are in [-1, 1] range.
"""

import gymnasium as gym
import numpy as np
from gym import spaces
from plasma_control_env import PlasmaControlEnv


class NormalizedPlasmaEnv(gym.Wrapper):
    """
    Observation normalization wrapper for PlasmaControlEnv.
    
    Uses hardcoded mean and std estimates from typical plasma responses
    to normalize observations to ~[-1, 1] range, improving RL convergence.
    """
    
    def __init__(self, env=None, max_steps=100):
        """
        Initialize the normalization wrapper.
        
        Args:
            env: PlasmaControlEnv instance (or None to create default)
            max_steps: Maximum steps per episode
        """
        if env is None:
            env = PlasmaControlEnv(max_steps=max_steps)
        
        super(NormalizedPlasmaEnv, self).__init__(env)
        
        # Hardcoded normalization statistics (estimated from typical runs)
        # These are the means of each observable
        self.obs_mean = np.array([
            1.65,   # R_centroid (meters)
            0.0,    # Z_centroid (meters)
            1.8,    # elongation
            0.4,    # triangularity
            15.0,   # Te_avg (keV)
            5.0,    # ne_avg (1e19 m^-3)
            15.0,   # Ip (MA)
            3.0     # q95
        ], dtype=np.float32)
        
        # Hardcoded normalization statistics (estimated std devs)
        self.obs_std = np.array([
            0.15,   # R_centroid
            0.1,    # Z_centroid
            0.4,    # elongation
            0.2,    # triangularity
            5.0,    # Te_avg
            1.5,    # ne_avg
            3.0,    # Ip
            1.0     # q95
        ], dtype=np.float32)
        
        # Avoid division by zero
        self.obs_std = np.maximum(self.obs_std, 1e-8)
        
    def _normalize_obs(self, obs):
        """Normalize observation to ~[-1, 1] range."""
        return ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment and return normalized observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, info
    
    def step(self, action):
        """Take step and return normalized observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, reward, terminated, truncated, info


def create_normalized_env(max_steps=50):
    """
    Factory function to create a normalized plasma control environment.
    
    Args:
        max_steps: Maximum steps per episode
        
    Returns:
        NormalizedPlasmaEnv instance
    """
    base_env = PlasmaControlEnv(max_steps=max_steps)
    return NormalizedPlasmaEnv(env=base_env, max_steps=max_steps)


if __name__ == "__main__":
    print("Testing Normalized Plasma Control Environment")
    print("=" * 50)
    
    # Create normalized environment
    env = create_normalized_env(max_steps=30)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial normalized observation: {obs}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Test a few steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step + 1}:")
        print(f"  Normalized obs range: [{obs.min():.2f}, {obs.max():.2f}]")
        print(f"  Reward: {reward:.2f}")
        print(f"  Action (normalized): {action}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Normalization test completed!")
