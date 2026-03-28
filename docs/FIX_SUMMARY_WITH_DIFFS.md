# Plasma Control RL - Fix Summary with Diffs

**Date**: March 27, 2026  
**Issue**: PPO-based plasma control stuck with constant reward (-876.368124) and no learning  
**Root Causes**: 
1. Action space clip at lower bound saturation (PPO init near 0)
2. Control penalty magnitude drowning reward signal
3. No per-component logging for invisible imbalance
4. Fixed-point environment (only depends on current action)

---

## File 1: plasma_control_env.py

### Change 1A: Normalize Action Space to [-1, 1]

**Before:**
```python
# Define action space: 4 coil currents [5-15 kA]
self.action_space = spaces.Box(
    low=np.array([5.0, 5.0, 5.0, 5.0]),    # Minimum coil currents
    high=np.array([15.0, 15.0, 15.0, 15.0]), # Maximum coil currents
    dtype=np.float32
)
```

**After:**
```python
# Define action space: 4 coil currents normalized to [-1, 1]
# These will be rescaled to [5-15 kA] in step()
self.action_space = spaces.Box(
    low=np.array([-1.0, -1.0, -1.0, -1.0]),  # Normalized minimum
    high=np.array([1.0, 1.0, 1.0, 1.0]),    # Normalized maximum
    dtype=np.float32
)

# Action rescaling parameters
self.action_low = 5.0   # kA
self.action_high = 15.0  # kA
```

**Why**: PPO initializes policy outputs near 0, which mapped directly to action=5 (clipped). Now 0 maps to center of range (10 kA), allowing proper exploration.

---

### Change 1B: Rescale Normalized Actions in step()

**Before:**
```python
def step(self, action):
    """
    Execute one step in the environment.
    
    Args:
        action: Array of 4 coil currents [kA]
        
    Returns:
        observation: New plasma state
        ...
    """
    # Clip action to valid range
    action = np.clip(action, 5.0, 15.0)
    self.current_coils = action
    
    # Use surrogate model to predict new plasma state
    plasma_responses = self.surrogate.predict(action)
```

**After:**
```python
def step(self, action):
    """
    Execute one step in the environment.
    
    Args:
        action: Array of 4 normalized coil currents [-1, 1]
        
    Returns:
        observation: New plasma state
        ...
    """
    # Normalize action from [-1, 1] to [5, 15] kA
    action = np.clip(action, -1.0, 1.0)
    action_rescaled = self.action_low + (action + 1.0) / 2.0 * (self.action_high - self.action_low)
    self.current_coils = action_rescaled
    
    # Use surrogate model to predict new plasma state
    plasma_responses = self.surrogate.predict(action_rescaled)
```

**Why**: Applies the rescaling formula: `actual_action = 5 + (normalized + 1) / 2 * 10`
- normalized=-1 → actual=5 kA
- normalized=0 → actual=10 kA (center, good starting point)
- normalized=+1 → actual=15 kA

---

### Change 1C: Add Per-Component Reward Logging

**Before:**
```python
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
    
    shape_reward = 10.0 * (2.0 - elongation_error - triangularity_error)
    reward += shape_reward
    
    # ... (more components)
    
    # 5. Control efficiency
    control_penalty = 0.1 * np.sum((action - 10.0)**2)
    reward -= control_penalty
    
    # 6. Bonus for meeting all targets simultaneously
    if (elongation_error < 0.1 and ... ):
        reward += 20.0  # Big bonus
            
    return reward
```

**After:**
```python
def _calculate_reward(self, plasma_responses, action):
    """
    Calculate reward based on plasma performance with per-component logging.
    
    Reward components:
    1. Shape control (elongation, triangularity)
    2. Position control (R, Z centroids)
    3. Performance (current, temperature)
    4. Stability (q95, avoid disruptions)
    5. Control efficiency (penalize extreme coil currents)
    6. Success bonus
    """
    reward_components = {}
    reward = 0.0
    
    # 1. Shape control rewards (primary objective)
    elongation_error = abs(plasma_responses['elongation'] - self.target_elongation)
    triangularity_error = abs(plasma_responses['triangularity'] - self.target_triangularity)
    
    shape_reward = 10.0 * (2.0 - elongation_error - triangularity_error)
    reward_components['shape'] = shape_reward
    reward += shape_reward
    
    # 2. Position control rewards  
    R_error = abs(plasma_responses['R_centroid'] - self.target_R_centroid)
    Z_error = abs(plasma_responses['Z_centroid'] - self.target_Z_centroid)
    
    position_reward = 5.0 * (1.0 - R_error - 2.0 * Z_error)
    reward_components['position'] = position_reward
    reward += position_reward
    
    # 3. Performance rewards
    Ip_error = abs(plasma_responses['Ip'] - self.target_Ip) / self.target_Ip
    performance_reward = 5.0 * (1.0 - Ip_error)
    reward_components['current'] = performance_reward
    reward += performance_reward
    
    # 4. Stability rewards
    q95 = plasma_responses['q95']
    if q95 > 2.0:
        stability_reward = 2.0
    else:
        stability_reward = -10.0 * (2.0 - q95)
    reward_components['stability'] = stability_reward
    reward += stability_reward
    
    # 5. Control efficiency - REDUCED from -0.1 to -0.01
    control_penalty = -0.01 * np.sum((action - 10.0)**2)
    reward_components['control'] = control_penalty
    reward += control_penalty
    
    # 6. Success bonus for meeting all targets simultaneously
    success_reward = 0.0
    if (elongation_error < 0.1 and triangularity_error < 0.05 and 
        R_error < 0.02 and Z_error < 0.02 and Ip_error < 0.05):
        success_reward = 50.0  # INCREASED from 20.0 to 50.0
    reward_components['success'] = success_reward
    reward += success_reward
            
    return reward, reward_components
```

**Why**: 
- Separates reward components into dict for logging
- Reduces control penalty from `-0.1 * sum(...)` to `-0.01 * sum(...)` (10x smaller)
- Increases success bonus from +20 to +50 for stronger incentive
- Returns tuple: `(total_reward, reward_components_dict)`

---

### Change 1D: Update Info Dict with Per-Component Rewards

**Before:**
```python
# Additional info for debugging
info = {
    'coil_currents': action.copy(),
    'plasma_responses': plasma_responses.copy(),
    'step': self.current_step,
    'targets_met': self._check_targets(plasma_responses)
}
```

**After:**
```python
# Use surrogate model to predict new plasma state
plasma_responses = self.surrogate.predict(action_rescaled)

# Convert to observation
self.state = self._responses_to_observation(plasma_responses)

# Calculate reward with per-component tracking
reward, reward_components = self._calculate_reward(plasma_responses, action_rescaled)

# ... (episode termination checks)

# Additional info for debugging with per-component rewards
info = {
    'coil_currents': action_rescaled.copy(),
    'plasma_responses': plasma_responses.copy(),
    'step': self.current_step,
    'targets_met': self._check_targets(plasma_responses),
    'reward_shape': reward_components['shape'],
    'reward_position': reward_components['position'],
    'reward_current': reward_components['current'],
    'reward_stability': reward_components['stability'],
    'reward_control': reward_components['control'],
    'reward_success': reward_components['success']
}
```

**Why**: Logs each reward component separately in the info dict so callbacks can track imbalance.

---

## File 2: simple_plasma_training.py

### Change 2A: Add ActionLoggingCallback Class

**Before:** (No custom callback)

**After:**
```python
class ActionLoggingCallback(BaseCallback):
    """Custom callback to log action statistics and reward components during training."""
    
    def __init__(self, log_dir, n_eval_episodes=5):
        super(ActionLoggingCallback, self).__init__()
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "action_logging.txt"
        self.n_eval_episodes = n_eval_episodes
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        """Called after every environment step."""
        return True
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        print(f"Training logging to: {self.log_file}")
    
    def _on_rollout_end(self) -> None:
        """Called after each rollout."""
        if self.num_timesteps - self.last_log_step >= 5000:  # Log every 5k steps
            # ... (logs action mean/std, reward components to file)
```

**Why**: Provides visibility into action diversity and reward component breakdown at regular intervals.

---

### Change 2B: Update PPO Hyperparameters

**Before:**
```python
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,         # Learning rate
    n_steps=1024,               # Steps to collect before update  
    batch_size=64,              # Batch size for training
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,              # Entropy coefficient (exploration)
    vf_coef=0.5,
    verbose=1,
    tensorboard_log=str(log_dir / "tensorboard")
)
```

**After:**
```python
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,         # REDUCED 3x for stability
    n_steps=2048,               # DOUBLED for better advantages
    batch_size=256,             # INCREASED 4x for stability
    n_epochs=10,                # (unchanged)
    gamma=0.99,                 # (unchanged)
    gae_lambda=0.95,            # (unchanged)
    clip_range=0.2,             # (unchanged)
    ent_coef=0.05,              # INCREASED 5x for exploration
    vf_coef=0.5,                # (unchanged)
    verbose=1,                  # (unchanged)
    tensorboard_log=str(log_dir / "tensorboard")
)
```

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| learning_rate | 3e-4 | 1e-4 | Prevent instability with 4x larger batch |
| n_steps | 1024 | 2048 | Better advantage estimation |
| batch_size | 64 | 256 | More stable gradient estimates |
| ent_coef | 0.01 | 0.05 | 5x more entropy for exploration |

---

### Change 2C: Increase Training Duration

**Before:**
```python
model.learn(
    total_timesteps=20000,      # Total training steps
    callback=eval_callback,
    tb_log_name="plasma_ppo"
)
```

**After:**
```python
model.learn(
    total_timesteps=100_000,    # INCREASED 5x
    callback=[eval_callback, action_callback],  # ADDED logging
    tb_log_name="plasma_ppo"
)
```

**Why**: 
- 20k was barely enough for 20 updates with PPO
- 100k gives 50+ updates, more opportunity to learn
- Callback list allows multiple callbacks

---

## File 3: normalized_plasma_env.py (NEW)

**Purpose**: Normalize observations to [-1, 1] range using hardcoded statistics

**Key Features**:
```python
class NormalizedPlasmaEnv(gym.Wrapper):
    """Observation normalization wrapper"""
    
    # Hardcoded mean estimates
    obs_mean = np.array([1.65, 0.0, 1.8, 0.4, 15.0, 5.0, 15.0, 3.0])
    
    # Hardcoded std estimates
    obs_std = np.array([0.15, 0.1, 0.4, 0.2, 5.0, 1.5, 3.0, 1.0])
    
    def _normalize_obs(self, obs):
        return ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
```

**Usage**:
```python
from normalized_plasma_env import create_normalized_env

env = create_normalized_env(max_steps=50)
```

---

## File 4: train_sac.py (NEW)

**Purpose**: SAC training variant for continuous control

**Key Differences from PPO**:
1. Uses replay buffer (remembers past experiences)
2. Automatic entropy coefficient tuning
3. State-dependent exploration (SDE)
4. Typically faster convergence for continuous control

**Hyperparameters**:
```python
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,      # Standard for SAC
    buffer_size=1_000_000,   # Large replay buffer
    batch_size=256,          # Same as PPO
    tau=0.005,               # Target network update coefficient
    ent_coef='auto',         # Automatic tuning
    use_sde=True,            # State-dependent exploration
    sde_sample_freq=4,       # Update exploration every 4 steps
)
```

**Run SAC Training**:
```bash
python train_sac.py train
python train_sac.py test <model_path>
```

---

## Summary of Changes

### Quantitative Impact

| Aspect | Change | Magnitude |
|--------|--------|-----------|
| Control Penalty | -0.1 → -0.01 | 10× reduction |
| Success Bonus | +20 → +50 | 2.5× increase |
| Entropy Coef | 0.01 → 0.05 | 5× increase |
| Learning Rate | 3e-4 → 1e-4 | 3× decrease |
| Batch Size | 64 → 256 | 4× increase |
| n_steps | 1024 → 2048 | 2× increase |
| Total Steps | 20k → 100k | 5× increase |

### Key Fixes

1. ✅ **Action space normalization**: Prevents PPO init saturation
2. ✅ **Reduced control penalty**: Lets reward signal through
3. ✅ **Per-component logging**: Visibility into imbalance
4. ✅ **Increased success bonus**: Strong target incentive
5. ✅ **Better hyperparameters**: Stability + exploration
6. ✅ **5x training budget**: More optimization iterations
7. ✅ **SAC variant**: Alternative continuous control algorithm

---

## Expected Results After Fixes

### Within First 10k Steps
- Reward changes from exactly -876.37 to varying values
- Episode lengths no longer always 50
- Action std > 0.2

### After 50k Steps
- Reward improved by 3-5x (e.g., -876 → -200)
- Episode lengths 20-40 steps (down from 50)
- Clear upward trend visible

### After 100k Steps
- Reward improved by 10x+ with some success episodes reaching -50 or better
- Episode lengths 15-30 steps
- Action logging shows meaningful component contributions
- SAC model likely converged 20-30% faster

---

## Validation

Use `VALIDATION_CHECKLIST.md` to verify all fixes are working correctly.

Key metrics to track:
1. Monitor logs show increasing reward trend
2. Episode lengths vary and decrease during training
3. Action statistics show non-zero std dev
4. Reward components tracked separately
5. Success bonus triggered occasionally

---

**All patches applied successfully! ✅**
