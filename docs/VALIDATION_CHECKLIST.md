# Plasma Control RL - Validation Checklist

## Expected Behavior After Fixes

After applying the fixes described in the request, the training system should show the following improvements:

---

## 1. Monitor Log Files Validation

### Location
- **PPO Training**: `rl_training_logs/training_monitor.csv`
- **PPO Evaluation**: `rl_training_logs/eval_monitor.csv`
- **SAC Training**: `rl_training_logs_sac/training_monitor.csv`
- **SAC Evaluation**: `rl_training_logs_sac/eval_monitor.csv`

### What to Check

#### ✅ **Before Fix** (Expected Broken Behavior)
```
episode,ep_len_mean,ep_rew_mean
1,50.0,-876.368124
2,50.0,-876.368124
3,50.0,-876.368124
4,50.0,-876.368124
5,50.0,-876.368124
...
```
**Problem Indicators:**
- `ep_rew_mean` is **exactly constant** across all episodes
- `ep_len_mean` is always exactly **50** (maximum steps)
- No variation whatsoever in 10+ episodes

#### ✅ **After Fix** (Expected Correct Behavior)
```
episode,ep_len_mean,ep_rew_mean
1,50.0,-450.234567
2,48.5,-420.123456
3,42.0,-380.567890
4,45.5,-350.234567
5,43.0,-320.123456
...
100,35.2,-150.567890
110,32.1,-120.234567
...
```
**Success Indicators:**
- `ep_rew_mean` **increases** (becomes less negative) over time
- `ep_len_mean` **varies** and often **decreases below 50** (agent learns to solve faster)
- Strong **upward trend** (reward improving) over 100+ episodes
- Variance increases as exploration improves

---

## 2. Per-Component Reward Logging

### Location
- **PPO**: `rl_training_logs/action_logging.txt`
- **SAC**: `rl_training_logs_sac/sac_action_logging.txt`

### What to Check

#### ✅ **After 5k Steps** (First Log Entry)
Look for a section like:
```
============================================================
Timestep: 5000
============================================================

Action Statistics (normalized [-1, 1]):
  Mean: [... random non-zero values ...]
  Std:  [... non-zero std devs ...]
  Min:  [...values near -1...]
  Max:  [...values near +1...]

Reward Statistics:
  Episode Total: -250.45
  Episode Mean: -5.01
  Episode Std: 2.34

  shape           : mean=   5.234 std=   1.456
  position        : mean=  -0.567 std=   0.789
  current         : mean=   2.345 std=   1.123
  stability       : mean=  -1.234 std=   0.456
  control         : mean=  -2.100 std=   0.890
  success         : mean=   0.000 std=   0.000
```

**Success Indicators:**
- Actions are **NOT all zeros** or **NOT all clipped to bounds**
- Action **std > 0** (diversity in action selection)
- **Individual reward components** can be tracked
- `control` component is now **much smaller** (was -0.1 * avg_sum_sq before)
- `success` component appears when targets are met

#### ✅ **After 100k Steps** (Final Log Entry)
Expected improvements:
```
  shape           : mean=  15.234 std=   2.100   (↑ improved shape control)
  position        : mean=   4.456 std=   1.234   (↑ improved position control)
  current         : mean=   4.789 std=   0.987   (↑ improved current control)
  stability       : mean=   1.234 std=   0.345   (↑ better stability)
  control         : mean=  -0.023 std=   0.034   (↓ very small penalty now)
  success         : mean=   2.567 std=   5.345   (↑ many successful episodes)

Episode Total: -25.34 (← much improved from -250.45)
```

---

## 3. Action Space Normalization Validation

### What to Check in Action Logging

#### ✅ **Action Statistics Should Show:**

1. **After 10k steps** (PPO starting to explore):
   - Mean actions NOT near `[0, 0, 0, 0]` (the init problem)
   - Std > 0.2 for most dimensions

2. **After 50k steps** (PPO has strong policy):
   - Mean actions show **intentional bias** (e.g., `[0.3, -0.2, 0.5, -0.1]`)
   - Std still > 0.1 (maintains exploration)

3. **Rescaled actions in info['coil_currents']** should show:
   - Values spread across `[5, 15] kA` range
   - NOT all at 5 kA or all at 15 kA (saturation stopped)
   - Variety: see [5.2, 14.8, 7.1, 11.3] type patterns

---

## 4. Episode Length Variation

### What to Check in Monitor CSV

#### ✅ **PPO Training Monitor**
```
# Sample of rows from training_monitor.csv
r,l,t
-876.37,50,1000   # Early: max length, terrible reward
-650.23,50,2000
-580.45,48,3000   # Starting to improve, length varies
-420.56,35,4000   # ↑ Good sign: episode length decreased
-350.23,32,5000
-200.45,28,6000   # Episode ends before max_steps (learned!)
-180.34,26,7000
-125.67,31,8000
```

**Success Indicators:**
- `l` (episode length) drops significantly during training
- By 50k+ steps, many episodes should end around 20-35 steps, not 50
- This proves **agent learned to reach targets early**

---

## 5. Hyperparameter Changes Validation

### What Changed

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `learning_rate` | 3e-4 | 1e-4 | Prevent instability with larger batch sizes |
| `n_steps` | 1024 | 2048 | Better advantage estimation, smoother updates |
| `batch_size` | 64 | 256 | More stable gradient estimates |
| `ent_coef` | 0.01 | 0.05 | Increase exploration 5x |
| `total_timesteps` | 20k | 100k | More training iterations |
| Action range | [5,15] direct | [-1,1] → [5,15] | No initialization saturation |
| Control penalty | -0.1 | -0.01 | Reward signal not drowned out |
| Success bonus | 20.0 | 50.0 | Strong incentive for simultaneous targets |

### Validation in TensorBoard

Open with:
```bash
tensorboard --logdir rl_training_logs/tensorboard/
```

**Look for:**
- `rollouts/ep_rew_mean`: Should trend upward from ~-800 to ~-100 or better
- `train/policy_loss`: Should stabilize (not diverge)
- `train/value_loss`: Should decrease smoothly
- `rollouts/ep_len_mean`: Should decrease from 50 to 25-35

---

## 6. SAC Comparison Expected Results

### SAC vs PPO Differences

| Metric | PPO | SAC | Expected Trend |
|--------|-----|-----|---|
| Convergence Speed | ~50k steps | ~30k steps | SAC faster due to replay buffer |
| Final Performance | -80 to -20 | -60 to -10 | SAC slightly better |
| Episode Length | 25-35 steps | 20-30 steps | Both learn to solve quickly |
| Action Diversity | std ≈ 0.3 | std ≈ 0.4 | SAC maintains more exploration |
| Training Stability | Can oscillate | Very smooth | SAC more stable |

### SAC-Specific Checks

- `ent_coef` (action log sums) should stabilize around 0.5-1.5
- `actor_loss` should slowly decrease or stay negative
- `critic_loss` should decay smoothly
- Entropy should be positive throughout

---

## 7. Success Criteria Checklist

Complete validation when you observe:

- [ ] **Reward** shows clear upward trend (episode total reward increases)
- [ ] **Variety** in episode lengths (not all exactly 50)
- [ ] **Actions** span [-1, 1] normalized range (not clipped to zero)
- [ ] **Zero reward components** appear in logs (shape, position, current not all negative)
- [ ] **Success bonus** triggered at least once in logs (reward_success > 0)
- [ ] **Training stability** no massive spikes or crashes
- [ ] **SAC model** converges 20-30% faster than PPO
- [ ] **Monitor CSV** shows monotonic or near-monotonic reward improvement

---

## 8. Quick Diagnostic Script

Use this to quickly validate:

```python
import pandas as pd
import numpy as np

# Load monitor logs
df_ppo = pd.read_csv('rl_training_logs/training_monitor.csv')

# Check reward trend
print("PPO Reward Trend:")
print(f"  First 10 episodes mean: {df_ppo['r'].iloc[:10].mean():.2f}")
print(f"  Last 10 episodes mean: {df_ppo['r'].iloc[-10:].mean():.2f}")
print(f"  Improvement: {df_ppo['r'].iloc[-10:].mean() - df_ppo['r'].iloc[:10].mean():.2f}")

# Check length variation
print("\nPPO Episode Length:")
print(f"  Min: {df_ppo['l'].min()}")
print(f"  Max: {df_ppo['l'].max()}")
print(f"  Mean: {df_ppo['l'].mean():.1f}")
print(f"  Std: {df_ppo['l'].std():.1f}")

# Verify it's NOT stuck
if df_ppo['r'].std() < 1.0:
    print("\n❌ STUCK: Reward variance too low")
else:
    print(f"\n✅ LEARNING: Reward variance = {df_ppo['r'].std():.2f}")
```

---

## 9. Common Issues & Fixes

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Reward still constant | State space is fixed | Check `next_state = surrogate(action)` issue |
| Actions still at bounds | Normalization not applied | Verify rescaling in `step()` |
| Episode always 50 steps | Agent not solving anything | Check reward shaping balance |
| Reward more negative after fixes | Control penalty too strong | Reduce control weight further |
| SAC training crashes | Learning rate too high | Reduce to 1e-4 |
| No successful episodes | Targets too tight | Increase tolerance in `_check_targets()` |

---

## 10. Expected Timeline

| Step Count | Expected Behavior |
|-----------|---|
| 0-5k | High variance, actions being learned, rewards vary wildly |
| 5-20k | Reward trend becomes clear upward, episode lengths vary |
| 20-50k | Consistent improvement, episode lengths 25-40 |
| 50-100k | Near-optimal policy, lengths 20-30, rare success bonus +50 |

---

**Final Note:** If after 100k steps you're not seeing clear improvement and episode length variation, the environment itself may have a fixed-point issue. This would require deeper investigation of the state dynamics.
