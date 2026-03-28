# Plasma Control RL - Implementation Guide

## ✅ Changes Applied Successfully

All requested fixes have been implemented in your plasma control RL environment. This document summarizes what was done and how to use it.

---

## 📋 Files Modified

### 1. **plasma_control_env.py** (Modified)
Core environment fixes:
- ✅ Action space normalized to [-1, 1] (rescaled to [5-15] kA inside step)
- ✅ Control penalty reduced from -0.1 to -0.01 (10× smaller)
- ✅ Success bonus increased from +20 to +50
- ✅ Per-component reward logging added (shape/position/current/stability/control/success)

**Key Changes:**
```python
# Before: self.action_space = spaces.Box(low=5, high=15)
# After:  self.action_space = spaces.Box(low=-1, high=1)
#         + rescaling in step(): action_rescaled = 5 + (action + 1) / 2 * 10

# Before: control_penalty = 0.1 * sum()
# After:  control_penalty = 0.01 * sum()  # 10× weaker

# Reward now returns: (total_reward, reward_components_dict)
```

---

### 2. **simple_plasma_training.py** (Modified)
PPO training script with improved hyperparameters:
- ✅ Learning rate: 3e-4 → 1e-4 (3× reduction)
- ✅ n_steps: 1024 → 2048 (2× increase)
- ✅ batch_size: 64 → 256 (4× increase)
- ✅ ent_coef: 0.01 → 0.05 (5× increase for exploration)
- ✅ total_timesteps: 20k → 100k (5× training budget)
- ✅ ActionLoggingCallback added for per-rollout statistics
- ✅ Callback list supports multiple callbacks

**Run:**
```bash
cd /home/jiraiya_toadsage/plasma_reactor
source venv/bin/activate
python simple_plasma_training.py
```

---

### 3. **normalized_plasma_env.py** (NEW)
Observation normalization wrapper:
- Normalizes observations to [-1, 1] range using hardcoded statistics
- Improves neural network convergence
- Optional wrapper around PlasmaControlEnv

**Usage:**
```python
from normalized_plasma_env import create_normalized_env

env = create_normalized_env(max_steps=50)
obs, info = env.reset()
# obs is now normalized to ~[-1, 1] range
```

---

### 4. **train_sac.py** (NEW)
SAC (Soft Actor-Critic) training variant:
- Alternative continuous control algorithm
- Often converges faster than PPO (20-30%)
- Maintains more stable exploration
- Uses same environment and 100k timesteps for fair comparison

**Run:**
```bash
source venv/bin/activate

# Train new SAC model
python train_sac.py train

# Test existing SAC model
python train_sac.py test rl_models_sac/final_plasma_model_sac
```

---

### 5. **FIX_SUMMARY_WITH_DIFFS.md** (NEW)
Comprehensive before/after code diffs showing all changes:
- Detailed explanations of why each change was made
- Side-by-side code comparisons
- Table of hyperparameter changes
- Expected quantitative impacts

**Read this to understand every change.**

---

### 6. **VALIDATION_CHECKLIST.md** (NEW)
Complete validation guide:
- What to expect **before** fixes (constant reward)
- What to expect **after** fixes (improving reward)
- How to validate using monitor logs
- TensorBoard metrics to watch
- SAC vs PPO expected differences
- Diagnostic Python script for quick checks
- Common issues and solutions

**Read this AFTER training to verify fixes are working.**

---

## 🚀 Quick Start

### Step 1: Verify Venv is Ready
```bash
cd /home/jiraiya_toadsage/plasma_reactor
ls -la venv/  # Should show bin, lib, pyvenv.cfg, etc.
```

### Step 2: Run PPO Training (Fixed Version)
```bash
source venv/bin/activate
python simple_plasma_training.py
```

This will:
- Train for 100,000 timesteps (instead of 20k)
- Log per-rollout statistics every 5,000 steps
- Save action/reward component logs
- Evaluate every 2,000 steps
- Auto-save best model

**Expected Output:**
```
🚀 Training Plasma Control Agent
=====================================
Creating training environment...
Creating evaluation environment...
Initializing PPO agent...
Training logging to: ./rl_training_logs/action_logging.txt
Starting training...
----- Timestep ----------
| time/               |
| total_timesteps    | 2048   |
...
```

### Step 3: Monitor Training Progress
Watch the monitor logs:
```bash
# In another terminal
watch 'tail -20 rl_training_logs/training_monitor.csv'

# Or use the diagnostic script:
python -c "
import pandas as pd
df = pd.read_csv('rl_training_logs/training_monitor.csv')
print(f'Latest reward: {df[\"r\"].iloc[-1]:.2f}')
print(f'Improvement: {df[\"r\"].iloc[-1] - df[\"r\"].iloc[0]:.2f}')
"
```

### Step 4: Check Action Logs
```bash
tail -50 rl_training_logs/action_logging.txt
```

Should show:
- Non-zero action std (exploration happening)
- Reward components separated
- Success bonus appearing after 50k+ steps

### Step 5: Optional - Try SAC Training
```bash
python train_sac.py train
```

Compare results:
```bash
# Check which converges faster
tail -5 rl_training_logs/training_monitor.csv
tail -5 rl_training_logs_sac/training_monitor.csv
```

---

## 📊 Hyperparameter Summary

### PPO (simple_plasma_training.py)
| Parameter | Old | New | Purpose |
|-----------|-----|-----|---------|
| learning_rate | 3e-4 | 1e-4 | Stability with larger batches |
| n_steps | 1024 | 2048 | Better advantage estimation |
| batch_size | 64 | 256 | Stable gradients |
| ent_coef | 0.01 | 0.05 | 5× more exploration |
| total_timesteps | 20k | 100k | More training iterations |

### Environment (plasma_control_env.py)
| Parameter | Old | New | Purpose |
|-----------|-----|-----|---------|
| Action range | [5, 15] direct | [-1, 1] rescaled | No init saturation |
| Control penalty | -0.1 | -0.01 | Reward signal clarity |
| Success bonus | +20 | +50 | Stronger target incentive |

### SAC (train_sac.py)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| learning_rate | 3e-4 | Actor/critic networks |
| buffer_size | 1M | Experience replay |
| batch_size | 256 | Gradient stability |
| ent_coef | auto | Tuned automatically |
| use_sde | True | State-dependent exploration |

---

## 📈 Expected Results

### Before Fixes ❌
```
reward: -876.368124 (constant across all episodes)
episode_length: 50 (always maximum)
action std: 0 (frozen at bounds)
episode_reward_std: 0 (no variation)
```

### After Fixes (100k steps) ✅
```
reward: Improves from -600 → -100 (major improvement)
episode_length: Varies 20-40 (learns to solve faster)
action std: 0.3-0.4 (healthy exploration)
episode_reward_std: Growing (training signal present)
success_episodes: >10% by step 100k (solving targets)
```

---

## 🔍 Validation Workflow

1. **Start Training:**
   ```bash
   python simple_plasma_training.py 2>&1 | tee training.log
   ```

2. **Monitor Reward Trend (10 minutes in):**
   - Check `rl_training_logs/training_monitor.csv`
   - Reward should START varying (not constant)
   - Episode lengths should START dropping (not stuck at 50)

3. **Check Action Logs (30 minutes in):**
   - Read `rl_training_logs/action_logging.txt`
   - Look for first entry around step 5000
   - Verify: action std > 0.2, reward components separated

4. **Final Validation (after 100k steps):**
   - Use script in VALIDATION_CHECKLIST.md
   - Plot reward trend (should show clear improvement)
   - Check SAC vs PPO comparison if both trained

---

## 📂 Output Directory Structure

After training, you'll have:

```
rl_training_logs/
  ├── training_monitor.csv        # Training episode rewards (r, l, t)
  ├── eval_monitor.csv            # Eval episode rewards
  ├── action_logging.txt           # Per-rollout statistics (every 5k steps)
  └── tensorboard/
      └── plasma_ppo/              # TensorBoard logs

rl_models/
  ├── best_model.zip              # Best model so far
  └── final_plasma_model.zip       # Final trained model

rl_training_logs_sac/              # Same structure for SAC variant
rl_models_sac/
```

---

## 🛠️ Troubleshooting

### Problem: Reward still constant after 10k steps
**Solution:** Check that:
1. Action rescaling is active: `grep -n "action_rescaled" plasma_control_env.py`
2. Reward components are returned: `grep -n "return reward, reward_components" plasma_control_env.py`
3. Check first 5 episodes are NOT -876.37 exactly

### Problem: Training very slow
**Solution:**
- Reduce `total_timesteps` for testing: `100_000 → 50_000`
- Increase `eval_freq` to reduce evaluation overhead: `2000 → 5000`
- On low-end machines, SAC might be slower (larger networks)

### Problem: GPU out of memory
**Solution:**
- Reduce `batch_size`: `256 → 128`
- Reduce `buffer_size` (SAC only): `1_000_000 → 500_000`

### Problem: Still not converging after 100k steps
**Solution:**
This suggests a deeper state-space issue (the fixed-point problem mentioned in root causes). Check:
1. Is surrogate model deterministic? (Same input → same output always?)
2. Does action actually affect state? (Compare two different actions)
3. Are targets physically achievable with the surrogate?

---

## 📚 References

- **PPO Algorithm:** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **SAC Algorithm:** [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- **Stable Baselines 3:** [Documentation](https://stable-baselines3.readthedocs.io/)

---

## 🎯 Key Takeaways

### What Was Broken
- PPO initializes policy outputs near 0
- Direct [5,15] action space mapped 0→5 (saturation)
- Control penalty (-0.1 magnitude) drowned other rewards
- No visibility into reward components

### What Was Fixed
- Action space normalized to [-1, 1]
- PPO output 0 now maps to 10 kA (center)
- Control penalty reduced 10×
- Per-component logging added
- Better hyperparameters for exploration
- 5× more training iterations
- SAC variant for comparison

### How to Verify
- Monitor CSV shows VARYING, IMPROVING rewards
- Episode lengths DROP (not stay at max)
- Action logging shows healthy exploration

---

**Status: ✅ All fixes implemented and ready to test!**

Next step: Run `python simple_plasma_training.py` and monitor the output.
