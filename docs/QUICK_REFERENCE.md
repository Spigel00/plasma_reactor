# Plasma Control RL - Quick Reference Card

## Essential Commands

### Setup & Activation
```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies installed
pip list | grep -E "stable-baselines3|tensorboard|numpy"
```

### Training Models

#### PPO Training (Fixed Version)
```bash
# Full training (100k steps, ~30-60 min)
python simple_plasma_training.py

# Monitor progress in another terminal
watch 'tail -20 rl_training_logs/training_monitor.csv'

# View TensorBoard (open http://localhost:6006)
tensorboard --logdir rl_training_logs/tensorboard &
```

#### SAC Training (Alternative Algorithm)
```bash
# Train SAC variant (often faster)
python train_sac.py train

# Test SAC model
python train_sac.py test rl_models_sac/final_plasma_model_sac
```

### Monitoring & Validation

#### Quick Reward Check
```bash
# View last 20 episodes
tail -20 rl_training_logs/training_monitor.csv

# Check improvement over time
python -c "
import pandas as pd
df = pd.read_csv('rl_training_logs/training_monitor.csv')
print('Reward Statistics:')
print(f'  Initial (first 5): {df.iloc[0:5][\"r\"].mean():.2f}')
print(f'  Current (last 5): {df.iloc[-5:][\"r\"].mean():.2f}')
print(f'  Improvement: {df.iloc[-5:][\"r\"].mean() - df.iloc[0:5][\"r\"].mean():.2f}')
"
```

#### Check Action Logging
```bash
# View latest logging entry
tail -50 rl_training_logs/action_logging.txt

# Count how many log entries exist (should be ~20 for 100k steps)
grep "Timestep:" rl_training_logs/action_logging.txt | wc -l
```

#### Episode Length Analysis
```bash
# Check if agent learns to solve faster
python -c "
import pandas as pd
df = pd.read_csv('rl_training_logs/training_monitor.csv')
print('Episode Length Progress:')
print(f'  First 100 episodes: {df.iloc[0:100][\"l\"].mean():.1f}')
print(f'  Last 100 episodes: {df.iloc[-100:][\"l\"].mean():.1f}')
print(f'  Min observed: {df[\"l\"].min()}')
print(f'  Max observed: {df[\"l\"].max()}')
"
```

### Model Testing

#### Test Trained PPO Model
```bash
# Interactive testing
python -c "
from stable_baselines3 import PPO
from plasma_control_env import PlasmaControlEnv

model = PPO.load('rl_models/final_plasma_model')
env = PlasmaControlEnv(max_steps=50)

obs, _ = env.reset()
total_reward = 0
for _ in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward
    if done or trunc:
        break

print(f'Total Reward: {total_reward:.2f}')
"
```

#### Compare PPO vs SAC
```bash
# Side-by-side performance check
python -c "
import pandas as pd
ppo_df = pd.read_csv('rl_training_logs/training_monitor.csv')
sac_df = pd.read_csv('rl_training_logs_sac/training_monitor.csv')

print('PPO vs SAC Comparison:')
print(f'PPO  - Current Reward: {ppo_df[\"r\"].iloc[-1]:.2f}')
print(f'SAC  - Current Reward: {sac_df[\"r\"].iloc[-1]:.2f}')
print(f'PPO  - Mean Episode Length: {ppo_df[\"l\"].mean():.1f}')
print(f'SAC  - Mean Episode Length: {sac_df[\"l\"].mean():.1f}')
print(f'PPO  - Total Episodes: {len(ppo_df)}')
print(f'SAC  - Total Episodes: {len(sac_df)}')
"
```

---

## Expected Training Log Format

### training_monitor.csv Structure
```
r,l,t
-642.3456,50,2048         # episode_reward, episode_length, time_elapsed
-580.2341,48,4096
-520.1234,42,6144
-450.5678,35,8192
...
```

**Healthy Progression:**
- ✅ `r` increases (becomes less negative)
- ✅ `l` varies (not always 50)
- ✅ `l` generally decreases (learns to solve faster)
- ✅ No long plateaus (steady improvement)

**Unhealthy Signs:**
- ❌ `r` is constant (e.g., -876.37 for every row)
- ❌ `l` is always exactly 50
- ❌ `r` worsens over time
- ❌ Sudden crashes to extreme values

### action_logging.txt Structure
```
============================================================
Timestep: 5000
============================================================

Action Statistics (normalized [-1, 1]):
  Mean: [0.12 -0.34 0.45 -0.23]
  Std:  [0.31 0.28 0.35 0.29]
  Min:  [-1.00 -1.00 -0.98 -0.95]
  Max:  [0.99 0.98 1.00 0.97]

Reward Statistics:
  Episode Total: -245.67
  Episode Mean: -4.91
  Episode Std: 1.23

  shape           : mean=   4.231 std=   1.456
  position        : mean=  -1.234 std=   0.789
  current         : mean=   2.456 std=   1.123
  stability       : mean=  -0.567 std=   0.456
  control         : mean=  -0.012 std=   0.018
  success         : mean=   0.000 std=   0.000
```

**Healthy Indicators:**
- ✅ Action Std > 0.2 (exploration happening)
- ✅ Control component very small (< -0.05)
- ✅ Sum of other components > -10 (reward signal present)
- ✅ Success component appears later (0 early, >0 late)

---

## File Change Summary

| File | Change Type | Purpose |
|------|-------------|---------|
| `plasma_control_env.py` | Modified | Core env fixes |
| `simple_plasma_training.py` | Modified | PPO hyperparameters |
| `normalized_plasma_env.py` | New | Optional observation normalization |
| `train_sac.py` | New | SAC training variant |
| `IMPLEMENTATION_GUIDE.md` | New | This guide + full docs |
| `FIX_SUMMARY_WITH_DIFFS.md` | New | Before/after code diffs |
| `VALIDATION_CHECKLIST.md` | New | Validation procedures |

---

## Hyperparameter Reference

### Changed in plasma_control_env.py
- Action space: `[5,15]` → `[-1,1]` (rescaled in step)
- Control penalty: `-0.1` → `-0.01` (**10× weaker**)
- Success bonus: `+20` → `+50` (**2.5× stronger**)

### Changed in simple_plasma_training.py
```python
# Before → After
learning_rate=3e-4 → 1e-4    # 3× smaller
n_steps=1024 → 2048          # 2× larger
batch_size=64 → 256          # 4× larger
ent_coef=0.01 → 0.05         # 5× larger
total_timesteps=20_000 → 100_000  # 5× larger
```

### SAC train_sac.py
```python
learning_rate=3e-4           # Standard for SAC
buffer_size=1_000_000        # Large replay buffer
batch_size=256               # Same as PPO
tau=0.005                    # Target network update
ent_coef='auto'              # Automatic tuning
use_sde=True                 # State-dependent exploration
```

---

## Typical Training Timeline

| Time | Events | Indicators |
|------|--------|-----------|
| 0 min | Start training | Reward: -600 to -900 |
| 5 min | First eval | Episode length starts varying |
| 15 min | 10k steps | Reward improving, std > 0 in actions |
| 30 min | 50k steps | Reward improved 3-5×, lengths 25-40 |
| 45 min | 75k steps | Clear upward trend, success bonus >0 |
| 60 min | 100k steps | Final model saved, major improvement |

---

## Failure Modes & Solutions

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Reward still -876.37 | Action rescaling not applied | Check `action_rescaled =` in step() |
| Episode always 50 steps | Reward too weak to solve | Check control penalty is -0.01 not -0.1 |
| Train crashes after 5k steps | Learning rate too high | Reduce to 5e-5 |
| SAC buffer full warning | Normal behavior | Can ignore if training progresses |
| Action std = 0 | Entropy too weak | Increase ent_coef to 0.1 |

---

## Performance Targets

### Minimum Success Criteria (50k steps)
- ✅ Reward: -876 → -300 (65% improvement)
- ✅ Episode length: Varies, often < 40
- ✅ Action std: > 0.15 in all dimensions

### Good Performance (100k steps)
- ✅ Reward: -876 → -100 (88% improvement)
- ✅ Episode length: Often 20-30 steps
- ✅ Action std: 0.25-0.35 in all dimensions
- ✅ Success events: >10 episodes with bonus

### Excellent Performance (100k+ steps)
- ✅ Reward: -876 → -50 or better (94%+ improvement)
- ✅ Episode length: Often < 25 steps
- ✅ Consistent improvements in all metrics
- ✅ Success bonus triggered regularly
- ✅ SAC model converges faster than PPO

---

## Key Insights

### Why Action Space Normalization Works
- **Problem**: PPO policy outputs ~Normal(0, 0.5) initially
  - In [-1, 1] space: maps to [5, 10] kA ✅ Good exploration
  - In [5, 15] space: maps to [5, 7.5] kA ❌ Clipped to 5
- **Solution**: Normalize to [-1, 1], rescale in step() ✅

### Why Control Penalty Was the Problem
- Control penalty: `-0.1 * (10-5)² = -2.5` per step
- Other rewards: +20 to -5 (small signals)
- **Result**: Penalty drowns everything
- **Fix**: `-0.01 * (10-5)² = -0.25` (negligible)

### Why More Entropy Helps
- Default exploration very weak (ent_coef=0.01)
- Agent gets stuck in local optimum
- **Fix**: 5× more entropy rewards policy for varied actions

---

## Debugging Python Snippets

### Plot Training Progress
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rl_training_logs/training_monitor.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Reward over time
ax1.plot(df['r'])
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward')
ax1.grid()

# Episode length over time
ax2.plot(df['l'])
ax2.set_xlabel('Episode')
ax2.set_ylabel('Length')
ax2.set_title('Episode Length')
ax2.grid()

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
```

### Extract Reward Statistics
```python
import pandas as pd

df = pd.read_csv('rl_training_logs/training_monitor.csv')

print("Reward Analysis:")
print(f"  Min: {df['r'].min():.2f}")
print(f"  Max: {df['r'].max():.2f}")
print(f"  Mean: {df['r'].mean():.2f}")
print(f"  Std: {df['r'].std():.2f}")
print(f"  Median: {df['r'].median():.2f}")

# Check convergence
recent_mean = df['r'].iloc[-50:].mean()
early_mean = df['r'].iloc[:50].mean()
print(f"  Early mean: {early_mean:.2f}")
print(f"  Recent mean: {recent_mean:.2f}")
print(f"  Improvement: {(recent_mean - early_mean):.2f} ({(recent_mean - early_mean)/abs(early_mean)*100:.1f}%)")
```

---

**All fixes ready! 🚀**  
**Next: `python simple_plasma_training.py`**
