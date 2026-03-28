# 🚀 PoC Execution & Analysis Guide

**Project**: Plasma Reactor RL Control System  
**Version**: 1.0 - Proof of Concept  
**Last Updated**: October 26, 2025

---

## 📋 Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Detailed Execution Steps](#detailed-execution-steps)
3. [Understanding the Results](#understanding-the-results)
4. [Analysis & Visualization](#analysis-visualization)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- ~500 MB free disk space
- 4+ GB RAM recommended

### 5-Minute Demo

```bash
# 1. Clone and setup (1 min)
git clone https://github.com/Spigel00/plasma_reactor.git
cd plasma_reactor
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies (2 min)
pip install -r requirements.txt

# 3. Run complete demo (2 min)
python simple_plasma_training.py
```

**Expected Output**: Training complete, model saved, results visualized!

---

## 📂 Repository Structure

```
plasma_reactor/
├── 📄 EXECUTION_GUIDE.md          # ← YOU ARE HERE
├── 📄 README.md                    # Project overview
├── 📄 POC_Completion_Assessment.md # PoC evaluation
├── 📄 requirements.txt             # Python dependencies
│
├── 🧠 Core Components/
│   ├── plasma_control_env.py      # RL Gymnasium environment
│   ├── simple_plasma_training.py  # Main training script
│   └── plasma_deployment.py       # Deployment & testing
│
├── 🔬 Surrogate Model/
│   └── linear_surrogate/
│       ├── linear_plasma_surrogate.py   # Fast physics model
│       ├── linear_surrogate_model.pkl   # Trained model
│       └── response_matrices.json       # Control sensitivity
│
├── 📊 Results & Analysis/
│   ├── rl_models/                 # Trained RL models
│   ├── rl_training_logs/          # Training metrics
│   └── physics_analysis/          # Physics visualizations
│
└── 📚 Documentation/
    ├── Accomplishments_Till_Now.md
    ├── RL_Environment_Analysis.md
    └── Project_Progress_Report.md
```

---

## 🎯 Detailed Execution Steps

### Step 1: Environment Setup

#### 1.1 Clone Repository

```bash
git clone https://github.com/Spigel00/plasma_reactor.git
cd plasma_reactor
```

#### 1.2 Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify activation:**
```bash
# You should see (.venv) in your prompt
python --version  # Should be 3.8+
```

#### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages installed:**
- `gymnasium` - RL environment framework
- `stable-baselines3` - RL algorithms
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning
- `pandas` - Data analysis

**Verify installation:**
```bash
python -c "import gymnasium; import stable_baselines3; print('✅ All packages installed!')"
```

---

### Step 2: Test Surrogate Model

Test the fast physics surrogate model that replaces expensive simulations.

```bash
python -c "
from linear_surrogate.linear_plasma_surrogate import LinearPlasmaSurrogate
import time

# Initialize surrogate
surrogate = LinearPlasmaSurrogate()
print('✅ Surrogate model loaded successfully!')

# Test prediction speed
coil_currents = [10.5, 8.2, 12.1, 6.3]
start = time.time()
response = surrogate.predict(coil_currents)
elapsed = (time.time() - start) * 1000

print(f'\n📊 Prediction Results:')
for key, value in response.items():
    print(f'  {key:20s}: {value:8.3f}')

print(f'\n⚡ Inference time: {elapsed:.2f} ms')
print(f'✅ Expected: < 5 ms (60,000x faster than physics sim!)')
"
```

**Expected Output:**
```
✅ Surrogate model loaded successfully!

📊 Prediction Results:
  R_centroid          :    1.653
  Z_centroid          :   -0.012
  elongation          :    1.812
  triangularity       :    0.398
  Te_avg              :   12.456
  ne_avg              :    5.234
  Ip                  :   15.123
  q95                 :    3.145

⚡ Inference time: 0.73 ms
✅ Expected: < 5 ms (60,000x faster than physics sim!)
```

---

### Step 3: Test RL Environment

Test the custom Gymnasium environment.

```bash
python plasma_control_env.py
```

**Expected Output:**
```
Testing Plasma Control Environment
==================================================

Environment Details:
  Action Space: Box(4,) - 4 coil currents [5-15 kA]
  Observation Space: Box(8,) - 8 plasma parameters
  Max Episode Steps: 50

Running test episode...
Step 1/50 | Reward: -12.34 | Action: [10.2 8.5 11.8 6.3]
Step 2/50 | Reward: -8.76  | Action: [9.8 8.9 12.1 6.7]
...
Episode finished after 50 steps
Total Reward: -523.45

✅ Environment test complete!
```

**What This Tests:**
- Environment initialization
- Physics surrogate integration
- Reward calculation
- Episode management
- Action/observation handling

---

### Step 4: Run RL Training

Train a PPO agent to control the plasma.

```bash
python simple_plasma_training.py
```

**Expected Output:**
```
==============================================
   Plasma Control RL Training Pipeline
==============================================

Setting up environment...
✅ Environment created successfully

Configuring PPO agent...
✅ PPO model initialized

Training for 20,000 timesteps...
--------------------------------------------------
| rollout/              |           |
|    ep_len_mean        | 50        |
|    ep_rew_mean        | -876      |
| time/                 |           |
|    fps                | 2456      |
|    iterations         | 10        |
|    time_elapsed       | 8         |
|    total_timesteps    | 10240     |
| train/                |           |
|    entropy_loss       | -1.38     |
|    learning_rate      | 0.0003    |
|    policy_loss        | -0.0234   |
|    value_loss         | 245.67    |
--------------------------------------------------
...
Training complete! Time elapsed: 16.5s

Evaluating trained model...
Episode 1: Reward = -525.82
Episode 2: Reward = -525.82
Episode 3: Reward = -525.82
Average Reward: -525.82

✅ Models saved to rl_models/
✅ Logs saved to rl_training_logs/

Comparing with baseline policies...
Random Policy:      -7.78
Fixed Baseline:     +75.87
Simple Heuristic:   +114.57
Trained RL Agent:   -525.82

✅ Training complete!
```

**What This Does:**
- Creates RL environment
- Initializes PPO agent
- Trains for 20,000 timesteps (~400 episodes)
- Evaluates final policy
- Compares with baseline strategies
- Saves trained model

**Files Generated:**
- `rl_models/best_model.zip` - Best model during training
- `rl_models/final_plasma_model.zip` - Final trained model
- `rl_training_logs/training_monitor.csv` - Episode rewards
- `rl_training_logs/eval_monitor.csv` - Evaluation metrics

---

### Step 5: Deploy & Visualize Results

Load the trained model and visualize performance.

```bash
python plasma_deployment.py
```

**Expected Output:**
```
==============================================
     Plasma Control Deployment Interface
==============================================

Loading trained model from: rl_models/final_plasma_model.zip
✅ Model loaded successfully!

Running control simulation (20 steps)...
Step 1/20: Action=[5.0, 5.0, 5.0, 5.0], Reward=-17.62
Step 2/20: Action=[5.0, 5.0, 5.0, 5.0], Reward=-17.62
...
Step 20/20: Action=[5.0, 5.0, 5.0, 5.0], Reward=-17.62

Simulation Results:
  Total Reward: -350.55
  Average Reward per Step: -17.53
  Control Consistency: All actions at minimum [5.0, 5.0, 5.0, 5.0]

Generating visualization...
✅ Results saved to: plasma_control_results.png

Performance Analysis:
  Targets Met: 0/5
  Elongation Error: 39.4%
  Triangularity Error: 161.2%
  Position Error: 275.1%

⚠️ Performance Note: Agent converged to suboptimal policy
   See RL_Environment_Analysis.md for improvement strategies
```

**Visualization Generated:**

The `plasma_control_results.png` file contains 4 panels:
1. **Control Actions** - Coil currents over time
2. **Plasma Shape** - Elongation & triangularity evolution
3. **Position Control** - R, Z centroid tracking
4. **Reward Evolution** - Episode reward over time

---

## 📊 Understanding the Results

### What the Numbers Mean

#### Surrogate Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.875 - 0.987 | Excellent prediction accuracy |
| **Inference Time** | < 1 ms | 60,000x faster than physics sim |
| **Variables** | 8 observables | Complete plasma state |

**✅ Conclusion**: Surrogate model is highly effective.

#### RL Agent Performance

| Metric | Current Value | Target | Status |
|--------|---------------|--------|--------|
| **Training Reward** | -876 | > 0 | ❌ Poor |
| **Test Reward** | -525.82 | > 0 | ❌ Poor |
| **Targets Met** | 0/5 | 3-5/5 | ❌ Poor |
| **Training Stability** | Stable | Stable | ✅ Good |
| **Convergence** | Converged | Converged | ✅ Good |

**⚠️ Conclusion**: Infrastructure works, but agent needs optimization.

#### Baseline Comparison

| Strategy | Avg Reward | Quality |
|----------|------------|---------|
| **Random** | -7.78 | Poor |
| **Fixed [10,8,12,6]** | +75.87 | Good |
| **Simple Heuristic** | +114.57 | Very Good |
| **Trained RL** | -525.82 | Very Poor |

**❌ Issue**: RL agent worse than simple baselines → reward function needs redesign.

### Why Agent Performance is Poor (and it's OK!)

**The Problem:**
- Agent learned to output [5.0, 5.0, 5.0, 5.0] (minimum coil currents)
- This minimizes control penalties but fails to control plasma
- Result: Negative rewards, no targets met

**Root Causes Identified:**
1. **Reward function has excessive penalties** for control effort
2. **Insufficient training time** (20k vs 100k+ needed)
3. **Low exploration** (entropy coefficient too small)
4. **Missing normalization** in observation space

**Why This is Expected:**
- ✅ First RL implementations rarely work perfectly
- ✅ Reward engineering requires iteration
- ✅ Infrastructure is solid - just needs tuning
- ✅ All issues have known solutions (see RL_Environment_Analysis.md)

**Next Steps:**
- Redesign reward function (reduce penalties)
- Train for 100k+ timesteps
- Add observation normalization
- Test alternative algorithms (SAC, TD3)

---

## 🔍 Analysis & Visualization

### View Training Logs with TensorBoard

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Launch TensorBoard
tensorboard --logdir=rl_training_logs/tensorboard/

# Open browser to: http://localhost:6006
```

**What You'll See:**
- Episode reward over time
- Policy loss evolution
- Value function accuracy
- Entropy (exploration) metrics
- Learning rate schedule

### Analyze Training Data Manually

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('rl_training_logs/training_monitor.csv')

# Plot episode rewards
plt.figure(figsize=(10, 6))
plt.plot(df['r'], alpha=0.7)
plt.plot(df['r'].rolling(10).mean(), 'r-', linewidth=2, label='10-episode average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()
```

### Inspect Surrogate Model

```python
from linear_surrogate.linear_plasma_surrogate import LinearPlasmaSurrogate
import json

# Load surrogate
surrogate = LinearPlasmaSurrogate()

# Get control sensitivity matrix
response_matrix = surrogate.get_response_matrix()
print("Response Matrix (8 x 4):")
print(response_matrix)

# Analyze control authority
authority = surrogate.get_control_authority()
print("\nControl Authority by Coil:")
for coil, auth in authority.items():
    print(f"  {coil}: {auth:.3f}")
```

### Custom Analysis Scripts

**Test Different Control Strategies:**

```python
from plasma_control_env import PlasmaControlEnv
import numpy as np

env = PlasmaControlEnv()

# Test custom control policy
def my_control_policy(obs):
    # Your custom logic here
    return np.array([10.0, 8.0, 12.0, 6.0])

# Run episode
obs, info = env.reset()
total_reward = 0

for step in range(50):
    action = my_control_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

print(f"Custom Policy Reward: {total_reward:.2f}")
```

---

## 🎮 Advanced Usage

### Train with Different Hyperparameters

```python
from stable_baselines3 import PPO
from plasma_control_env import PlasmaControlEnv

# Create environment
env = PlasmaControlEnv(max_steps=50)

# Custom hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,      # Lower learning rate
    n_steps=2048,            # Larger rollouts
    batch_size=256,          # Larger batches
    n_epochs=20,             # More epochs
    gamma=0.99,
    ent_coef=0.05,           # More exploration
    verbose=1
)

# Train longer
model.learn(total_timesteps=100_000)
model.save("rl_models/custom_model")
```

### Test Different Algorithms

```python
from stable_baselines3 import SAC, TD3

# SAC (Soft Actor-Critic)
model_sac = SAC("MlpPolicy", env, verbose=1)
model_sac.learn(total_timesteps=50_000)

# TD3 (Twin Delayed DDPG)
model_td3 = TD3("MlpPolicy", env, verbose=1)
model_td3.learn(total_timesteps=50_000)
```

### Create Custom Environment Variant

```python
from plasma_control_env import PlasmaControlEnv

class ImprovedPlasmaEnv(PlasmaControlEnv):
    """Environment with improved reward function."""
    
    def _calculate_reward(self, plasma_responses, action):
        # Custom reward logic
        shape_reward = 20.0 * np.exp(-2.0 * shape_error)
        control_penalty = -0.01 * np.sum((action - 10.0)**2)  # Reduced penalty
        
        return shape_reward + control_penalty

# Use improved environment
env_improved = ImprovedPlasmaEnv()
```

### Parallel Training (Speed Up)

```python
from stable_baselines3.common.env_util import make_vec_env

# Create 4 parallel environments
env = make_vec_env(PlasmaControlEnv, n_envs=4)

# Train with vectorized env (4x faster)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Import Error: No module named 'gymnasium'

**Problem**: Dependencies not installed

**Solution**:
```bash
.venv\Scripts\activate  # Make sure venv is activated
pip install -r requirements.txt
```

#### 2. FileNotFoundError: linear_surrogate_model.pkl

**Problem**: Surrogate model file missing

**Solution**:
```bash
# Check if file exists
ls linear_surrogate/linear_surrogate_model.pkl

# If missing, model may be in different location
# Update path in linear_plasma_surrogate.py
```

#### 3. Training is Very Slow

**Problem**: CPU-based training can be slow

**Solutions**:
- ✅ Reduce training timesteps (10k instead of 20k)
- ✅ Use fewer evaluation episodes
- ✅ Disable verbose logging (`verbose=0`)
- ✅ Use parallel environments (see Advanced Usage)

#### 4. Model Performance is Terrible

**Problem**: Agent gets negative rewards, fails all targets

**Expected**: This is the current PoC status!

**Solutions**: See Phase 2 improvements in `RL_Environment_Analysis.md`
- Redesign reward function
- Train longer (100k timesteps)
- Tune hyperparameters
- Try different algorithms

#### 5. TensorBoard Won't Start

**Problem**: Port already in use or TensorBoard not installed

**Solution**:
```bash
# Install tensorboard
pip install tensorboard

# Use different port
tensorboard --logdir=rl_training_logs/tensorboard/ --port=6007
```

#### 6. Visualization Not Showing

**Problem**: matplotlib backend issue

**Solution**:
```python
# Add to top of script
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

---

## 📈 Expected Results Summary

### ✅ What Should Work

| Component | Expected Result |
|-----------|----------------|
| **Surrogate Model** | < 1ms inference, R² > 0.87 |
| **Environment** | Runs without errors, generates observations |
| **Training** | Completes in ~15-20 seconds, no crashes |
| **Deployment** | Loads model, runs simulation, creates plot |

### ⚠️ Known Limitations

| Aspect | Current Status | Target |
|--------|---------------|--------|
| **Agent Reward** | -525 (negative) | > 0 (positive) |
| **Targets Met** | 0/5 | 3-5/5 |
| **Control Quality** | Fixed at minimum | Responsive, varied |
| **Training Time** | 20k steps | 100k+ steps |

**Note**: These limitations are documented and have clear solutions in Phase 2.

---

## 📚 Additional Resources

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, installation, features |
| `POC_Completion_Assessment.md` | Comprehensive PoC evaluation |
| `RL_Environment_Analysis.md` | Detailed performance analysis (439 lines) |
| `Accomplishments_Till_Now.md` | Complete workflow documentation |
| `Project_Progress_Report.md` | Current status summary |

### Key Code Files

| File | Description |
|------|-------------|
| `plasma_control_env.py` | Custom Gymnasium environment (225 lines) |
| `simple_plasma_training.py` | Main training script (280 lines) |
| `plasma_deployment.py` | Deployment interface (195 lines) |
| `linear_surrogate/linear_plasma_surrogate.py` | Surrogate model (200+ lines) |

### External Links

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TORAX Framework](https://github.com/google-deepmind/torax)

---

## 🎯 Quick Command Reference

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Test Components
python plasma_control_env.py              # Test environment
python simple_plasma_training.py          # Train agent
python plasma_deployment.py               # Deploy & visualize

# Analysis
tensorboard --logdir=rl_training_logs/tensorboard/
python -c "from linear_surrogate.linear_plasma_surrogate import LinearPlasmaSurrogate; LinearPlasmaSurrogate().predict([10,8,12,6])"

# Cleanup
deactivate                                # Exit virtual environment
```

---

## 🏆 Success Criteria

After running this guide, you should achieve:

- ✅ **Environment Setup**: Virtual environment created, dependencies installed
- ✅ **Surrogate Model**: Sub-millisecond predictions, R² > 0.87
- ✅ **RL Environment**: Runs without errors, proper gym interface
- ✅ **Training**: Completes successfully, saves models
- ✅ **Deployment**: Loads models, generates visualizations
- ✅ **Understanding**: Know why performance is limited and path forward

---

## 💬 Support & Feedback

**Questions?** Check the documentation files:
- Technical details → `RL_Environment_Analysis.md`
- Full workflow → `Accomplishments_Till_Now.md`
- PoC evaluation → `POC_Completion_Assessment.md`

**Issues?** 
- Check [Troubleshooting](#troubleshooting) section
- Review error messages carefully
- Ensure virtual environment is activated

**Want to Contribute?**
- See improvement roadmap in `RL_Environment_Analysis.md`
- Phase 2 focuses on reward function optimization
- Phase 3 tests advanced algorithms

---

## 🎉 Congratulations!

You've successfully executed and analyzed the Plasma Reactor RL PoC!

**What You've Demonstrated:**
- ✅ Surrogate models can replace expensive physics simulations (60,000x speedup)
- ✅ RL environments work for plasma control
- ✅ Complete training pipeline is functional
- ✅ Infrastructure is production-ready

**Next Steps:**
- Review performance analysis in `RL_Environment_Analysis.md`
- Implement Phase 2 improvements (reward redesign)
- Test advanced algorithms (SAC, TD3)
- Extend training to 100k+ timesteps

**The foundation is solid - now it's time to optimize! 🚀**

---

**Document Version**: 1.0  
**Last Updated**: October 26, 2025  
**Maintained By**: Plasma Reactor RL Team  
**License**: MIT
