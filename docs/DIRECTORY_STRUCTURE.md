# 📁 Repository Structure

**Last Updated**: October 26, 2025  
**Status**: Cleaned and Organized

---

## 🎯 Quick Navigation

```
plasma_reactor/                          # Root directory
│
├── 📘 Documentation (Start Here!)
│   ├── README.md                        # Project overview & quick start
│   ├── EXECUTION_GUIDE.md              # How to run the PoC (DETAILED)
│   ├── POC_Completion_Assessment.md    # PoC evaluation & results
│   ├── RL_Environment_Analysis.md      # Performance analysis (439 lines)
│   ├── Accomplishments_Till_Now.md     # Technical documentation
│   └── Project_Progress_Report.md      # Current status
│
├── 🚀 Core Code (Main Components)
│   ├── plasma_control_env.py           # RL Gymnasium environment
│   ├── simple_plasma_training.py       # Training pipeline
│   └── plasma_deployment.py            # Deployment & visualization
│
├── 🧠 Surrogate Model
│   └── linear_surrogate/
│       ├── linear_plasma_surrogate.py  # Fast physics model class
│       ├── linear_surrogate_model.pkl  # Trained Ridge regression
│       ├── response_matrices.json      # Control sensitivity data
│       └── response_matrix_visualization.png
│
├── 📊 Results & Artifacts
│   ├── rl_models/                      # Trained RL models
│   │   ├── best_model.zip              # Best during training
│   │   └── final_plasma_model.zip      # Final trained model
│   ├── rl_training_logs/               # Training metrics
│   │   ├── training_monitor.csv        # Episode rewards
│   │   ├── eval_monitor.csv            # Evaluation results
│   │   ├── evaluations.npz             # Numpy evaluation data
│   │   └── tensorboard/                # TensorBoard logs
│   ├── physics_analysis/               # Physics analysis outputs
│   │   ├── comprehensive_plasma_analysis.png
│   │   └── complete_physics_analysis_report.md
│   ├── rl_logs/                        # Additional RL logs
│   │   └── training_config.json
│   └── plasma_control_results.png      # Deployment demo result
│
├── 🔬 Reference Code (TORAX Integration)
│   ├── torax/                          # Original TORAX analysis
│   ├── data/                           # Parameter configurations
│   ├── models/                         # Model storage
│   └── outputs/                        # Analysis outputs
│
├── ⚙️ Configuration
│   ├── requirements.txt                # Python dependencies
│   ├── .gitignore                      # Git ignore rules
│   └── LICENSE                         # MIT License
│
└── 📦 Archive (Old/Deprecated)
    └── archive/                        # Historical documents
        ├── Phase1_Problem_Validation.md
        ├── Phase2_Ideation.md
        └── CLEANUP_PLAN.md
```

---

## 📂 Directory Details

### 📘 Documentation (6 files)

All project documentation in one place.

| File | Purpose | Size | Priority |
|------|---------|------|----------|
| `EXECUTION_GUIDE.md` | **START HERE** - Complete execution instructions | ~900 lines | 🔥 CRITICAL |
| `README.md` | Project overview, installation, features | ~350 lines | 🔥 CRITICAL |
| `POC_Completion_Assessment.md` | PoC evaluation and success criteria | ~550 lines | ⭐ Important |
| `RL_Environment_Analysis.md` | Performance analysis & improvements | ~439 lines | ⭐ Important |
| `Accomplishments_Till_Now.md` | Technical documentation | ~400 lines | ℹ️ Reference |
| `Project_Progress_Report.md` | Status summary | ~80 lines | ℹ️ Reference |

**Reading Order for New Users:**
1. `README.md` - Understand what the project does
2. `EXECUTION_GUIDE.md` - Learn how to run it
3. `POC_Completion_Assessment.md` - Understand current status
4. `RL_Environment_Analysis.md` - Dive into technical details

---

### 🚀 Core Code (3 files)

Essential Python files for the PoC.

#### `plasma_control_env.py` (225 lines)
**Purpose**: Custom Gymnasium environment for plasma control  
**Key Classes**: `PlasmaControlEnv`  
**Usage**:
```python
from plasma_control_env import PlasmaControlEnv
env = PlasmaControlEnv(max_steps=50)
obs, info = env.reset()
```

**Features**:
- Action space: 4 coil currents [5-15 kA]
- Observation space: 8 plasma parameters
- Physics-based reward function
- Disruption detection

#### `simple_plasma_training.py` (280 lines)
**Purpose**: Main RL training script  
**Key Functions**: `train_plasma_controller()`, `evaluate_policy()`, `compare_with_baseline()`  
**Usage**:
```bash
python simple_plasma_training.py
```

**Features**:
- PPO agent training
- Progress monitoring
- Model evaluation
- Baseline comparison
- Automatic model saving

#### `plasma_deployment.py` (195 lines)
**Purpose**: Load and deploy trained models  
**Key Classes**: `PlasmaControlDeployment`  
**Usage**:
```bash
python plasma_deployment.py
```

**Features**:
- Model loading
- Control simulation
- Performance visualization
- Result analysis

---

### 🧠 Surrogate Model

Fast physics model that replaces expensive simulations.

#### `linear_surrogate/linear_plasma_surrogate.py` (200+ lines)
**Purpose**: Linear surrogate model for plasma prediction  
**Key Classes**: `LinearPlasmaSurrogate`  
**Performance**: < 1ms inference, R² > 0.87

**Usage**:
```python
from linear_surrogate.linear_plasma_surrogate import LinearPlasmaSurrogate

surrogate = LinearPlasmaSurrogate()
response = surrogate.predict([10.5, 8.2, 12.1, 6.3])
```

**Methods**:
- `predict(coil_currents)` - Get plasma response
- `get_response_matrix()` - Control sensitivity
- `get_control_authority()` - Actuator effectiveness

#### Artifacts
- `linear_surrogate_model.pkl` - Trained Ridge regression models
- `response_matrices.json` - Sensitivity data (8x4 matrix)
- `response_matrix_visualization.png` - Heatmap visualization

---

### 📊 Results & Artifacts

Generated during training and analysis.

#### `rl_models/` - Trained RL Models
- `best_model.zip` - Best model during training (by evaluation reward)
- `final_plasma_model.zip` - Final trained model after 20k timesteps

**Load models**:
```python
from stable_baselines3 import PPO
model = PPO.load("rl_models/final_plasma_model")
```

#### `rl_training_logs/` - Training Metrics
- `training_monitor.csv` - Episode rewards, lengths, times
- `eval_monitor.csv` - Evaluation episode results
- `evaluations.npz` - Numpy format evaluation data
- `tensorboard/` - TensorBoard logs

**View in TensorBoard**:
```bash
tensorboard --logdir=rl_training_logs/tensorboard/
```

#### `physics_analysis/` - Physics Visualizations
- `comprehensive_plasma_analysis.png` - 15-panel analysis dashboard
- `complete_physics_analysis_report.md` - Technical report

#### Other Results
- `plasma_control_results.png` - Deployment visualization (4 panels)
- `rl_logs/training_config.json` - Training configuration

---

### 🔬 Reference Code

Original analysis code and data (kept for reference).

#### `torax/` - TORAX Integration
Complete plasma physics analysis using TORAX framework.

**Key files**:
- `torax/complete_plasma_analysis.py` - Full analysis pipeline
- `torax/generate_simple_physics.py` - Synthetic data generation
- `torax/good_runs/` - NetCDF physics data files
- `torax/plots/` - Generated visualizations

**Not needed for PoC execution** - Surrogate model already trained.

#### `data/` - Parameter Configurations
Various TORAX configuration files for different plasma scenarios.

#### `models/` - Model Storage
Alternative location for saved models.

#### `outputs/` - Analysis Outputs
Additional analysis results and plots.

---

### ⚙️ Configuration Files

#### `requirements.txt`
Python package dependencies. Install with:
```bash
pip install -r requirements.txt
```

**Key packages**:
- gymnasium==0.29.1
- stable-baselines3==2.1.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- scikit-learn>=1.3.0

#### `.gitignore`
Git ignore patterns. Excludes:
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Archive directory
- Temporary files

#### `LICENSE`
MIT License - Open source, permissive license.

---

### 📦 Archive

Old documentation and deprecated files (not needed for PoC).

- `Phase1_Problem_Validation.md` - Early problem definition
- `Phase2_Ideation.md` - Initial planning
- `CLEANUP_PLAN.md` - Repository cleanup notes

**Safe to ignore** - Historical reference only.

---

## 🎯 File Count Summary

```
Total Files (excluding archive & .git):

📘 Documentation:        6 files  (~2,700 lines)
🚀 Core Code:           3 files  (~700 lines)
🧠 Surrogate Model:     4 files  (~200 lines code + models)
📊 Results:            ~15 files (models, logs, plots)
🔬 Reference Code:    ~100+ files (TORAX integration)
⚙️ Configuration:       3 files

Essential for PoC:     ~15 files
Reference material:    ~100+ files
Total documented:      ~2,000+ lines markdown
```

---

## 🚀 Quick Start Files

**Minimum files needed to run PoC:**

1. `requirements.txt` - Install dependencies
2. `plasma_control_env.py` - RL environment
3. `simple_plasma_training.py` - Training script
4. `plasma_deployment.py` - Deployment
5. `linear_surrogate/` - Surrogate model (4 files)
6. `EXECUTION_GUIDE.md` - Instructions

**That's only ~10 files to get started!**

---

## 📖 Documentation Reading Path

### For First-Time Users:
1. `README.md` (10 min read) - Project overview
2. `EXECUTION_GUIDE.md` (30 min read + execution) - Run the PoC
3. Experiment with code
4. `POC_Completion_Assessment.md` (20 min read) - Understand results

### For Technical Deep Dive:
1. `Accomplishments_Till_Now.md` - Complete workflow
2. `RL_Environment_Analysis.md` - Performance analysis
3. Review source code in `Core Code/` directory
4. Inspect `physics_analysis/` visualizations

### For Contributors:
1. All documentation files
2. Core code + surrogate model
3. `RL_Environment_Analysis.md` - Improvement roadmap
4. Phase 2 planning in analysis document

---

## 🧹 Cleanup Summary

**Removed (duplicates/tests):**
- ❌ `advanced_plasma_rl.py` - Failed training attempt
- ❌ `train_plasma_rl.py` - Old test script
- ❌ `linear_plasma_surrogate.py` - Root duplicate
- ❌ `complete_plasma_analysis.py` - Root duplicate

**Archived (old docs):**
- 📦 `Phase1_Problem_Validation.md`
- 📦 `Phase2_Ideation.md`
- 📦 `CLEANUP_PLAN.md`

**Result**: Clean, organized repository with clear structure! ✨

---

## 💡 Tips

### Finding Files Quickly

**Looking for documentation?** → Root directory  
**Need to run PoC?** → `EXECUTION_GUIDE.md`  
**Want to modify code?** → `Core Code/` (3 files)  
**Analyzing results?** → `rl_training_logs/`, `physics_analysis/`  
**Understanding surrogate?** → `linear_surrogate/`

### File Naming Convention

- `*.md` - Markdown documentation
- `*.py` - Python source code
- `*.pkl` - Pickled Python objects (models)
- `*.json` - JSON data files
- `*.csv` - CSV data files
- `*.png` - Image visualizations
- `*.zip` - Compressed RL models

---

**This structure is designed for clarity, reproducibility, and ease of use!** 🎯

**Last Updated**: October 26, 2025  
**Status**: ✅ Clean & Organized
