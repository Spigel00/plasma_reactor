# Plasma Reactor Analysis & RL Surrogate Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TORAX](https://img.shields.io/badge/Framework-TORAX-orange.svg)](https://github.com/google-deepmind/torax)

A comprehensive plasma physics analysis toolkit with reinforcement learning integration for tokamak plasma control. This project implements deep plasma dynamics analysis and creates fast linear surrogate models ready for RL training environments.

## 🚀 Project Overview

This project provides a complete pipeline for:
- **Plasma Physics Analysis**: Deep analysis of tokamak plasma dynamics including temperature profiles, current evolution, and shape control
- **Surrogate Model Development**: Fast linear regression models mapping control inputs to plasma observables
- **RL Integration**: Ready-to-use interfaces for reinforcement learning environments
- **Visualization**: Comprehensive plotting and analysis dashboards

### Key Features

- ✅ **Complete Analysis Pipeline**: From raw NetCDF data to RL-ready models
- ✅ **Physics-Based**: Built on TORAX simulation framework from Google DeepMind
- ✅ **Fast Inference**: Sub-millisecond prediction for real-time RL training
- ✅ **Comprehensive Visualization**: Multi-panel analysis plots and dashboards
- ✅ **Production Ready**: Clean interfaces and organized codebase

## 📊 Analysis Capabilities

### Plasma Dynamics Analysis
- **Temperature Profiles**: Electron (Te) and ion (Ti) temperature evolution
- **Density Analysis**: Electron density (ne) spatial and temporal dynamics
- **Current Analysis**: Plasma current (Ip) and current density j(ρ,t) profiles
- **Shape Control**: Elongation (κ) and triangularity (δ) evolution
- **Position Control**: Centroid tracking (R, Z coordinates)
- **Safety Factor**: q-profile evolution for MHD stability analysis

### Control System Analysis
- **Coil Current Analysis**: 4-coil control system characterization
- **Response Mapping**: Linear mapping from coil currents to plasma observables
- **Sensitivity Analysis**: Quantified control authority for each actuator
- **Correlation Studies**: Cross-correlation between controls and responses

## 🤖 Machine Learning Integration

### Linear Surrogate Model
- **Input**: 4 coil currents [kA]
- **Output**: 8 plasma observables (position, shape, current, temperatures)
- **Performance**: R² > 0.87 for 7/8 variables
- **Speed**: Sub-millisecond inference time

### RL-Ready Interface
```python
from linear_surrogate.linear_plasma_surrogate import LinearPlasmaSurrogate

# Initialize surrogate model
surrogate = LinearPlasmaSurrogate()

# Predict plasma response to coil current changes
coil_currents = [10.5, 8.2, 12.1, 6.3]  # [kA]
responses = surrogate.predict(coil_currents)

# Get control sensitivity matrix
response_matrix = surrogate.get_response_matrix()  # Shape: (8, 4)
```

## 📁 Project Structure

```
plasma_reactor/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── DAILY_ANALYSIS_SUMMARY.md         # Comprehensive analysis report
├── torax/                             # Main analysis code
│   ├── plots/                         # Generated visualizations
│   │   ├── comprehensive_plasma_analysis.png
│   │   ├── response_matrix_visualization.png
│   │   └── ...
│   ├── good_runs/                     # Curated NetCDF data files
│   │   ├── synthetic_complete_physics.nc
│   │   └── state_history_*.nc
│   ├── linear_surrogate/              # Surrogate model components
│   │   ├── linear_plasma_surrogate.py # RL interface
│   │   ├── linear_surrogate_model.pkl # Trained model
│   │   └── response_matrices.json     # Control sensitivity data
│   ├── physics_analysis/              # Analysis outputs
│   ├── clean_plasma_analysis.py       # Main analysis script
│   ├── generate_simple_physics.py     # Synthetic data generator
│   └── ...
├── docs/                              # Documentation
└── examples/                          # Usage examples
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Spigel00/plasma_reactor.git
cd plasma_reactor
```

2. **Create virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
cd torax
python clean_plasma_analysis.py
```

## 🚦 Quick Start

### 1. Run Complete Analysis
```bash
cd torax
python clean_plasma_analysis.py
```
This will:
- Load plasma physics data
- Perform comprehensive dynamics analysis
- Generate visualization plots
- Create linear surrogate model
- Save all outputs to organized directories

### 2. Use Surrogate Model for RL
```python
import sys
sys.path.append('torax/linear_surrogate')
from linear_plasma_surrogate import LinearPlasmaSurrogate

# Initialize model
surrogate = LinearPlasmaSurrogate()

# Example control input (coil currents in kA)
control_input = [10.5, 8.2, 12.1, 6.3]

# Get plasma response prediction
response = surrogate.predict(control_input)
print("Predicted plasma state:", response)

# Get sensitivity matrix for control design
sensitivity = surrogate.get_response_matrix()
print("Control sensitivity shape:", sensitivity.shape)
```

### 3. Generate Synthetic Data
```bash
cd torax
python generate_simple_physics.py
```

## 📈 Analysis Results

The analysis generates comprehensive insights into plasma behavior:

### Model Performance
- **R_centroid**: R² = 0.980 (Excellent prediction)
- **Z_centroid**: R² = 0.893 (Very good prediction)  
- **elongation**: R² = 0.978 (Excellent prediction)
- **triangularity**: R² = 0.875 (Very good prediction)
- **ne_avg**: R² = 0.978 (Excellent prediction)
- **Ip**: R² = 0.987 (Excellent prediction)
- **q95**: R² = 0.980 (Excellent prediction)

### Control Authority
- **Plasma Current (Ip)**: Most sensitive to coil_4 (1.630 MA/kA)
- **Elongation**: Most sensitive to coil_2 (0.133 /kA)
- **R Centroid**: Most sensitive to coil_1 (0.030 m/kA)
- **Triangularity**: Most sensitive to coil_3 (0.054 /kA)

## 🎯 Use Cases

### Research Applications
- **Plasma Control Research**: Study optimal control strategies for tokamak operation
- **RL Algorithm Development**: Test reinforcement learning algorithms on plasma control
- **Control System Design**: Analyze actuator effectiveness and design feedback systems
- **Physics Studies**: Investigate plasma behavior and stability limits

### Educational Applications
- **Plasma Physics Education**: Visualize complex plasma dynamics
- **Control Theory**: Demonstrate multi-input multi-output control systems
- **Machine Learning**: Example of physics-informed surrogate modeling
- **Data Science**: Time series analysis and regression modeling

## 📚 Documentation

- **[DAILY_ANALYSIS_SUMMARY.md](DAILY_ANALYSIS_SUMMARY.md)**: Comprehensive analysis report with all technical details
- **[TORAX_Complete_Implementation.md](TORAX_Complete_Implementation.md)**: Implementation details and setup
- **[TORAX_Complete_Workflow.md](TORAX_Complete_Workflow.md)**: Analysis workflow documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TORAX Team at Google DeepMind**: For the excellent plasma physics simulation framework
- **Python Scientific Community**: For the robust ecosystem of scientific computing tools
- **Plasma Physics Community**: For advancing the science of controlled fusion

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **GitHub Issues**: [plasma_reactor/issues](https://github.com/Spigel00/plasma_reactor/issues)
- **Email**: [Contact through GitHub profile]

## 🔬 Technical Details

### Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning models
- **xarray**: NetCDF data handling
- **netCDF4**: Scientific data format support

### System Requirements
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~500MB for code + data
- **CPU**: Any modern processor (analysis is not computationally intensive)
- **OS**: Windows, Linux, or macOS

---

**Ready for plasma physics analysis and reinforcement learning research!** 🚀

*Built with ❤️ for the plasma physics and machine learning communities*