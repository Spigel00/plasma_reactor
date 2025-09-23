# TORAX Fusion Simulation Framework - Complete 1-Day Workflow

## 🚀 Overview

This guide provides a complete step-by-step workflow for setting up and working with the TORAX fusion simulation framework, including repository setup, running simulations, creating custom configurations, and performing parameter sweeps.

## 📋 Prerequisites

- **Operating System**: Windows, Linux, or macOS
- **Python**: Version 3.11 or higher
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~2GB free space for installation and results

## 🕐 1-Day Roadmap

### Phase 1: Setup and Installation (30-45 minutes)

#### Step 1.1: Repository Setup
```bash
# For VS Code Local Setup (Windows PowerShell)
cd C:\Users\[username]\Desktop\plasma_reactor  # or your preferred directory
git clone https://github.com/google-deepmind/torax.git
cd torax
pip install -e .

# For Google Colab Setup
!git clone https://github.com/google-deepmind/torax.git
%cd torax
!pip install -e .
```

#### Step 1.2: Verify Installation
```bash
# Test basic import
python -c "import torax; print('TORAX installed successfully!')"

# Check JAX functionality
python -c "import jax; print('JAX version:', jax.__version__)"
```

### Phase 2: Example Exploration (15-30 minutes)

#### Step 2.1: Run Basic Example
```bash
# Command for both VS Code and Colab
python torax/run_simulation_main.py --config torax/examples/basic_config.py --log_progress --quit
```

**Expected Output:**
- Simulation completes in ~12 seconds
- Creates output file: `/tmp/torax_results/state_history_[timestamp].nc`
- Logs simulation progress with time steps

#### Step 2.2: Understand Example Structure
Examine `torax/examples/basic_config.py` to understand:
- Configuration dictionary structure
- Default plasma parameters
- Source and transport models

### Phase 3: Custom Configuration (30-45 minutes)

#### Step 3.1: Create Custom Config File
Create `torax/examples/my_config.py` with custom parameters:

```python
CONFIG = {
    'profile_conditions': {
        'Ip': 15.0e6,  # 15 MA plasma current (in Amperes)
        'T_i': {0.0: {0.0: 8.0, 1.0: 0.2}},  # Ion temperature profile
        'T_e': {0.0: {0.0: 8.0, 1.0: 0.2}},  # Electron temperature profile
        'T_i_right_bc': 0.2,  # Boundary conditions
        'T_e_right_bc': 0.2,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,
        'n_e_nbar_is_fGW': True,
        'nbar': 1.0,
        'n_e': {0: {0.0: 1.2, 1.0: 1.0}},
    },
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},  # 50-50 D-T mix
        'Z_eff': 1.8,
    },
    'sources': {
        'generic_heat': {
            'gaussian_location': 0.2,
            'gaussian_width': 0.1,
            'P_total': 50.0e6,  # 50 MW heating power (in Watts)
            'electron_heat_fraction': 0.6,
        },
        # ... other sources
    },
    # ... other configuration sections
}
```

#### Step 3.2: Test Custom Configuration
```bash
python torax/run_simulation_main.py --config torax/examples/my_config.py --log_progress --quit
```

### Phase 4: Parameter Sweep Implementation (45-60 minutes)

#### Step 4.1: Create Parameter Sweep Script
Key components of the parameter sweep:

1. **Configuration Generator Function:**
```python
def get_config_with_params(Ip_MA=15.0, Ic_MA=None, P_MW=50.0):
    config = copy.deepcopy(BASE_CONFIG)
    config['profile_conditions']['Ip'] = Ip_MA * 1e6  # Convert MA to A
    config['sources']['generic_heat']['P_total'] = P_MW * 1e6  # Convert MW to W
    return config
```

2. **Simulation Runner:**
```python
def run_single_simulation(Ip_MA, P_MW, config_file, output_dir):
    # Create temporary config file
    # Run subprocess with torax/run_simulation_main.py
    # Parse output and collect results
    return results_dict
```

3. **Parameter Loop:**
```python
P_values = range(10, 101, 10)  # 10, 20, ..., 100 MW
for P_MW in P_values:
    results.append(run_single_simulation(Ip_MA=15.0, P_MW=P_MW))
```

#### Step 4.2: Execute Parameter Sweep
```bash
python parameter_sweep.py
```

**Expected Results:**
- 10 simulations (P: 10-100 MW, step: 10 MW)
- Total runtime: ~2.5 minutes
- Output: CSV and JSON files with results

### Phase 5: Results Analysis (30-45 minutes)

#### Step 5.1: Examine Results Table
Example results format:

| Ip [MA] | Ic [MA] | P [MW] | Success | Runtime [s] | Output File |
|---------|---------|--------|---------|-------------|-------------|
| 15.0    | 0.0     | 10     | ✓       | 14.2        | state_history_*.nc |
| 15.0    | 0.0     | 20     | ✓       | 14.5        | state_history_*.nc |
| ...     | ...     | ...    | ...     | ...         | ... |

#### Step 5.2: Data Analysis Tasks
1. **Load Results:**
```python
import pandas as pd
results = pd.read_csv('parameter_sweep_results.csv')
```

2. **Basic Analysis:**
- Success rate across power levels
- Average runtime per simulation
- Power scaling relationships
- Parameter sensitivity analysis

#### Step 5.3: Visualization (Optional)
```python
import matplotlib.pyplot as plt

# Plot power vs runtime
plt.figure(figsize=(10, 6))
successful_runs = results[results['success'] == True]
plt.plot(successful_runs['P_MW'], successful_runs['runtime_seconds'], 'bo-')
plt.xlabel('Heating Power (MW)')
plt.ylabel('Runtime (seconds)')
plt.title('TORAX Simulation Runtime vs Heating Power')
plt.grid(True)
plt.show()
```

### Phase 6: Documentation and Wrap-up (15-30 minutes)

#### Step 6.1: Document Results
Create summary documentation including:
- Parameter space explored
- Success metrics
- Key findings
- Lessons learned

#### Step 6.2: File Organization
Organize your workspace:
```
plasma_reactor/
├── torax/                          # Main TORAX repository
│   ├── examples/
│   │   ├── basic_config.py        # Original example
│   │   └── my_config.py           # Your custom config
│   └── run_simulation_main.py     # Main simulation script
├── parameter_sweep.py             # Your parameter sweep script
├── parameter_sweep_results.csv    # Results table
├── parameter_sweep_results.json   # Detailed results
└── torax_setup_commands.md        # Setup documentation
```

## 🎯 Key Learning Outcomes

By the end of this 1-day workflow, you will have:

1. ✅ **Installed TORAX** and verified the setup
2. ✅ **Run example simulations** and understood the output
3. ✅ **Created custom configurations** with specific plasma parameters
4. ✅ **Implemented parameter sweeps** to explore parameter space
5. ✅ **Analyzed results** and generated summary tables
6. ✅ **Documented the workflow** for future reference

## 🛠️ Command Reference

### Essential Commands

```bash
# Install TORAX
git clone https://github.com/google-deepmind/torax.git
cd torax
pip install -e .

# Run basic simulation
python torax/run_simulation_main.py --config torax/examples/basic_config.py --quit

# Run custom simulation
python torax/run_simulation_main.py --config torax/examples/my_config.py --quit

# Run parameter sweep
python parameter_sweep.py

# Check help
python torax/run_simulation_main.py --help
```

### For Google Colab

```python
# Installation
!git clone https://github.com/google-deepmind/torax.git
%cd torax
!pip install -e .

# Import and run
import torax
# Run simulations using Python API (advanced)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Error: 'imas_core'**
   - This is expected and non-critical
   - TORAX works without IMAS functionality

2. **JAX Backend Warnings**
   - Expected on Windows (TPU not available)
   - CPU backend works fine for learning

3. **Permission Errors on Output**
   - Use `--output_dir` to specify writable directory
   - Ensure sufficient disk space

4. **Simulation Hangs**
   - Always use `--quit` flag for batch processing
   - Set appropriate timeouts in parameter sweeps

### Performance Tips

- Use shorter `t_final` for parameter sweeps (e.g., 3.0 instead of 5.0 seconds)
- Increase `fixed_dt` for faster but less accurate runs
- Monitor memory usage for large parameter sweeps

## 📚 Next Steps

After completing this 1-day workflow, consider:

1. **Advanced Configurations**: Explore more complex geometry and transport models
2. **Physics Analysis**: Analyze plasma profiles, currents, and temperatures
3. **Parameter Optimization**: Use optimization algorithms to find optimal parameters
4. **Integration**: Connect TORAX with other plasma physics tools
5. **Custom Sources**: Implement custom heating and current drive models

## 🤝 Resources

- **TORAX Documentation**: [https://github.com/google-deepmind/torax](https://github.com/google-deepmind/torax)
- **JAX Documentation**: [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
- **Plasma Physics References**: Standard tokamak physics textbooks
- **Community**: GitHub issues and discussions

---

*This workflow is designed to be completed in approximately 4-6 hours, making it perfect for a focused 1-day learning session on fusion simulation with TORAX.*