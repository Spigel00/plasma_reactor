# Torax Fusion Simulation Framework Setup

## Repository Setup Commands

### For VS Code Local Setup

#### Windows PowerShell
```powershell
# Navigate to your workspace
cd C:\Users\ashwa\Desktop\plasma_reactor

# Clone the repository (note: it's from Google DeepMind, not UKAEA)
git clone https://github.com/google-deepmind/torax.git

# Navigate to the cloned directory
cd torax

# Install in editable mode (assuming Python virtual environment is activated)
pip install -e .

# Alternative: If you want to install with development dependencies
pip install -e .[dev]

# Alternative: If you want to install with tutorial dependencies
pip install -e .[tutorial]
```

#### Linux/Mac Terminal
```bash
# Navigate to your workspace
cd ~/workspace  # or your preferred directory

# Clone the repository
git clone https://github.com/google-deepmind/torax.git

# Navigate to the cloned directory
cd torax

# Install in editable mode
pip install -e .

# Alternative with development dependencies
pip install -e .[dev]

# Alternative with tutorial dependencies
pip install -e .[tutorial]
```

### For Google Colab Setup

```python
# In a Colab cell
!git clone https://github.com/google-deepmind/torax.git
%cd torax
!pip install -e .

# Import necessary libraries after installation
import torax
import jax
import numpy as np
import matplotlib.pyplot as plt
```

## System Requirements

- **Python**: >= 3.11
- **Key Dependencies**:
  - JAX >= 0.7.0
  - JAXlib >= 0.7.0
  - NumPy > 2
  - Matplotlib >= 3.3.0
  - SciPy >= 1.13.0
  - And many more (see pyproject.toml)

## Verification Commands

```python
# Test the installation
python -c "import torax; print('Torax installed successfully!')"

# Check JAX functionality
python -c "import jax; print('JAX version:', jax.__version__)"

# List available examples
import os
print(os.listdir('torax/examples/'))
```