# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom config for plasma reactor simulation with adjustable parameters."""

CONFIG = {
    'profile_conditions': {
        # Plasma current Ip in Amperes (MA * 1e6)
        'Ip': 15.0e6,  # 15 MA plasma current
        
        # Initial and boundary conditions for temperature
        'T_i': {0.0: {0.0: 8.0, 1.0: 0.2}},  # Ion temperature (keV)
        'T_i_right_bc': 0.2,  # Ion temperature boundary condition at edge
        'T_e': {0.0: {0.0: 8.0, 1.0: 0.2}},  # Electron temperature (keV)
        'T_e_right_bc': 0.2,  # Electron temperature boundary condition at edge
        
        # Density conditions
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,  # Boundary condition for electron density (Greenwald fraction)
        'n_e_nbar_is_fGW': True,
        'nbar': 1.0,  # Line-averaged density (Greenwald fraction)
        'n_e': {0: {0.0: 1.2, 1.0: 1.0}},  # Initial electron density profile
    },
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},  # 50-50 Deuterium-Tritium mix
        'Z_eff': 1.8,  # Effective charge (includes impurities)
    },
    'numerics': {
        't_final': 5.0,  # Simulation time in seconds
        'fixed_dt': 0.05,  # Fixed time step
    },
    # Circular geometry for simplified simulation
    'geometry': {
        'geometry_type': 'circular',
        'R_major': 6.2,  # Major radius in meters
        'a_minor': 2.0,  # Minor radius in meters 
        'B_0': 5.3,  # Magnetic field on axis in Tesla
        'elongation_LCFS': 1.72,  # Plasma elongation
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation)
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs)
        'generic_heat': {
            'gaussian_location': 0.2,  # Heating deposition location (normalized radius)
            'gaussian_width': 0.1,     # Heating deposition width
            'P_total': 50.0e6,         # Total heating power P in Watts (50 MW)
            'electron_heat_fraction': 0.6,  # Fraction of power to electrons
        },
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}

# For easy parameter modification in parameter sweeps
def get_config_with_params(Ip_MA=15.0, Ic_MA=None, P_MW=50.0):
    """
    Get configuration with modified parameters.
    
    Args:
        Ip_MA: Plasma current in MA (megaamperes)
        Ic_MA: Coil current in MA (if applicable - placeholder for future use)
        P_MW: Total heating power in MW (megawatts)
    
    Returns:
        Modified CONFIG dictionary
    """
    import copy
    config = copy.deepcopy(CONFIG)
    
    # Set plasma current (convert MA to A)
    config['profile_conditions']['Ip'] = Ip_MA * 1e6
    
    # Set heating power (convert MW to W)
    config['sources']['generic_heat']['P_total'] = P_MW * 1e6
    
    # Note: Ic_MA is kept as a parameter for consistency with the request,
    # but in TORAX the coil current is typically handled through the 
    # magnetic equilibrium in the geometry configuration
    
    return config