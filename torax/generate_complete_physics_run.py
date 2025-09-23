#!/usr/bin/env python3
"""
Generate Complete Physics Run for Analysis

This script creates a full TORAX simulation with all physics variables
for comprehensive analysis and surrogate model development.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import json
import os

# Add the torax directory to path
import sys
torax_path = str(Path(__file__).parent / "torax")
if torax_path not in sys.path:
    sys.path.insert(0, torax_path)

try:
    import torax
    from torax import config as config_lib
    from torax import geometry
    from torax import sim as sim_lib
    from torax import output
    from torax.config import config_args
    from torax.config import runtime_params as runtime_params_lib
    from torax.config import runtime_params_slice
    from torax.config import build_sim
    print("✓ TORAX modules imported successfully")
except ImportError as e:
    print(f"Error importing TORAX: {e}")
    print("Creating synthetic physics data instead...")

def create_complete_physics_config():
    """Create a comprehensive TORAX configuration for physics analysis."""
    
    config = {
        'runtime_params': {
            'numerics': {
                'dt_reduction_factor': 3,
                'fixed_dt': 0.1,
                't_final': 5.0,
                'ion_heat_eq': True,
                'el_heat_eq': True,
                'dens_eq': True,
                'current_eq': True,
            },
            'profile_conditions': {
                'set_pedestal': False,
                'nbar': 0.85,
                'Ip': 15.0e6,  # 15 MA
                'Zeff': 2.5,
                'Ti_bound_right': 0.1,
                'Te_bound_right': 0.1,
            },
            'transport': {
                'chii_const': 1.0,
                'chie_const': 1.0,
                'D_const': 0.5,
                'V_const': 0.0,
            },
        },
        'geometry': {
            'geometry_type': 'circular',
            'Rmaj': 6.2,
            'Rmin': 2.0,
            'B0': 5.3,
            'drho_norm': 0.01,
        },
        'sources': {
            'generic_heat_source': {
                'mode': 'prescribed',
                'total_power': 50e6,  # 50 MW
                'profile_form': 'exponential',
                'He_fraction': 0.5,
                'Hi_fraction': 0.5,
            },
            'bootstrap_current': {
                'mode': 'prescribed',
            },
            'external_current': {
                'mode': 'prescribed',
            }
        },
        'stepper': {
            'stepper_type': 'linear',
            'predictor_corrector': False,
        },
        'time_step_calculator': {
            'calculator_type': 'fixed',
        }
    }
    
    return config

def run_torax_simulation():
    """Run a complete TORAX simulation with physics output."""
    
    print("Setting up TORAX simulation...")
    
    try:
        # Create config
        config_dict = create_complete_physics_config()
        
        # Build simulation
        print("Building simulation configuration...")
        sim = build_sim.build_sim_from_config(config_dict)
        
        # Run simulation
        print("Running TORAX simulation...")
        output_data = sim_lib.run_simulation(sim)
        
        # Save results
        output_file = 'complete_physics_run.nc'
        print(f"Saving results to: {output_file}")
        output.save_to_netcdf(output_data, output_file)
        
        print(f"✓ Complete physics simulation saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error running TORAX simulation: {e}")
        return None

def create_synthetic_physics_data():
    """Create synthetic physics data for analysis if TORAX run fails."""
    
    print("Creating synthetic physics data for analysis...")
    
    # Time grid
    time = np.linspace(0, 5.0, 50)
    nt = len(time)
    
    # Radial grid  
    rho = np.linspace(0, 1, 25)
    nr = len(rho)
    
    # Create synthetic physics profiles
    data = {}
    
    # Base profiles
    for t_idx in range(nt):
        t = time[t_idx]
        
        # Evolving plasma parameters
        Ip_t = 15.0 + 2.0 * np.sin(2*np.pi*t/5.0)  # 15 ± 2 MA
        P_heat_t = 50.0 + 10.0 * np.cos(2*np.pi*t/3.0)  # 50 ± 10 MW
        
        # Temperature profiles (keV)
        Te_profile = 20.0 * (1 - rho**2)**2 * (1 + 0.1*np.sin(2*np.pi*t/2.0))
        Ti_profile = 15.0 * (1 - rho**2)**2 * (1 + 0.05*np.cos(2*np.pi*t/2.5))
        
        # Density profiles (10^19 m^-3)
        ne_profile = 10.0 * (1 - rho**2)**0.5 * (1 + 0.05*np.sin(2*np.pi*t/4.0))
        ni_profile = ne_profile / 1.5  # Account for Zeff
        
        # Current density profiles (MA/m^2)
        q_profile = 1.0 + 2.0 * rho**2 + 0.1*np.sin(2*np.pi*t/3.0)
        jtot_profile = Ip_t / (np.pi * 2.0**2) * (1 - rho**2) * np.exp(-rho**2)
        j_bootstrap = 0.3 * jtot_profile
        j_external = jtot_profile - j_bootstrap
        
        # Store time-dependent data
        if t_idx == 0:
            data['temp_el'] = np.zeros((nt, nr))
            data['temp_ion'] = np.zeros((nt, nr))
            data['ne'] = np.zeros((nt, nr))
            data['ni'] = np.zeros((nt, nr))
            data['q'] = np.zeros((nt, nr))
            data['jtot'] = np.zeros((nt, nr))
            data['j_bootstrap'] = np.zeros((nt, nr))
            data['j_external'] = np.zeros((nt, nr))
            data['Ip'] = np.zeros(nt)
            data['P_heat'] = np.zeros(nt)
            
            # Shape metrics (time-dependent scalars)
            data['elongation'] = np.zeros(nt)
            data['triangularity'] = np.zeros(nt)
            data['R_centroid'] = np.zeros(nt)
            data['Z_centroid'] = np.zeros(nt)
            
            # External controls (time-dependent)
            data['coil_current_1'] = np.zeros(nt)
            data['coil_current_2'] = np.zeros(nt)
            data['coil_current_3'] = np.zeros(nt)
            data['coil_current_4'] = np.zeros(nt)
            
        data['temp_el'][t_idx, :] = Te_profile
        data['temp_ion'][t_idx, :] = Ti_profile
        data['ne'][t_idx, :] = ne_profile
        data['ni'][t_idx, :] = ni_profile
        data['q'][t_idx, :] = q_profile
        data['jtot'][t_idx, :] = jtot_profile
        data['j_bootstrap'][t_idx, :] = j_bootstrap
        data['j_external'][t_idx, :] = j_external
        data['Ip'][t_idx] = Ip_t
        data['P_heat'][t_idx] = P_heat_t
        
        # Shape evolution with control response
        data['elongation'][t_idx] = 1.6 + 0.1*np.sin(2*np.pi*t/4.0)
        data['triangularity'][t_idx] = 0.3 + 0.05*np.cos(2*np.pi*t/6.0)
        data['R_centroid'][t_idx] = 6.2 + 0.02*np.sin(2*np.pi*t/3.0)
        data['Z_centroid'][t_idx] = 0.0 + 0.01*np.cos(2*np.pi*t/5.0)
        
        # Coil currents (controls)
        data['coil_current_1'][t_idx] = 10.0 + 1.0*np.sin(2*np.pi*t/4.0)  # kA
        data['coil_current_2'][t_idx] = 8.0 + 0.5*np.cos(2*np.pi*t/3.0)
        data['coil_current_3'][t_idx] = 12.0 + 0.8*np.sin(2*np.pi*t/5.0)
        data['coil_current_4'][t_idx] = 6.0 + 0.3*np.cos(2*np.pi*t/6.0)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            # 2D profiles (time, rho)
            'temp_el': (['time', 'rho_cell_norm'], data['temp_el']),
            'temp_ion': (['time', 'rho_cell_norm'], data['temp_ion']),
            'ne': (['time', 'rho_cell_norm'], data['ne']),
            'ni': (['time', 'rho_cell_norm'], data['ni']),
            'q': (['time', 'rho_cell_norm'], data['q']),
            'jtot': (['time', 'rho_cell_norm'], data['jtot']),
            'j_bootstrap': (['time', 'rho_cell_norm'], data['j_bootstrap']),
            'j_external': (['time', 'rho_cell_norm'], data['j_external']),
            
            # 1D time series
            'Ip': (['time'], data['Ip']),
            'P_heat': (['time'], data['P_heat']),
            'elongation': (['time'], data['elongation']),
            'triangularity': (['time'], data['triangularity']),
            'R_centroid': (['time'], data['R_centroid']),
            'Z_centroid': (['time'], data['Z_centroid']),
            
            # Control variables
            'coil_current_1': (['time'], data['coil_current_1']),
            'coil_current_2': (['time'], data['coil_current_2']),
            'coil_current_3': (['time'], data['coil_current_3']),
            'coil_current_4': (['time'], data['coil_current_4']),
        },
        coords={
            'time': time,
            'rho_cell_norm': rho,
        },
        attrs={
            'title': 'Synthetic TORAX Physics Data for Analysis',
            'description': 'Complete plasma physics data with time evolution',
            'created': 'Synthetic data for analysis pipeline',
            'units': {
                'temp_el': 'keV',
                'temp_ion': 'keV', 
                'ne': '10^19 m^-3',
                'ni': '10^19 m^-3',
                'Ip': 'A',
                'P_heat': 'W',
                'q': 'dimensionless',
                'jtot': 'MA/m^2',
                'elongation': 'dimensionless',
                'triangularity': 'dimensionless',
                'R_centroid': 'm',
                'Z_centroid': 'm',
                'coil_current_1': 'kA',
                'coil_current_2': 'kA',
                'coil_current_3': 'kA',
                'coil_current_4': 'kA',
            }
        }
    )
    
    # Save synthetic data
    output_file = 'synthetic_complete_physics.nc'
    ds.to_netcdf(output_file)
    
    print(f"✓ Synthetic physics data saved to {output_file}")
    print(f"  Variables: {len(ds.data_vars)}")
    print(f"  Time points: {len(time)}")
    print(f"  Radial points: {len(rho)}")
    
    return output_file

def main():
    """Main function to generate complete physics data."""
    
    print("GENERATING COMPLETE PHYSICS RUN FOR ANALYSIS")
    print("="*60)
    
    # Try to run TORAX simulation first
    physics_file = run_torax_simulation()
    
    # If TORAX fails, create synthetic data
    if physics_file is None:
        physics_file = create_synthetic_physics_data()
    
    # Verify the created file
    if physics_file and Path(physics_file).exists():
        print(f"\n✓ Physics data ready for analysis: {physics_file}")
        
        # Quick verification
        ds = xr.open_dataset(physics_file)
        print(f"  Data variables: {len(ds.data_vars)}")
        print(f"  Time dimension: {ds.sizes.get('time', 0)} points")
        print(f"  Spatial dimensions: {dict(ds.sizes)}")
        ds.close()
        
        return physics_file
    else:
        print("✗ Failed to generate physics data")
        return None

if __name__ == "__main__":
    result = main()