#!/usr/bin/env python3
"""
Generate TORAX simulation with proper physics output for analysis.

This script runs a TORAX simulation specifically configured to output
complete physics data including Te, ne, currents, and shape metrics.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

def create_physics_config():
    """Create configuration optimized for physics data output."""
    
    config_content = '''# Physics data generation config
CONFIG = {
    'profile_conditions': {
        'Ip': 15.0e6,  # 15 MA plasma current
        'T_i': {0.0: {0.0: 10.0, 0.5: 5.0, 1.0: 0.5}},  # Ion temperature profile (keV)
        'T_i_right_bc': 0.5,
        'T_e': {0.0: {0.0: 12.0, 0.5: 6.0, 1.0: 0.8}},  # Electron temperature profile (keV)
        'T_e_right_bc': 0.8,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,
        'n_e_nbar_is_fGW': True,
        'nbar': 0.85,  # Line-averaged density
        'n_e': {0: {0.0: 1.0, 0.5: 0.8, 1.0: 0.3}},  # Electron density profile
    },
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'Z_eff': 2.0,
    },
    'numerics': {
        't_final': 3.0,  # 3 second simulation
        'fixed_dt': 0.05,  # Fine time resolution
    },
    'geometry': {
        'geometry_type': 'circular',
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
        'elongation_LCFS': 1.8,  # Slightly elongated
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        'generic_current': {},
        'generic_particle': {},  # Simplified particle source
        'gas_puff': {},
        'pellet': {},
        'generic_heat': {
            'gaussian_location': 0.25,
            'gaussian_width': 0.15,
            'P_total': 75.0e6,  # 75 MW heating power
            'electron_heat_fraction': 0.67,
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
}'''
    
    config_file = "physics_config.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file

def run_physics_simulation():
    """Run TORAX simulation to generate proper physics data."""
    
    print("="*60)
    print("GENERATING TORAX PHYSICS DATA")
    print("="*60)
    
    # Create physics config
    config_file = create_physics_config()
    print(f"✓ Created physics configuration: {config_file}")
    
    # Prepare simulation command
    python_exe = r"C:/Users/ashwa/Desktop/plasma_reactor/.venv/Scripts/python.exe"
    cmd = [
        python_exe, 
        "torax/run_simulation_main.py", 
        "--config", config_file,
        "--quit"  # Auto-quit after simulation
    ]
    
    print(f"✓ Running simulation command: {' '.join(cmd)}")
    print("  This may take 1-2 minutes...")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            input="q\\n",
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        runtime = time.time() - start_time
        
        print(f"\\n✓ Simulation completed in {runtime:.1f} seconds")
        print(f"  Return code: {result.returncode}")
        
        # Check for output file
        output_file = None
        if "Wrote simulation output to" in result.stderr:
            for line in result.stderr.split('\\n'):
                if "Wrote simulation output to" in line:
                    output_file = line.split("Wrote simulation output to")[-1].strip()
                    break
        
        if output_file and os.path.exists(output_file):
            print(f"✓ Physics data written to: {output_file}")
            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"  File size: {file_size:.1f} KB")
            return output_file
        else:
            print("⚠️  No output file found")
            print("\\nSTDOUT:")
            print(result.stdout)
            print("\\nSTDERR:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("✗ Simulation timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"✗ Simulation failed: {str(e)}")
        return None

def examine_physics_output(output_file):
    """Examine the generated physics output file."""
    
    if not output_file or not os.path.exists(output_file):
        print("No output file to examine")
        return
    
    print(f"\\n{'='*60}")
    print("EXAMINING PHYSICS OUTPUT")
    print("="*60)
    
    try:
        import xarray as xr
        import json
        
        ds = xr.open_dataset(output_file)
        
        print(f"File: {os.path.basename(output_file)}")
        print(f"Size: {os.path.getsize(output_file)/1024:.1f} KB")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
        
        # Look for specific physics variables
        physics_vars = ['temp_el', 'temp_ion', 'ne', 'ni', 'psi', 'q_profile', 'j_bootstrap', 'chi_e', 'chi_i']
        found_vars = []
        for var in physics_vars:
            if var in ds.data_vars:
                found_vars.append(var)
                shape_str = " × ".join(map(str, ds[var].shape))
                print(f"  ✓ {var}: {ds[var].dims} [{shape_str}]")
        
        if not found_vars:
            print("  ⚠️  No standard physics variables found")
            if ds.data_vars:
                print("  Available variables:")
                for var in ds.data_vars:
                    shape_str = " × ".join(map(str, ds[var].shape))
                    print(f"    {var}: {ds[var].dims} [{shape_str}]")
        
        # Check config
        if 'config' in ds.attrs:
            try:
                config = json.loads(ds.attrs['config'])
                print(f"\\n✓ Configuration found in attributes")
                if 'profile_conditions' in config:
                    profile = config['profile_conditions']
                    if 'Ip' in profile:
                        ip_val = profile['Ip']
                        if isinstance(ip_val, list) and len(ip_val) == 2:
                            print(f"  Plasma current: {ip_val[1][0]/1e6:.1f} MA")
                        else:
                            print(f"  Plasma current: {ip_val/1e6:.1f} MA")
            except:
                print("  ⚠️  Could not parse configuration")
        
        ds.close()
        
        return len(found_vars) > 0
        
    except Exception as e:
        print(f"✗ Error examining output: {str(e)}")
        return False

def main():
    """Main function to generate and examine physics data."""
    
    # Change to torax directory
    os.chdir("C:/Users/ashwa/Desktop/plasma_reactor/torax")
    
    # Run physics simulation
    output_file = run_physics_simulation()
    
    # Examine the output
    has_physics = examine_physics_output(output_file)
    
    print(f"\\n{'='*60}")
    if output_file and has_physics:
        print("✓ SUCCESS: Generated NetCDF file with physics data")
        print(f"  File: {output_file}")
        print("  Ready for Step 1 analysis (filtering good files)")
    elif output_file:
        print("⚠️  PARTIAL SUCCESS: Generated NetCDF file but no physics variables found")
        print("  File may contain coordinate grids only")
    else:
        print("✗ FAILED: No output file generated")
        print("  Check TORAX installation and configuration")
    
    print("="*60)

if __name__ == "__main__":
    main()