#!/usr/bin/env python3
"""
Generate TORAX simulation with complete physics data extraction.

This script runs TORAX programmatically to access the full StateHistory
object which contains all physics variables, then manually saves them to NetCDF.
"""

import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import json

# Add TORAX to path
sys.path.insert(0, os.path.join(os.getcwd(), 'torax'))

def run_full_physics_simulation():
    """Run TORAX simulation and extract complete physics data."""
    
    print("="*60)
    print("TORAX COMPLETE PHYSICS DATA EXTRACTION")
    print("="*60)
    
    try:
        # Import TORAX modules
        from torax._src import run_simulation
        from torax._src.config import config_loader
        from torax._src.output_tools import output
        
        print("✓ TORAX modules imported successfully")
        
        # Use the working configuration
        config_path = "torax/examples/my_config.py"
        print(f"✓ Loading configuration: {config_path}")
        
        # Load and run simulation
        torax_config = config_loader.build_torax_config_from_file(config_path)
        print("✓ Configuration loaded successfully")
        
        # Run the simulation to get StateHistory
        print("✓ Running simulation...")
        data_tree, state_history = run_simulation.run_simulation(torax_config)
        print(f"✓ Simulation completed with {len(state_history.times)} time points")
        
        # Create output directory
        output_dir = Path("physics_data_output")
        output_dir.mkdir(exist_ok=True)
        
        # Convert StateHistory to comprehensive NetCDF
        print("✓ Converting StateHistory to NetCDF...")
        output_file = output_dir / "complete_physics_data.nc"
        
        # Create xarray dataset with all physics variables
        coords = {
            'time': state_history.times,
            'rho_cell_norm': state_history.rho_cell_norm,
            'rho_face_norm': state_history.rho_face_norm,
            'rho_norm': state_history.rho_norm,
        }
        
        data_vars = {}
        
        # Core profiles (temperature, density, current, etc.)
        if hasattr(state_history, '_stacked_core_profiles'):
            profiles = state_history._stacked_core_profiles
            
            # Temperature profiles
            if hasattr(profiles, 'temp_el'):
                data_vars['temp_el'] = (['time', 'rho_cell_norm'], profiles.temp_el)
                print("  + Added electron temperature (temp_el)")
            if hasattr(profiles, 'temp_ion'):
                data_vars['temp_ion'] = (['time', 'rho_cell_norm'], profiles.temp_ion)
                print("  + Added ion temperature (temp_ion)")
            
            # Density profiles
            if hasattr(profiles, 'ne'):
                data_vars['ne'] = (['time', 'rho_cell_norm'], profiles.ne)
                print("  + Added electron density (ne)")
            if hasattr(profiles, 'ni'):
                data_vars['ni'] = (['time', 'rho_cell_norm'], profiles.ni)
                print("  + Added ion density (ni)")
            
            # Current and magnetic quantities
            if hasattr(profiles, 'psi'):
                data_vars['psi'] = (['time', 'rho_face_norm'], profiles.psi)
                print("  + Added poloidal flux (psi)")
            if hasattr(profiles, 'j_total'):
                data_vars['j_total'] = (['time', 'rho_cell_norm'], profiles.j_total)
                print("  + Added total current density (j_total)")
            if hasattr(profiles, 'q_profile'):
                data_vars['q_profile'] = (['time', 'rho_cell_norm'], profiles.q_profile)
                print("  + Added safety factor (q_profile)")
            
            # Plasma current
            if hasattr(profiles, 'currents') and hasattr(profiles.currents, 'Ip'):
                data_vars['Ip'] = (['time'], profiles.currents.Ip)
                print("  + Added plasma current (Ip)")
        
        # Transport coefficients
        if hasattr(state_history, '_stacked_core_transport'):
            transport = state_history._stacked_core_transport
            
            if hasattr(transport, 'chi_e'):
                data_vars['chi_e'] = (['time', 'rho_cell_norm'], transport.chi_e)
                print("  + Added electron heat diffusivity (chi_e)")
            if hasattr(transport, 'chi_i'):
                data_vars['chi_i'] = (['time', 'rho_cell_norm'], transport.chi_i)
                print("  + Added ion heat diffusivity (chi_i)")
            if hasattr(transport, 'D_e'):
                data_vars['D_e'] = (['time', 'rho_cell_norm'], transport.D_e)
                print("  + Added particle diffusivity (D_e)")
        
        # Sources
        if hasattr(state_history, '_stacked_core_sources'):
            sources = state_history._stacked_core_sources
            
            if hasattr(sources, 'qei'):
                data_vars['qei_source'] = (['time', 'rho_cell_norm'], sources.qei)
                print("  + Added electron-ion exchange source (qei)")
        
        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Add configuration as attributes
        config_dict = torax_config.model_dump()
        ds.attrs['config'] = json.dumps(config_dict)
        ds.attrs['description'] = 'Complete TORAX physics simulation data'
        ds.attrs['generated_by'] = 'TORAX physics extraction script'
        
        # Save to NetCDF
        ds.to_netcdf(output_file)
        file_size = output_file.stat().st_size / 1024  # KB
        
        print(f"✓ Complete physics data saved to: {output_file}")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Variables: {len(data_vars)}")
        print(f"  Time points: {len(state_history.times)}")
        
        return str(output_file), len(data_vars)
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure TORAX is properly installed")
        return None, 0
    except Exception as e:
        print(f"✗ Error during physics extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def examine_extracted_data(output_file):
    """Examine the extracted physics data."""
    
    if not output_file or not Path(output_file).exists():
        print("No physics data file to examine")
        return False
    
    print(f"\\n{'='*60}")
    print("EXAMINING EXTRACTED PHYSICS DATA")
    print("="*60)
    
    try:
        ds = xr.open_dataset(output_file)
        
        print(f"File: {Path(output_file).name}")
        print(f"Size: {Path(output_file).stat().st_size/1024:.1f} KB")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Data variables: {len(ds.data_vars)}")
        
        # Categorize variables
        temp_vars = [v for v in ds.data_vars if 'temp' in v]
        density_vars = [v for v in ds.data_vars if any(x in v for x in ['ne', 'ni', 'density'])]
        current_vars = [v for v in ds.data_vars if any(x in v for x in ['current', 'Ip', 'j_'])]
        transport_vars = [v for v in ds.data_vars if any(x in v for x in ['chi', 'D_e'])]
        magnetic_vars = [v for v in ds.data_vars if any(x in v for x in ['psi', 'q_', 'B_'])]
        
        print(f"\\nPhysics variable categories:")
        print(f"  Temperature: {temp_vars}")
        print(f"  Density: {density_vars}")
        print(f"  Current: {current_vars}")
        print(f"  Transport: {transport_vars}")
        print(f"  Magnetic: {magnetic_vars}")
        
        # Show sample data for key variables
        key_vars = ['temp_el', 'ne', 'Ip', 'q_profile']
        print(f"\\nSample data:")
        for var in key_vars:
            if var in ds.data_vars:
                if 'time' in ds[var].dims:
                    if ds[var].ndim == 1:  # Time series only
                        print(f"  {var}(t): {ds[var].values[:3]} ... (shape: {ds[var].shape})")
                    else:  # Profile data
                        print(f"  {var}(t=0, center): {ds[var].isel(time=0).values[len(ds[var].values[0])//2]:.3f}")
                        print(f"    Shape: {ds[var].shape}, Range: [{ds[var].min().values:.3f}, {ds[var].max().values:.3f}]")
        
        ds.close()
        return True
        
    except Exception as e:
        print(f"✗ Error examining data: {e}")
        return False

def main():
    """Main function to extract and examine complete physics data."""
    
    # Change to torax directory
    os.chdir("C:/Users/ashwa/Desktop/plasma_reactor/torax")
    
    # Extract complete physics data
    output_file, num_vars = run_full_physics_simulation()
    
    # Examine the extracted data
    success = examine_extracted_data(output_file)
    
    print(f"\\n{'='*60}")
    if output_file and success and num_vars > 0:
        print("✓ SUCCESS: Generated NetCDF file with complete physics data")
        print(f"  File: {output_file}")
        print(f"  Variables: {num_vars}")
        print("  Ready for comprehensive analysis!")
    else:
        print("✗ FAILED: Could not extract complete physics data")
        print("  Falling back to coordinate-only files for analysis")
    
    print("="*60)

if __name__ == "__main__":
    main()