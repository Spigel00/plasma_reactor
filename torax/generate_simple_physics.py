#!/usr/bin/env python3
"""
Generate Complete Physics Run for Analysis

This script creates synthetic physics data for comprehensive analysis.
"""

import numpy as np
import xarray as xr
from pathlib import Path

def create_synthetic_physics_data():
    """Create synthetic physics data for analysis."""
    
    print("Creating synthetic physics data for analysis...")
    
    # Time grid
    time = np.linspace(0, 5.0, 50)
    nt = len(time)
    
    # Radial grid  
    rho = np.linspace(0, 1, 25)
    nr = len(rho)
    
    # Create synthetic physics profiles
    data = {}
    
    # Initialize arrays
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
    
    # Fill with physics-based profiles
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
        
        # Store profiles
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
        
        # Coil currents (controls) with realistic correlations
        base_coils = [10.0, 8.0, 12.0, 6.0]  # kA
        for i in range(4):
            perturbation = 0.5 * np.sin(2*np.pi*t/(3+i)) + 0.2 * np.cos(2*np.pi*t/(4+i))
            data[f'coil_current_{i+1}'][t_idx] = base_coils[i] + perturbation
    
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
        }
    )
    
    # Add units as individual variable attributes (NetCDF compatible)
    ds['temp_el'].attrs['units'] = 'keV'
    ds['temp_el'].attrs['long_name'] = 'Electron temperature'
    ds['temp_ion'].attrs['units'] = 'keV'
    ds['temp_ion'].attrs['long_name'] = 'Ion temperature'
    ds['ne'].attrs['units'] = '10^19 m^-3'
    ds['ne'].attrs['long_name'] = 'Electron density'
    ds['ni'].attrs['units'] = '10^19 m^-3'
    ds['ni'].attrs['long_name'] = 'Ion density'
    ds['Ip'].attrs['units'] = 'MA'
    ds['Ip'].attrs['long_name'] = 'Plasma current'
    ds['P_heat'].attrs['units'] = 'MW'
    ds['P_heat'].attrs['long_name'] = 'Heating power'
    ds['q'].attrs['units'] = 'dimensionless'
    ds['q'].attrs['long_name'] = 'Safety factor'
    ds['jtot'].attrs['units'] = 'MA/m^2'
    ds['jtot'].attrs['long_name'] = 'Total current density'
    ds['elongation'].attrs['units'] = 'dimensionless'
    ds['elongation'].attrs['long_name'] = 'Plasma elongation'
    ds['triangularity'].attrs['units'] = 'dimensionless'
    ds['triangularity'].attrs['long_name'] = 'Plasma triangularity'
    ds['R_centroid'].attrs['units'] = 'm'
    ds['R_centroid'].attrs['long_name'] = 'Major radius centroid'
    ds['Z_centroid'].attrs['units'] = 'm'
    ds['Z_centroid'].attrs['long_name'] = 'Vertical centroid'
    
    for i in range(4):
        ds[f'coil_current_{i+1}'].attrs['units'] = 'kA'
        ds[f'coil_current_{i+1}'].attrs['long_name'] = f'Coil {i+1} current'
    
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
    
    physics_file = create_synthetic_physics_data()
    
    # Verify the created file
    if physics_file and Path(physics_file).exists():
        print(f"\n✓ Physics data ready for analysis: {physics_file}")
        
        # Quick verification
        ds = xr.open_dataset(physics_file)
        print(f"  Data variables: {len(ds.data_vars)}")
        print(f"  Time dimension: {ds.sizes.get('time', 0)} points")
        print(f"  Spatial dimensions: {dict(ds.sizes)}")
        print(f"  Key variables: {list(ds.data_vars.keys())[:8]}...")
        ds.close()
        
        return physics_file
    else:
        print("✗ Failed to generate physics data")
        return None

if __name__ == "__main__":
    result = main()