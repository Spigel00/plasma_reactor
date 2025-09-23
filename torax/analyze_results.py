#!/usr/bin/env python3
"""
TORAX Simulation Results Analysis Script

This script analyzes NetCDF output files from TORAX fusion simulations.
It loads simulation data, examines the structure, and creates visualizations
for key plasma parameters including temperature, density, and q-profiles.

Author: Generated for TORAX workflow
Date: September 2025
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

class ToraxAnalyzer:
    """Analyzer class for TORAX simulation NetCDF files."""
    
    def __init__(self, results_dir: str = r'C:\tmp\torax_results'):
        """
        Initialize the analyzer with results directory.
        
        Args:
            results_dir: Path to directory containing NetCDF files
        """
        self.results_dir = Path(results_dir)
        self.datasets = {}
        self.metadata = {}
        
    def load_datasets(self) -> Dict[str, xr.Dataset]:
        """
        Load all NetCDF files from the results directory.
        
        Returns:
            Dictionary mapping filenames to loaded datasets
        """
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return {}
            
        nc_files = list(self.results_dir.glob("*.nc"))
        if not nc_files:
            print(f"No NetCDF files found in {self.results_dir}")
            return {}
            
        print(f"Found {len(nc_files)} NetCDF files:")
        
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                self.datasets[nc_file.name] = ds
                print(f"  ✓ {nc_file.name} - Size: {nc_file.stat().st_size/1024:.1f} KB")
                
                # Extract metadata from attributes if available
                if hasattr(ds, 'attrs') and 'config' in ds.attrs:
                    try:
                        config = json.loads(ds.attrs['config'])
                        self.metadata[nc_file.name] = config
                    except (json.JSONDecodeError, TypeError):
                        print(f"    Warning: Could not parse config for {nc_file.name}")
                        
            except Exception as e:
                print(f"  ✗ Failed to load {nc_file.name}: {e}")
                
        return self.datasets
    
    def examine_structure(self) -> None:
        """Examine and print the structure of loaded datasets."""
        if not self.datasets:
            print("No datasets loaded. Run load_datasets() first.")
            return
            
        print("\n" + "="*60)
        print("DATASET STRUCTURE ANALYSIS")
        print("="*60)
        
        for filename, ds in self.datasets.items():
            print(f"\nFile: {filename}")
            print("-" * 40)
            print(f"Dimensions: {dict(ds.sizes)}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            print(f"Data variables: {list(ds.data_vars.keys())}")
            
            if ds.data_vars:
                print("Variable details:")
                for var in ds.data_vars:
                    shape_str = " × ".join(map(str, ds[var].shape))
                    print(f"  {var}: {ds[var].dims} [{shape_str}]")
            else:
                print("  No data variables found (coordinates only)")
                
            # Check for data in attributes
            if filename in self.metadata:
                config = self.metadata[filename]
                print(f"Configuration found in attributes:")
                if 'profile_conditions' in config:
                    profile = config['profile_conditions']
                    for param in ['Ip', 'P', 'Ic']:
                        if param in profile:
                            value = profile[param].get('value', 'N/A')
                            print(f"  {param}: {value}")
    
    def create_coordinate_plots(self) -> None:
        """Create plots showing the coordinate grids."""
        if not self.datasets:
            print("No datasets loaded.")
            return
            
        # Get a representative dataset
        sample_ds = next(iter(self.datasets.values()))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('TORAX Simulation Coordinate Grids', fontsize=16, fontweight='bold')
        
        # Time evolution
        if 'time' in sample_ds.coords:
            axes[0, 0].plot(sample_ds.time.values, 'bo-', markersize=4)
            axes[0, 0].set_xlabel('Time Index')
            axes[0, 0].set_ylabel('Time (s)')
            axes[0, 0].set_title('Time Grid')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Radial coordinates
        if 'rho_cell_norm' in sample_ds.coords:
            axes[0, 1].plot(sample_ds.rho_cell_norm.values, 'ro-', markersize=4, label='Cell centers')
        if 'rho_face_norm' in sample_ds.coords:
            axes[0, 1].plot(sample_ds.rho_face_norm.values, 'bs-', markersize=3, label='Face centers')
        if 'rho_norm' in sample_ds.coords:
            axes[0, 1].plot(sample_ds.rho_norm.values, 'g^-', markersize=3, label='General grid')
        
        axes[0, 1].set_xlabel('Grid Index')
        axes[0, 1].set_ylabel('Normalized Minor Radius (ρ)')
        axes[0, 1].set_title('Radial Grid Structure')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Grid sizes comparison
        grid_names = []
        grid_sizes = []
        for coord in ['time', 'rho_cell_norm', 'rho_face_norm', 'rho_norm']:
            if coord in sample_ds.coords:
                grid_names.append(coord.replace('_', '\n'))
                grid_sizes.append(len(sample_ds.coords[coord]))
        
        axes[1, 0].bar(grid_names, grid_sizes, color=['blue', 'red', 'green', 'orange'][:len(grid_names)])
        axes[1, 0].set_ylabel('Grid Size')
        axes[1, 0].set_title('Coordinate Grid Sizes')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # File sizes comparison
        filenames = []
        filesizes = []
        for filename in self.datasets.keys():
            file_path = self.results_dir / filename
            if file_path.exists():
                filenames.append(filename.replace('state_history_', '').replace('.nc', ''))
                filesizes.append(file_path.stat().st_size / 1024)  # KB
        
        if filenames:
            bars = axes[1, 1].bar(range(len(filenames)), filesizes, color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('Simulation Files')
            axes[1, 1].set_ylabel('File Size (KB)')
            axes[1, 1].set_title('NetCDF File Sizes')
            axes[1, 1].set_xticks(range(len(filenames)))
            axes[1, 1].set_xticklabels(filenames, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, size in zip(bars, filesizes):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                                f'{size:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_configuration_summary(self) -> None:
        """Create a summary of simulation configurations."""
        if not self.metadata:
            print("No configuration metadata found in datasets.")
            return
            
        print("\n" + "="*60)
        print("SIMULATION CONFIGURATION SUMMARY")
        print("="*60)
        
        config_data = []
        for filename, config in self.metadata.items():
            row = {'File': filename.replace('state_history_', '').replace('.nc', '')}
            
            # Extract key parameters
            if 'profile_conditions' in config:
                profile = config['profile_conditions']
                for param in ['Ip', 'P', 'Ic']:
                    if param in profile and 'value' in profile[param]:
                        value = profile[param]['value']
                        # Handle list format: ['float64', [value]]
                        if isinstance(value, list) and len(value) == 2:
                            if isinstance(value[1], list) and len(value[1]) > 0:
                                row[param] = value[1][0]  # Extract actual value
                            else:
                                row[param] = value[1]
                        else:
                            row[param] = value
            
            # Extract timing info from filename
            timestamp = filename.split('_')[-1].replace('.nc', '')
            if len(timestamp) == 14:  # YYYYMMDD_HHMMSS format
                date_part = timestamp[:8]
                time_part = timestamp[8:]
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                row['Time'] = formatted_time
            
            config_data.append(row)
        
        if config_data:
            df = pd.DataFrame(config_data)
            print(df.to_string(index=False))
            
            # Save to CSV in the torax directory
            csv_path = Path(__file__).parent / 'simulation_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"\nConfiguration summary saved to: {csv_path}")
    
    def generate_report(self, save_plots: bool = True) -> None:
        """
        Generate a comprehensive analysis report.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("="*60)
        print("TORAX SIMULATION ANALYSIS REPORT")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        
        # Load and analyze datasets
        self.load_datasets()
        self.examine_structure()
        self.create_configuration_summary()
        
        # Create visualizations
        if self.datasets:
            print(f"\nGenerating coordinate visualization...")
            fig = self.create_coordinate_plots()
            
            if save_plots:
                plot_path = Path(__file__).parent / 'coordinate_analysis.png'
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
            
            plt.show()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        if not self.datasets:
            print("⚠️  No valid datasets found.")
        else:
            print(f"✓ Analyzed {len(self.datasets)} simulation files")
            
        if not any(ds.data_vars for ds in self.datasets.values()):
            print("ℹ️  Note: NetCDF files contain coordinate grids but no physics data variables.")
            print("   This may be expected if TORAX stores results in a different format")
            print("   or if simulations terminated early.")

def main():
    """Main function to run the analysis."""
    # Create analyzer instance
    analyzer = ToraxAnalyzer()
    
    # Generate comprehensive report
    analyzer.generate_report(save_plots=True)
    
    print("\nFor more detailed analysis, you can use the analyzer object:")
    print("  analyzer = ToraxAnalyzer()")
    print("  analyzer.load_datasets()")
    print("  analyzer.examine_structure()")

if __name__ == "__main__":
    main()