#!/usr/bin/env python3
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

"""
Parameter sweep script for Torax fusion simulation.

This script sweeps the heating power P from 10 to 100 MW in steps of 10 MW,
keeping plasma current Ip and other parameters constant, and logs the results.
"""

import os
import sys
import tempfile
import json
import csv
import time
from typing import Dict, Any
import numpy as np

# Add the torax directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import torax modules
from torax._src import simulation_app
from torax._src.config import build_sim
from torax.examples import my_config


def run_single_simulation(Ip_MA: float, P_MW: float, output_dir: str) -> Dict[str, Any]:
    """
    Run a single Torax simulation with specified parameters.
    
    Args:
        Ip_MA: Plasma current in MA
        P_MW: Heating power in MW
        output_dir: Directory to store output files
        
    Returns:
        Dictionary containing simulation results and parameters
    """
    
    # Get configuration with modified parameters
    config = my_config.get_config_with_params(Ip_MA=Ip_MA, P_MW=P_MW)
    
    # Build the simulation configuration
    sim = build_sim.build_sim_from_config(config)
    
    # Run the simulation
    try:
        # Create a unique output filename
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        output_file = os.path.join(output_dir, f"simulation_Ip{Ip_MA:.1f}_P{P_MW:.1f}_{timestamp}.nc")
        
        # Run simulation
        output_file_path = simulation_app.main(
            sim=sim,
            output_dir=output_dir,
        )
        
        # Extract key results from the simulation
        # For this demo, we'll return the input parameters and basic metrics
        results = {
            'Ip_MA': Ip_MA,
            'P_MW': P_MW,
            'Ic_MA': 0.0,  # Placeholder - not directly controlled in this config
            'output_file': output_file_path,
            'simulation_time': 5.0,  # From our config
            'success': True,
            'timestamp': timestamp
        }
        
        print(f"✓ Simulation completed: Ip={Ip_MA:.1f} MA, P={P_MW:.1f} MW")
        return results
        
    except Exception as e:
        print(f"✗ Simulation failed: Ip={Ip_MA:.1f} MA, P={P_MW:.1f} MW, Error: {str(e)}")
        return {
            'Ip_MA': Ip_MA,
            'P_MW': P_MW,
            'Ic_MA': 0.0,
            'output_file': None,
            'simulation_time': None,
            'success': False,
            'error': str(e),
            'timestamp': int(time.time() * 1000)
        }


def parameter_sweep(
    P_range: tuple = (10, 100, 10),
    Ip_MA: float = 15.0,
    output_dir: str = None
) -> str:
    """
    Perform parameter sweep over heating power P.
    
    Args:
        P_range: Tuple of (start, stop, step) for power sweep in MW
        Ip_MA: Fixed plasma current in MA
        output_dir: Directory for output files
        
    Returns:
        Path to the results CSV file
    """
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="torax_sweep_")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate power values
    P_start, P_stop, P_step = P_range
    P_values = np.arange(P_start, P_stop + P_step, P_step)
    
    print(f"Starting parameter sweep:")
    print(f"  Power range: {P_start}-{P_stop} MW (step: {P_step} MW)")
    print(f"  Fixed Ip: {Ip_MA} MA")
    print(f"  Output directory: {output_dir}")
    print(f"  Total simulations: {len(P_values)}")
    print(f"  Power values: {P_values}")
    print()
    
    results = []
    
    for i, P_MW in enumerate(P_values, 1):
        print(f"[{i}/{len(P_values)}] Running simulation with P = {P_MW:.1f} MW...")
        
        result = run_single_simulation(
            Ip_MA=Ip_MA,
            P_MW=float(P_MW),
            output_dir=output_dir
        )
        
        results.append(result)
        
        # Brief pause between simulations
        time.sleep(0.5)
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, "parameter_sweep_results.csv")
    
    with open(csv_file, 'w', newline='') as f:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Save results to JSON as well
    json_file = os.path.join(output_dir, "parameter_sweep_results.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("Parameter Sweep Summary:")
    print("=" * 50)
    
    successful_runs = [r for r in results if r['success']]
    failed_runs = [r for r in results if not r['success']]
    
    print(f"Total simulations: {len(results)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    print()
    
    if successful_runs:
        print("Successful Simulations:")
        print("Ip [MA] | P [MW] | Output File")
        print("-" * 40)
        for result in successful_runs:
            output_name = os.path.basename(result['output_file']) if result['output_file'] else 'N/A'
            print(f"{result['Ip_MA']:7.1f} | {result['P_MW']:6.1f} | {output_name}")
    
    if failed_runs:
        print()
        print("Failed Simulations:")
        for result in failed_runs:
            print(f"  Ip={result['Ip_MA']:.1f} MA, P={result['P_MW']:.1f} MW: {result.get('error', 'Unknown error')}")
    
    print(f"")
    print(f"Results saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    print(f"  Output directory: {output_dir}")
    
    return csv_file


def main():
    """Main function to run the parameter sweep."""
    
    # Configuration
    P_range = (10, 100, 10)  # Start, stop, step (MW)
    Ip_MA = 15.0  # Fixed plasma current (MA)
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "torax_parameter_sweep_results")
    
    try:
        csv_file = parameter_sweep(
            P_range=P_range,
            Ip_MA=Ip_MA,
            output_dir=output_dir
        )
        
        print(f"\\nParameter sweep completed successfully!")
        print(f"Results available in: {csv_file}")
        
    except Exception as e:
        print(f"\\nParameter sweep failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)