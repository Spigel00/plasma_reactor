#!/usr/bin/env python3
"""
Final working parameter sweep with corrected paths.
"""

import os
import subprocess
import csv
import time
import json


def create_config_file(Ip_MA: float, P_MW: float, filename: str):
    """Create a complete config file with specified parameters."""
    
    config_content = f'''# Auto-generated config for parameter sweep
# Ip = {Ip_MA} MA, P = {P_MW} MW

CONFIG = {{
    'profile_conditions': {{
        'Ip': {Ip_MA * 1e6},  # {Ip_MA} MA in Amperes
        'T_i': {{0.0: {{0.0: 8.0, 1.0: 0.2}}}},
        'T_i_right_bc': 0.2,
        'T_e': {{0.0: {{0.0: 8.0, 1.0: 0.2}}}},
        'T_e_right_bc': 0.2,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': 0.3,
        'n_e_nbar_is_fGW': True,
        'nbar': 1.0,
        'n_e': {{0: {{0.0: 1.2, 1.0: 1.0}}}},
    }},
    'plasma_composition': {{
        'main_ion': {{'D': 0.5, 'T': 0.5}},
        'Z_eff': 1.8,
    }},
    'numerics': {{
        't_final': 3.0,  # Shorter simulation for parameter sweep
        'fixed_dt': 0.1,
    }},
    'geometry': {{
        'geometry_type': 'circular',
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
        'elongation_LCFS': 1.72,
    }},
    'neoclassical': {{
        'bootstrap_current': {{}},
    }},
    'sources': {{
        'generic_current': {{}},
        'generic_particle': {{}},
        'gas_puff': {{}},
        'pellet': {{}},
        'generic_heat': {{
            'gaussian_location': 0.2,
            'gaussian_width': 0.1,
            'P_total': {P_MW * 1e6},  # {P_MW} MW in Watts
            'electron_heat_fraction': 0.6,
        }},
        'fusion': {{}},
        'ei_exchange': {{}},
        'ohmic': {{}},
    }},
    'pedestal': {{}},
    'transport': {{
        'model_name': 'constant',
    }},
    'solver': {{
        'solver_type': 'linear',
    }},
    'time_step_calculator': {{
        'calculator_type': 'chi',
    }},
}}
'''
    
    with open(filename, 'w') as f:
        f.write(config_content)


def run_parameter_sweep():
    """Run the parameter sweep with corrected paths."""
    
    # Full parameter range as requested: 10 to 100 MW in steps of 10
    P_values = list(range(10, 101, 10))  # [10, 20, 30, ..., 100] MW
    Ip_MA = 15.0
    Ic_MA = 0.0  # Placeholder for coil current
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torax_dir = os.path.dirname(script_dir)  # Parent directory of torax
    output_dir = os.path.join(torax_dir, "torax_parameter_sweep_final")
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    python_exe = r"C:/Users/ashwa/Desktop/plasma_reactor/.venv/Scripts/python.exe"
    run_simulation_script = os.path.join(script_dir, "run_simulation_main.py")
    
    print(f"TORAX Parameter Sweep")
    print(f"=" * 40)
    print(f"Power range: {min(P_values)}-{max(P_values)} MW (step: {P_values[1]-P_values[0]} MW)")
    print(f"Fixed Ip: {Ip_MA} MA")
    print(f"Fixed Ic: {Ic_MA} MA (placeholder)")
    print(f"Total simulations: {len(P_values)}")
    print(f"Output directory: {output_dir}")
    print(f"Python executable: {python_exe}")
    print(f"Simulation script: {run_simulation_script}")
    print()
    
    results = []
    
    for i, P_MW in enumerate(P_values, 1):
        print(f"[{i:2d}/{len(P_values)}] P = {P_MW:3d} MW...", end=" ")
        
        # Create config file
        config_filename = os.path.join(output_dir, f"config_Ip{Ip_MA:.0f}_P{P_MW:03d}.py")
        create_config_file(Ip_MA, P_MW, config_filename)
        
        try:
            # Run simulation from the torax directory
            cmd = [
                python_exe,
                run_simulation_script,
                "--config", config_filename,
                "--quit"
            ]
            
            start_time = time.time()
            
            # Run from the torax directory
            result = subprocess.run(
                cmd,
                input="q\\n",
                capture_output=True,
                text=True,
                timeout=120,
                cwd=script_dir  # Set working directory to torax directory
            )
            
            runtime = time.time() - start_time
            success = result.returncode == 0
            
            # Extract output file path
            output_file = None
            if success and "Wrote simulation output to" in result.stderr:
                for line in result.stderr.split('\\n'):
                    if "Wrote simulation output to" in line:
                        output_file = line.split("Wrote simulation output to")[-1].strip()
                        break
            
            # Store results
            result_dict = {
                'Ip_MA': Ip_MA,
                'P_MW': P_MW,
                'Ic_MA': Ic_MA,
                'success': success,
                'runtime_seconds': round(runtime, 1),
                'output_file': output_file,
                'config_file': os.path.basename(config_filename)
            }
            
            results.append(result_dict)
            
            if success:
                print(f"✓ SUCCESS ({runtime:.1f}s)")
            else:
                print(f"✗ FAILED ({runtime:.1f}s) - Code {result.returncode}")
                # Print some error details for debugging
                if result.stderr:
                    error_lines = result.stderr.split('\\n')[:2]
                    for line in error_lines:
                        if line.strip() and "CRITICAL" not in line and "INFO" not in line:
                            print(f"    Error: {line.strip()}")
        
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT (>2min)")
            results.append({
                'Ip_MA': Ip_MA,
                'P_MW': P_MW,
                'Ic_MA': Ic_MA,
                'success': False,
                'error': 'Timeout > 2 minutes',
                'config_file': os.path.basename(config_filename)
            })
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append({
                'Ip_MA': Ip_MA,
                'P_MW': P_MW,
                'Ic_MA': Ic_MA,
                'success': False,
                'error': str(e),
                'config_file': os.path.basename(config_filename)
            })
    
    # Save results
    csv_file = os.path.join(output_dir, "parameter_sweep_results.csv")
    json_file = os.path.join(output_dir, "parameter_sweep_results.json")
    
    # Write CSV with the requested format
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['Ip_MA', 'Ic_MA', 'P_MW', 'success', 'runtime_seconds', 'output_file', 'config_file']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Write detailed JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print()
    print("=" * 70)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 70)
    print(f"Total simulations: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
    print()
    
    if successful:
        print("SUCCESSFUL SIMULATIONS:")
        print("Ip [MA] | Ic [MA] | P [MW] | Runtime [s] | Output File")
        print("-" * 65)
        for r in successful:
            output_name = os.path.basename(r['output_file']) if r.get('output_file') else 'N/A'
            runtime_str = f"{r.get('runtime_seconds', 0):.1f}s"
            print(f"{r['Ip_MA']:7.1f} | {r['Ic_MA']:7.1f} | {r['P_MW']:6.1f} | {runtime_str:11} | {output_name}")
    
    if failed:
        print("\\nFAILED SIMULATIONS:")
        for r in failed:
            error = r.get('error', 'Unknown error')
            print(f"  P={r['P_MW']:3.0f} MW: {error}")
    
    print(f"\\nResults saved to:")
    print(f"  📊 CSV file: {csv_file}")
    print(f"  📄 JSON file: {json_file}")
    print(f"  📁 Config files: {output_dir}")
    
    if successful:
        avg_runtime = sum(r.get('runtime_seconds', 0) for r in successful) / len(successful)
        print(f"\\nAverage runtime per successful simulation: {avg_runtime:.1f} seconds")
    
    return csv_file, successful, failed


if __name__ == "__main__":
    try:
        print("Starting TORAX Parameter Sweep...")
        print("This will sweep power P from 10 to 100 MW in steps of 10 MW")
        print("while keeping Ip and Ic constant.")
        print()
        
        csv_file, successful, failed = run_parameter_sweep()
        
        print(f"\\n🎉 Parameter sweep completed!")
        print(f"Results table available in: {os.path.basename(csv_file)}")
        
        if successful:
            print(f"✅ {len(successful)} simulations successful")
        if failed:
            print(f"❌ {len(failed)} simulations failed")
            
    except KeyboardInterrupt:
        print(f"\\n⚠️  Parameter sweep interrupted by user")
    except Exception as e:
        print(f"\\n💥 Parameter sweep crashed: {e}")
        import traceback
        traceback.print_exc()