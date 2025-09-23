#!/usr/bin/env python3
"""
Quick test of parameter sweep with 2 values.
"""

import os
import subprocess
import csv
import time
import json


def create_config_file(Ip_MA: float, P_MW: float, filename: str):
    """Create a complete config file with specified parameters."""
    
    config_content = f'''# Auto-generated config for parameter sweep
CONFIG = {{
    'profile_conditions': {{
        'Ip': {Ip_MA * 1e6},
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
    'plasma_composition': {{'main_ion': {{'D': 0.5, 'T': 0.5}}, 'Z_eff': 1.8}},
    'numerics': {{'t_final': 2.0, 'fixed_dt': 0.1}},
    'geometry': {{'geometry_type': 'circular', 'R_major': 6.2, 'a_minor': 2.0, 'B_0': 5.3, 'elongation_LCFS': 1.72}},
    'neoclassical': {{'bootstrap_current': {{}}}},
    'sources': {{
        'generic_current': {{}}, 'generic_particle': {{}}, 'gas_puff': {{}}, 'pellet': {{}},
        'generic_heat': {{'gaussian_location': 0.2, 'gaussian_width': 0.1, 'P_total': {P_MW * 1e6}, 'electron_heat_fraction': 0.6}},
        'fusion': {{}}, 'ei_exchange': {{}}, 'ohmic': {{}},
    }},
    'pedestal': {{}},
    'transport': {{'model_name': 'constant'}},
    'solver': {{'solver_type': 'linear'}},
    'time_step_calculator': {{'calculator_type': 'chi'}},
}}'''
    
    with open(filename, 'w') as f:
        f.write(config_content)


def quick_test():
    """Quick test with 2 power values."""
    
    P_values = [30, 50]  # Just 2 test values
    Ip_MA = 15.0
    
    output_dir = os.path.join(os.getcwd(), "quick_test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Quick Test: P = {P_values} MW, Ip = {Ip_MA} MA")
    print()
    
    results = []
    
    for i, P_MW in enumerate(P_values, 1):
        print(f"[{i}/{len(P_values)}] Testing P = {P_MW} MW...")
        
        config_filename = os.path.join(output_dir, f"test_config_P{P_MW}.py")
        create_config_file(Ip_MA, P_MW, config_filename)
        
        try:
            python_exe = r"C:/Users/ashwa/Desktop/plasma_reactor/.venv/Scripts/python.exe"
            cmd = [python_exe, "torax/run_simulation_main.py", "--config", config_filename, "--quit"]
            
            start_time = time.time()
            result = subprocess.run(cmd, input="q\\n", capture_output=True, text=True, timeout=90)
            runtime = time.time() - start_time
            
            success = result.returncode == 0
            
            output_file = None
            if success and "Wrote simulation output to" in result.stderr:
                for line in result.stderr.split('\\n'):
                    if "Wrote simulation output to" in line:
                        output_file = line.split("Wrote simulation output to")[-1].strip()
                        break
            
            results.append({
                'Ip_MA': Ip_MA, 'P_MW': P_MW, 'Ic_MA': 0.0,
                'success': success, 'runtime_seconds': round(runtime, 1),
                'output_file': output_file
            })
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {status} ({runtime:.1f}s)")
            if output_file:
                print(f"    Output: {os.path.basename(output_file)}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append({'Ip_MA': Ip_MA, 'P_MW': P_MW, 'Ic_MA': 0.0, 'success': False, 'error': str(e)})
    
    # Save test results
    csv_file = os.path.join(output_dir, "test_results.csv")
    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    successful = [r for r in results if r['success']]
    print(f"\\nTest Results: {len(successful)}/{len(results)} successful")
    
    if successful:
        print("\\nSUCCESS! Parameter sweep mechanism works.")
        print("Ready to run full sweep with 10 power values.")
        return True
    else:
        print("\\nFAILED! Need to debug the simulation issues.")
        return False


if __name__ == "__main__":
    quick_test()