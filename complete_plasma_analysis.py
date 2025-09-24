#!/usr/bin/env python3
"""
Complete Plasma Physics Analysis - Clean Version

This script performs Steps 2 and 3 of the plasma analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import pickle
import json
from datetime import datetime

class CompletePlasmaAnalyzer:
    """Complete plasma physics analyzer with surrogate modeling."""
    
    def __init__(self, netcdf_file='data/synthetic_complete_physics.nc'):
        """Initialize with physics data file."""
        self.physics_file = netcdf_file
        self.ds = None
        self.analysis_results = {}
        
        # Create output directories
        self.analysis_dir = Path('outputs')
        self.surrogate_dir = Path('models')
        
        for dir_path in [self.analysis_dir, self.surrogate_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_physics_data(self):
        """Load and verify physics data."""
        
        print("LOADING COMPLETE PHYSICS DATA")
        print("="*50)
        
        try:
            self.ds = xr.open_dataset(self.physics_file)
            print(f"✓ Loaded: {self.physics_file}")
            print(f"  Variables: {len(self.ds.data_vars)}")
            print(f"  Time points: {len(self.ds.time)}")
            print(f"  Time range: {float(self.ds.time[0]):.2f} - {float(self.ds.time[-1]):.2f} s")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def step2_analyze_plasma_dynamics(self):
        """Step 2: Comprehensive analysis of plasma dynamics."""
        
        print("STEP 2: ANALYZING PLASMA DYNAMICS")
        print("="*50)
        
        if self.ds is None:
            print("Error: No physics data loaded")
            return False
        
        # Extract key data
        time = self.ds.time.values
        rho = self.ds.rho_cell_norm.values
        
        # Time evolution analysis
        self._analyze_time_evolution(time)
        
        # Profile analysis
        self._analyze_radial_profiles(time, rho)
        
        # Correlation analysis
        self._analyze_correlations()
        
        # Create plots
        self._create_analysis_plots()
        
        print("✓ Step 2 complete: Plasma dynamics analyzed")
        return True
    
    def _analyze_time_evolution(self, time):
        """Analyze time evolution of key parameters."""
        
        print("Analyzing time evolution...")
        
        # Extract time series
        Te_avg = self.ds.temp_el.mean(dim='rho_cell_norm')
        Ti_avg = self.ds.temp_ion.mean(dim='rho_cell_norm')
        ne_avg = self.ds.ne.mean(dim='rho_cell_norm')
        
        # Coil currents
        coil_currents = np.column_stack([
            self.ds[f'coil_current_{i}'].values for i in range(1, 5)
        ])
        
        # Store results
        self.analysis_results['time_evolution'] = {
            'time': time,
            'Te_avg': Te_avg.values,
            'Ti_avg': Ti_avg.values,
            'ne_avg': ne_avg.values,
            'Ip': self.ds.Ip.values,
            'elongation': self.ds.elongation.values,
            'triangularity': self.ds.triangularity.values,
            'R_centroid': self.ds.R_centroid.values,
            'Z_centroid': self.ds.Z_centroid.values,
            'coil_currents': coil_currents,
        }
        
        print(f"  Analyzed time evolution for {len(time)} time points")
    
    def _analyze_radial_profiles(self, time, rho):
        """Analyze radial profiles."""
        
        print("Analyzing radial profiles...")
        
        # Select 5 representative time slices
        n_slices = 5
        time_indices = np.linspace(0, len(time)-1, n_slices, dtype=int)
        
        profiles = {}
        for t_idx in time_indices:
            t_val = time[t_idx]
            
            profiles[f't_{t_val:.1f}s'] = {
                'time': t_val,
                'Te': self.ds.temp_el[t_idx, :].values,
                'Ti': self.ds.temp_ion[t_idx, :].values,
                'ne': self.ds.ne[t_idx, :].values,
                'q': self.ds.q[t_idx, :].values,
            }
        
        self.analysis_results['radial_profiles'] = {
            'rho': rho,
            'profiles': profiles
        }
        
        print(f"  Analyzed profiles at {len(profiles)} time slices")
    
    def _analyze_correlations(self):
        """Analyze correlations between control and response variables."""
        
        print("Analyzing control-response correlations...")
        
        # Control data
        control_data = {
            'coil_1': self.ds.coil_current_1.values,
            'coil_2': self.ds.coil_current_2.values,
            'coil_3': self.ds.coil_current_3.values,
            'coil_4': self.ds.coil_current_4.values,
        }
        
        # Response data
        response_data = {
            'R_centroid': self.ds.R_centroid.values,
            'Z_centroid': self.ds.Z_centroid.values,
            'elongation': self.ds.elongation.values,
            'triangularity': self.ds.triangularity.values,
            'Te_avg': self.ds.temp_el.mean(dim='rho_cell_norm').values,
            'Ip': self.ds.Ip.values,
        }
        
        # Calculate correlation matrix
        all_data = {**control_data, **response_data}
        df = pd.DataFrame(all_data)
        correlation_matrix = df.corr()
        
        # Extract control-response correlations
        control_response_corr = {}
        for control in control_data.keys():
            control_response_corr[control] = {}
            for response in response_data.keys():
                corr_val = correlation_matrix.loc[control, response]
                control_response_corr[control][response] = corr_val
        
        self.analysis_results['correlations'] = {
            'full_matrix': correlation_matrix,
            'control_response': control_response_corr
        }
        
        print(f"  Calculated correlations for {len(control_data)} controls and {len(response_data)} responses")
    
    def _create_analysis_plots(self):
        """Create comprehensive analysis visualizations."""
        
        print("Creating analysis plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        time = self.analysis_results['time_evolution']['time']
        
        # Row 1: Time evolution plots
        # Temperature evolution
        axes[0,0].plot(time, self.analysis_results['time_evolution']['Te_avg'], 'r-', label='Te avg', linewidth=2)
        axes[0,0].plot(time, self.analysis_results['time_evolution']['Ti_avg'], 'b-', label='Ti avg', linewidth=2)
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Temperature (keV)')
        axes[0,0].set_title('Temperature Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plasma current
        axes[0,1].plot(time, self.analysis_results['time_evolution']['Ip'], 'g-', linewidth=2)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Plasma Current (MA)')
        axes[0,1].set_title('Plasma Current Evolution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Shape metrics
        axes[0,2].plot(time, self.analysis_results['time_evolution']['elongation'], 'purple', linewidth=2, label='κ')
        axes[0,2].plot(time, self.analysis_results['time_evolution']['triangularity'], 'brown', linewidth=2, label='δ')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Shape Metrics')
        axes[0,2].set_title('Shape Evolution')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Centroid positions
        axes[0,3].plot(time, self.analysis_results['time_evolution']['R_centroid'], 'red', linewidth=2, label='R')
        axes[0,3].plot(time, self.analysis_results['time_evolution']['Z_centroid'], 'blue', linewidth=2, label='Z')
        axes[0,3].set_xlabel('Time (s)')
        axes[0,3].set_ylabel('Centroid (m)')
        axes[0,3].set_title('Centroid Position')
        axes[0,3].legend()
        axes[0,3].grid(True, alpha=0.3)
        
        # Row 2: Radial profiles
        rho = self.analysis_results['radial_profiles']['rho']
        profiles = self.analysis_results['radial_profiles']['profiles']
        
        # Temperature profiles
        for i, (time_key, profile) in enumerate(profiles.items()):
            alpha = 0.3 + 0.7 * i / (len(profiles) - 1)
            axes[1,0].plot(rho, profile['Te'], 'r-', alpha=alpha, linewidth=2, label=f'Te {time_key}')
            axes[1,0].plot(rho, profile['Ti'], 'b--', alpha=alpha, linewidth=2, label=f'Ti {time_key}')
        axes[1,0].set_xlabel('ρ')
        axes[1,0].set_ylabel('Temperature (keV)')
        axes[1,0].set_title('Temperature Profiles')
        axes[1,0].grid(True, alpha=0.3)
        
        # Density profiles
        for i, (time_key, profile) in enumerate(profiles.items()):
            alpha = 0.3 + 0.7 * i / (len(profiles) - 1)
            axes[1,1].plot(rho, profile['ne'], 'orange', alpha=alpha, linewidth=2, label=f'ne {time_key}')
        axes[1,1].set_xlabel('ρ')
        axes[1,1].set_ylabel('Density (10^19 m^-3)')
        axes[1,1].set_title('Density Profiles')
        axes[1,1].grid(True, alpha=0.3)
        
        # Safety factor profiles
        for i, (time_key, profile) in enumerate(profiles.items()):
            alpha = 0.3 + 0.7 * i / (len(profiles) - 1)
            axes[1,2].plot(rho, profile['q'], 'purple', alpha=alpha, linewidth=2, label=f'q {time_key}')
        axes[1,2].set_xlabel('ρ')
        axes[1,2].set_ylabel('Safety Factor q')
        axes[1,2].set_title('q Profiles')
        axes[1,2].grid(True, alpha=0.3)
        
        # Correlation matrix
        corr_matrix = self.analysis_results['correlations']['full_matrix']
        im = axes[1,3].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[1,3].set_xticks(range(len(corr_matrix.columns)))
        axes[1,3].set_yticks(range(len(corr_matrix.index)))
        axes[1,3].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[1,3].set_yticklabels(corr_matrix.index)
        axes[1,3].set_title('Correlation Matrix')
        
        # Row 3: Control-response analysis
        control_response = self.analysis_results['correlations']['control_response']
        
        for i, (coil_name, responses) in enumerate(control_response.items()):
            if i < 4:  # Only plot first 4 coils
                response_names = list(responses.keys())
                values = list(responses.values())
                
                axes[2,i].bar(response_names, values, alpha=0.7)
                axes[2,i].set_title(f'{coil_name.replace("_", " ").title()} Correlations')
                axes[2,i].set_ylabel('Correlation')
                axes[2,i].tick_params(axis='x', rotation=45)
                axes[2,i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Comprehensive Plasma Physics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.analysis_dir / 'comprehensive_plasma_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Analysis plot saved to: {plot_path}")
    
    def step3_create_linear_surrogate(self):
        """Step 3: Create linear surrogate model."""
        
        print("STEP 3: CREATING LINEAR SURROGATE MODEL")
        print("="*50)
        
        if self.ds is None:
            print("Error: No physics data loaded")
            return False
        
        # Prepare training data
        control_inputs, response_outputs = self._prepare_surrogate_data()
        
        # Build models
        models = self._build_linear_models(control_inputs, response_outputs)
        
        # Create interface
        self._create_surrogate_interface(models)
        
        # Generate response matrices
        self._generate_response_matrices(models)
        
        print("✓ Step 3 complete: Linear surrogate model ready")
        return True
    
    def _prepare_surrogate_data(self):
        """Prepare data for surrogate model training."""
        
        print("Preparing surrogate training data...")
        
        # Control inputs: coil currents
        control_inputs = np.column_stack([
            self.ds.coil_current_1.values,
            self.ds.coil_current_2.values,
            self.ds.coil_current_3.values,
            self.ds.coil_current_4.values
        ])
        
        # Response outputs: key observables
        response_outputs = {
            'R_centroid': self.ds.R_centroid.values,
            'Z_centroid': self.ds.Z_centroid.values,
            'elongation': self.ds.elongation.values,
            'triangularity': self.ds.triangularity.values,
            'Te_avg': self.ds.temp_el.mean(dim='rho_cell_norm').values,
            'ne_avg': self.ds.ne.mean(dim='rho_cell_norm').values,
            'Ip': self.ds.Ip.values,
            'q95': self.ds.q.isel(rho_cell_norm=-1).values,  # q at edge
        }
        
        print(f"  Control inputs: {control_inputs.shape[1]} variables, {control_inputs.shape[0]} samples")
        print(f"  Response outputs: {len(response_outputs)} variables")
        
        return control_inputs, response_outputs
    
    def _build_linear_models(self, control_inputs, response_outputs):
        """Build linear regression models."""
        
        print("Building linear regression models...")
        
        models = {}
        
        # Scale inputs
        scaler = StandardScaler()
        control_scaled = scaler.fit_transform(control_inputs)
        
        for response_name, response_data in response_outputs.items():
            # Use Ridge regression
            model = Ridge(alpha=1.0)
            model.fit(control_scaled, response_data)
            
            # Calculate R² score
            r2 = model.score(control_scaled, response_data)
            
            # Store model
            models[response_name] = {
                'model': model,
                'scaler': scaler,
                'r2_score': r2,
                'feature_names': ['coil_1', 'coil_2', 'coil_3', 'coil_4'],
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }
            
            print(f"  {response_name}: R² = {r2:.3f}")
        
        return models
    
    def _create_surrogate_interface(self, models):
        """Create surrogate model interface."""
        
        print("Creating surrogate model interface...")
        
        # Save model
        model_file = self.surrogate_dir / 'linear_surrogate_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        
        # Create interface code
        interface_code = '''import pickle
import numpy as np

class LinearPlasmaSurrogate:
    """Fast linear surrogate model for plasma control."""
    
    def __init__(self, model_path="linear_surrogate_model.pkl"):
        """Initialize surrogate model."""
        with open(model_path, 'rb') as f:
            self.models = pickle.load(f)
        
        self.control_names = ['coil_1', 'coil_2', 'coil_3', 'coil_4']
        self.response_names = list(self.models.keys())
        self.baseline_controls = np.array([10.0, 8.0, 12.0, 6.0])  # kA
    
    def predict(self, coil_currents):
        """Predict plasma responses for given coil currents.
        
        Args:
            coil_currents: Array with coil currents in kA
            
        Returns:
            Dictionary of predicted responses
        """
        coil_currents = np.array(coil_currents).reshape(1, -1)
        
        responses = {}
        for response_name, model_data in self.models.items():
            scaler = model_data['scaler']
            model = model_data['model']
            
            controls_scaled = scaler.transform(coil_currents)
            prediction = model.predict(controls_scaled)
            
            responses[response_name] = prediction[0]
        
        return responses
    
    def get_response_matrix(self, perturbation=0.1):
        """Get linear response matrix."""
        baseline_response = self.predict(self.baseline_controls)
        
        response_matrix = np.zeros((len(self.response_names), len(self.control_names)))
        
        for i, control_name in enumerate(self.control_names):
            perturbed_controls = self.baseline_controls.copy()
            perturbed_controls[i] += perturbation
            
            perturbed_response = self.predict(perturbed_controls)
            
            for j, response_name in enumerate(self.response_names):
                delta_response = perturbed_response[response_name] - baseline_response[response_name]
                sensitivity = delta_response / perturbation
                response_matrix[j, i] = sensitivity
        
        return response_matrix

# Example usage:
# surrogate = LinearPlasmaSurrogate()
# responses = surrogate.predict([10.5, 8.2, 12.1, 6.3])
# response_matrix = surrogate.get_response_matrix()
'''
        
        interface_file = self.surrogate_dir / 'linear_plasma_surrogate.py'
        with open(interface_file, 'w') as f:
            f.write(interface_code)
        
        print(f"  Model saved to: {model_file}")
        print(f"  Interface saved to: {interface_file}")
    
    def _generate_response_matrices(self, models):
        """Generate response matrices."""
        
        print("Generating response matrices...")
        
        # Calculate response matrix
        baseline_controls = np.array([10.0, 8.0, 12.0, 6.0])  # kA
        perturbation = 0.1  # kA
        
        response_names = list(models.keys())
        control_names = ['coil_1', 'coil_2', 'coil_3', 'coil_4']
        
        response_matrix = np.zeros((len(response_names), len(control_names)))
        
        # Calculate baseline responses
        baseline_scaled = models[response_names[0]]['scaler'].transform([baseline_controls])
        baseline_responses = {}
        for response_name in response_names:
            model = models[response_name]['model']
            baseline_responses[response_name] = model.predict(baseline_scaled)[0]
        
        # Calculate sensitivities
        for i, control_name in enumerate(control_names):
            perturbed_controls = baseline_controls.copy()
            perturbed_controls[i] += perturbation
            perturbed_scaled = models[response_names[0]]['scaler'].transform([perturbed_controls])
            
            for j, response_name in enumerate(response_names):
                model = models[response_name]['model']
                perturbed_response = model.predict(perturbed_scaled)[0]
                sensitivity = (perturbed_response - baseline_responses[response_name]) / perturbation
                response_matrix[j, i] = sensitivity
        
        # Save response matrix
        matrix_data = {
            'response_matrix': response_matrix.tolist(),
            'response_names': response_names,
            'control_names': control_names,
            'baseline_controls': baseline_controls.tolist(),
            'perturbation_size': perturbation
        }
        
        matrix_file = self.surrogate_dir / 'response_matrices.json'
        with open(matrix_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        im = plt.imshow(response_matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, label='Response Sensitivity')
        plt.xlabel('Control Inputs (Coils)')
        plt.ylabel('Response Outputs')
        plt.title('Linear Response Matrix')
        plt.xticks(range(len(control_names)), control_names)
        plt.yticks(range(len(response_names)), response_names)
        
        # Add values as text
        for i in range(len(response_names)):
            for j in range(len(control_names)):
                plt.text(j, i, f'{response_matrix[i, j]:.3f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        matrix_plot_file = self.surrogate_dir / 'response_matrix_visualization.png'
        plt.savefig(matrix_plot_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Response matrix saved to: {matrix_file}")
        print(f"  Matrix plot saved to: {matrix_plot_file}")
        
        # Print summary
        print("  Response Matrix Summary:")
        for i, response in enumerate(response_names):
            max_sensitivity_idx = np.argmax(np.abs(response_matrix[i, :]))
            max_sensitivity = response_matrix[i, max_sensitivity_idx]
            controlling_coil = control_names[max_sensitivity_idx]
            print(f"    {response}: Most sensitive to {controlling_coil} ({max_sensitivity:.3f})")
    
    def generate_final_report(self):
        """Generate final report."""
        
        print("GENERATING FINAL ANALYSIS REPORT")
        print("="*50)
        
        report_path = self.analysis_dir / 'complete_physics_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Complete Plasma Physics Analysis Report\\n\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Step 2: Plasma Dynamics Analysis\\n\\n")
            f.write(f"**Data Source**: {self.physics_file}\\n")
            f.write(f"**Time Range**: {float(self.ds.time[0]):.2f} - {float(self.ds.time[-1]):.2f} s\\n")
            f.write(f"**Time Points**: {len(self.ds.time)}\\n")
            f.write(f"**Radial Points**: {len(self.ds.rho_cell_norm)}\\n\\n")
            
            f.write("### Key Variables Analyzed:\\n")
            f.write("- Electron temperature Te(t)\\n")
            f.write("- Ion temperature Ti(t)\\n") 
            f.write("- Electron density ne(t)\\n")
            f.write("- Plasma current Ip(t)\\n")
            f.write("- Shape metrics (elongation, triangularity)\\n")
            f.write("- Centroid positions (R, Z)\\n")
            f.write("- Safety factor q(rho, t)\\n")
            f.write("- Coil currents (control inputs)\\n\\n")
            
            f.write("## Step 3: Linear Surrogate Model\\n\\n")
            f.write("**Model Type**: Ridge Linear Regression\\n")
            f.write("**Control Inputs**: 4 coil currents (kA)\\n")
            f.write("**Response Outputs**: 8 plasma observables\\n\\n")
            
            f.write("### Deliverables:\\n")
            f.write(f"- **Analysis plots**: `{self.analysis_dir}/comprehensive_plasma_analysis.png`\\n")
            f.write(f"- **Surrogate model**: `{self.surrogate_dir}/linear_surrogate_model.pkl`\\n")
            f.write(f"- **Model interface**: `{self.surrogate_dir}/linear_plasma_surrogate.py`\\n")
            f.write(f"- **Response matrices**: `{self.surrogate_dir}/response_matrices.json`\\n")
            f.write(f"- **Matrix visualization**: `{self.surrogate_dir}/response_matrix_visualization.png`\\n\\n")
            
            f.write("### RL Integration Ready:\\n")
            f.write("```python\\n")
            f.write("from linear_plasma_surrogate import LinearPlasmaSurrogate\\n\\n")
            f.write("surrogate = LinearPlasmaSurrogate()\\n")
            f.write("responses = surrogate.predict([10.5, 8.2, 12.1, 6.3])\\n")
            f.write("response_matrix = surrogate.get_response_matrix()\\n")
            f.write("```\\n")
        
        print(f"Final report saved to: {report_path}")
        return report_path

def main():
    """Main analysis workflow."""
    
    print("COMPLETE PLASMA PHYSICS ANALYSIS WORKFLOW")
    print("="*60)
    
    # Initialize analyzer
    analyzer = CompletePlasmaAnalyzer()
    
    # Load physics data
    if not analyzer.load_physics_data():
        print("Failed to load physics data")
        return
    
    # Step 2: Analyze plasma dynamics
    if not analyzer.step2_analyze_plasma_dynamics():
        print("Failed to complete Step 2")
        return
    
    # Step 3: Create linear surrogate
    if not analyzer.step3_create_linear_surrogate():
        print("Failed to complete Step 3")
        return
    
    # Generate final report
    report_path = analyzer.generate_final_report()
    
    print("\\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\\nDeliverables:")
    print(f"- Comprehensive analysis: outputs/")
    print(f"- Linear surrogate model: models/")
    print(f"- Final report: {report_path}")
    print("\\nSurrogate model ready for RL integration!")

if __name__ == "__main__":
    main()