import pickle
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
