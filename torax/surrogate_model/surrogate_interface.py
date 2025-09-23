import pickle
import numpy as np

class PlasmaControlSurrogate:
    def __init__(self, model_path="surrogate_model.pkl"):
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        self.feature_names = self.model_data['feature_names']
        self.response_names = self.model_data['response_names']
    
    def predict(self, control_inputs):
        """Predict plasma responses for control inputs [Ip_MA, P_MW, B_0]."""
        control_inputs = np.array(control_inputs).reshape(1, -1)
        responses = {}
        
        for response_name in self.response_names:
            scaler = self.model_data['scalers'][response_name]
            model = self.model_data['models'][response_name]
            control_scaled = scaler.transform(control_inputs)
            prediction = model.predict(control_scaled)[0]
            responses[response_name] = prediction
        
        return responses

# Example usage:
# surrogate = PlasmaControlSurrogate()
# responses = surrogate.predict([15.0, 50.0, 5.3])
# print(responses)
