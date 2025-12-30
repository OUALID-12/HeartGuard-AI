
import joblib
import os
import pandas as pd
import numpy as np

class HeartDiseasePredictor:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'heart_disease_model.pkl')
        self.artifacts = None
        self.model = None
        self.scaler = None
        self.encoders = None
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        self.artifacts = joblib.load(self.model_path)
        self.model = self.artifacts['model']
        self.scaler = self.artifacts['scaler']
        self.encoders = self.artifacts['encoders']
        self.feature_names = self.artifacts['feature_names']
        
    def predict(self, patient_data):
        if self.model is None:
            self.load_model()
            
        # Create DataFrame from input data
        # Ensure keys match what the model expects (lowercase)
        input_data = {k.lower(): v for k, v in patient_data.items()}
        
        # Create DataFrame with correct columns
        df = pd.DataFrame([input_data])
        
        # Ensure all columns exist
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing feature: {col}")
                
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Encode categorical variables
        for col, le in self.encoders.items():
            if col in df.columns:
                # Handle unknown labels if necessary, but for now assume valid input from forms
                # Using map might be safer than transform for single values
                # But let's use transform and handle exceptions if needed
                 df[col] = le.transform(df[col])
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        return prediction, probability
