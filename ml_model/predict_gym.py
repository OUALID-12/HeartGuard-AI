import os
import joblib
import pandas as pd
import numpy as np

class GymRecommender:
    def __init__(self, model_name='gym_recommender.pkl'):
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_name)
        self.artifacts = None
        self.model = None
        self.scaler = None
        self.encoders = None
        self.target_le = None
        self.feature_names = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Gym recommender not found at {self.model_path}")
        self.artifacts = joblib.load(self.model_path)
        self.model = self.artifacts['model']
        self.scaler = self.artifacts['scaler']
        self.encoders = self.artifacts.get('encoders', {})
        self.target_le = self.artifacts['target_le']
        self.feature_names = self.artifacts['feature_names']

    def _prepare_input(self, input_dict):
        # Normalize keys to expected names
        df = pd.DataFrame([input_dict])
        # Fill missing columns with NaN
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]

        # Encode categorical using stored encoders
        for col, le in self.encoders.items():
            if col in df.columns:
                val = df.loc[0, col]
                if pd.isna(val):
                    df.loc[0, col] = -1
                else:
                    # perform mapping robustly
                    try:
                        df.loc[0, col] = le.transform([str(val)])[0]
                    except Exception:
                        # Unknown category: try to map by case-insensitive match
                        classes = [c.lower() for c in le.classes_]
                        try:
                            idx = classes.index(str(val).lower())
                            df.loc[0, col] = idx
                        except ValueError:
                            # fallback to -1
                            df.loc[0, col] = -1
        # Fill numeric NA with median (simple rule)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if pd.isna(df.loc[0, col]):
                df.loc[0, col] = 0

        return df

    def recommend(self, input_dict, top_k=1):
        if self.model is None:
            self.load()

        df = self._prepare_input(input_dict)

        X = df.copy()
        # Scale features
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[0]
        classes = self.target_le.inverse_transform(np.arange(len(probs)))

        # Get top k
        idx_sorted = np.argsort(probs)[::-1]
        top_idx = idx_sorted[:top_k]
        results = [(classes[i], float(probs[i])) for i in top_idx]
        return results
