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
        # Backwards compatibility: older artifacts may include scaler/encoders
        self.scaler = self.artifacts.get('scaler')
        self.encoders = self.artifacts.get('encoders', {})
        self.target_le = self.artifacts['target_le']
        self.feature_names = self.artifacts['feature_names']

    def _prepare_input(self, input_dict):
        # Normalize keys to expected names
        df = pd.DataFrame([input_dict])
        # Fill missing columns with NaN (preprocessor will handle imputing)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]

        # If we have old-style encoders, perform mapping; otherwise leave raw values for pipeline preprocessing
        if self.encoders:
            for col, le in self.encoders.items():
                if col in df.columns:
                    val = df.loc[0, col]
                    if pd.isna(val):
                        df.loc[0, col] = -1
                    else:
                        try:
                            df.loc[0, col] = le.transform([str(val)])[0]
                        except Exception:
                            classes = [c.lower() for c in le.classes_]
                            try:
                                idx = classes.index(str(val).lower())
                                df.loc[0, col] = idx
                            except ValueError:
                                df.loc[0, col] = -1

        return df

    def recommend(self, input_dict, top_k=1):
        if self.model is None:
            self.load()

        df = self._prepare_input(input_dict)

        X = df.copy()

        # If we have a scaler saved separately (old artifact), apply it, otherwise pass raw X to pipeline
        if self.scaler is not None and not hasattr(self.model, 'named_steps'):
            # old behavior: scale numeric columns
            X_scaled = self.scaler.transform(X)
            probs = self.model.predict_proba(X_scaled)[0]
        else:
            # The model is a pipeline that includes preprocessing
            probs = self.model.predict_proba(X)[0]

        # Ensure classes mapping aligns with probabilities
        # If target_le maps class indices, rebuild ordered class names
        classes = self.target_le.inverse_transform(np.arange(len(probs)))

        # Get top k
        idx_sorted = np.argsort(probs)[::-1]
        top_idx = idx_sorted[:top_k]
        results = [(classes[i], float(probs[i])) for i in top_idx]
        return results
