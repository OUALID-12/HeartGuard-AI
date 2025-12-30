import os
import joblib
import pandas as pd
import numpy as np

class BodyFatPredictor:
    def __init__(self, model_name='bodyfat_regressor.pkl'):
        self.model_path = os.path.join(os.path.dirname(__file__), 'saved_models', model_name)
        self.artifacts = None
        self.model = None
        self.feature_names = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Body fat model not found at {self.model_path}")
        self.artifacts = joblib.load(self.model_path)
        self.model = self.artifacts['model']
        self.feature_names = self.artifacts['feature_names']

    def _prepare_input(self, input_dict):
        df = pd.DataFrame([input_dict])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]
        return df

    def predict(self, input_dict):
        if self.model is None:
            self.load()
        df = self._prepare_input(input_dict)
        # model is a pipeline including preprocessing
        pred = float(self.model.predict(df)[0])
        # approximate uncertainty from estimator predictions if available
        uncertainty = None
        if hasattr(self.model.named_steps.get('rf'), 'estimators_') and hasattr(self.model.named_steps['rf'], 'estimators_'):
            try:
                est_preds = np.array([est.predict(self.model.named_steps['rf']._validate_X_predict(df)) for est in self.model.named_steps['rf'].estimators_])
                uncertainty = float(est_preds.std())
            except Exception:
                uncertainty = None
        return {'bodyfat_percent': pred, 'uncertainty': uncertainty}
