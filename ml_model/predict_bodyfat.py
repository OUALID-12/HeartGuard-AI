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
        # Accept derived features if provided (BMI, WHR) or compute them lazily
        d = input_dict.copy()
        try:
            if 'Weight' in d and 'Height' in d and ('BMI' not in d or d.get('BMI') is None):
                # Weight (lbs) and Height (inches) -> BMI
                wkg = float(d['Weight']) * 0.45359237
                hm = float(d['Height']) * 0.0254
                d['BMI'] = wkg / (hm ** 2)
        except Exception:
            pass
        try:
            if 'Abdomen' in d and 'Hip' in d and ('WHR' not in d or d.get('WHR') is None):
                d['WHR'] = float(d['Abdomen']) / float(d['Hip'])
        except Exception:
            pass

        df = pd.DataFrame([d])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_names]
        return df

    def predict(self, input_dict):
        if self.model is None:
            self.load()

        # Check for density-based direct conversion (Siri's equation)
        density = None
        # Accept different key names
        for k in ['Density', 'density', 'Density (gm/cm^3)']:
            if k in input_dict and input_dict.get(k) not in (None, ''):
                try:
                    density = float(input_dict.get(k))
                    break
                except Exception:
                    density = None

        siri_result = None
        if density is not None and density > 0:
            try:
                siri_percent = 495.0 / density - 450.0
                siri_result = float(siri_percent)
            except Exception:
                siri_result = None

        df = self._prepare_input(input_dict)
        # model is a pipeline including preprocessing
        model_pred = float(self.model.predict(df)[0])

        # approximate uncertainty from estimator predictions if available (for RandomForest)
        uncertainty = None
        try:
            rf = self.model.named_steps.get('rf')
            if rf is not None and hasattr(rf, 'estimators_'):
                est_preds = np.array([est.predict(rf._validate_X_predict(df)) for est in rf.estimators_])
                uncertainty = float(est_preds.std())
        except Exception:
            uncertainty = None

        result = {
            'bodyfat_percent_model': model_pred,
            'uncertainty': uncertainty,
            'model_version': self.artifacts.get('model_version') if self.artifacts else None
        }
        if siri_result is not None:
            result['bodyfat_percent_siri'] = siri_result
            # Provide a simple decision hint if both available
            try:
                result['consensus_delta'] = abs(model_pred - siri_result)
            except Exception:
                result['consensus_delta'] = None

        # If user requested explanation, include feature importances (if available)
        explain = input_dict.get('explain') or input_dict.get('_explain')
        if explain and self.artifacts:
            fi = self.artifacts.get('feature_importances')
            if fi:
                # Return top 5 contributors
                sorted_feats = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
                result['top_features'] = sorted_feats

        return result
