import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def train_model():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'bodyfat.csv')
    model_dir = os.path.join(base_dir, 'ml_model', 'saved_models')
    model_path = os.path.join(model_dir, 'bodyfat_regressor.pkl')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load data
    df = pd.read_csv(data_path)

    # Derived features
    # Weight (lbs) and Height (inches) -> BMI
    # Note: Dataset says Height is in inches, Weight in lbs
    df['BMI'] = (df['Weight'] * 0.45359237) / ((df['Height'] * 0.0254) ** 2)
    df['WHR'] = df['Abdomen'] / df['Hip']

    # Features and Target
    # We want to predict BodyFat without using Density
    target = 'BodyFat'
    feature_names = [
        'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 
        'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist', 'BMI', 'WHR'
    ]

    X = df[feature_names]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Metrics
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Train R2: {train_score:.4f}")
    print(f"Test R2: {test_score:.4f}")

    # Artifacts
    artifacts = {
        'model': pipeline,
        'feature_names': feature_names,
        'model_version': 'v1.0',
        'feature_importances': dict(zip(feature_names, pipeline.named_steps['rf'].feature_importances_)),
        'train_score': train_score,
        'test_score': test_score
    }

    # Save
    joblib.dump(artifacts, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
