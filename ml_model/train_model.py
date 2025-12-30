
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .data_loader import load_data

def train_and_save_model():
    print("Loading data...")
    df = load_data()
    
    # Preprocessing
    print("Preprocessing data...")
    
    # Map columns to standard names used in our application
    # Input CSV columns: Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium,Heart Disease
    column_mapping = {
        'Age': 'age',
        'Sex': 'sex',
        'Chest pain type': 'chest_pain_type',
        'BP': 'resting_bp',
        'Cholesterol': 'cholesterol',
        'FBS over 120': 'fasting_bs',
        'EKG results': 'resting_ecg',
        'Max HR': 'max_heart_rate',
        'Exercise angina': 'exercise_angina',
        'ST depression': 'oldpeak',
        'Slope of ST': 'st_slope',
        'Number of vessels fluro': 'ca',
        'Thallium': 'thal',
        'Heart Disease': 'target'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Handle target column encoding (Presence/Absence -> 1/0)
    if df['target'].dtype == 'object':
        df['target'] = df['target'].map({'Presence': 1, 'Absence': 0})
        
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Handle categorical encoding for features if any remain (most in this dataset are numeric codes already)
    # Check for object columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': X.columns.tolist(),
        'target_col': 'target'
    }
    
    model_path = os.path.join(save_dir, 'heart_disease_model.pkl')
    joblib.dump(artifacts, model_path)
    print(f"Model and artifacts saved to {model_path}")
    
    return accuracy

if __name__ == "__main__":
    train_and_save_model()
