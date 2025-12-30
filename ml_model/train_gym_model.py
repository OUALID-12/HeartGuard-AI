import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_and_save_gym_recommender(filepath=None, save_name='gym_recommender.pkl'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    if filepath is None:
        filepath = os.path.join(os.path.dirname(base_dir), 'data', 'gym_members_exercise_tracking.csv')

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=['Workout_Type'])

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Fill or compute BMI if missing
    if 'BMI' not in df.columns or df['BMI'].isnull().any():
        if 'Weight (kg)' in df.columns and 'Height (m)' in df.columns:
            df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)

    # Select features and target
    features = ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)',
                'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

    # Keep only existing features
    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df['Workout_Type'].copy()

    # Handle missing numeric values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Encode categorical columns
    encoders = {}
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    y = target_le.fit_transform(y.astype(str))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_le.classes_))

    # Save artifacts
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    artifacts = {
        'model': clf,
        'scaler': scaler,
        'encoders': encoders,
        'target_le': target_le,
        'feature_names': X.columns.tolist()
    }
    model_path = os.path.join(save_dir, save_name)
    joblib.dump(artifacts, model_path)
    print(f"Saved gym recommender to {model_path}")

    return acc, model_path


if __name__ == '__main__':
    train_and_save_gym_recommender()
