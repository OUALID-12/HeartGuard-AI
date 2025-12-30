import os
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint


def train_and_save_gym_recommender(filepath=None, save_name='gym_recommender.pkl', search='random', n_iter=20, cv=4, random_state=42):
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

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop'
    )

    # Encode target
    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))

    # Make a holdout split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc)

    # Build pipeline with classifier
    base_clf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', base_clf)
    ])

    # Hyperparameter search space
    param_dist = {
        'clf__n_estimators': sp_randint(100, 500),
        'clf__max_depth': [None, 5, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2']
    }

    if search is None or search == 'none':
        print("Training without hyperparameter search...")
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = {}
    else:
        print(f"Running {'RandomizedSearchCV' if search=='random' else 'GridSearchCV'} for hyperparameter tuning...")
        if search == 'random':
            rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='accuracy', n_jobs=-1, random_state=random_state, verbose=1)
            rs.fit(X_train, y_train)
            best_model = rs.best_estimator_
            best_params = rs.best_params_
            print(f"Random search best params: {best_params}")
            print(f"Random search best CV score: {rs.best_score_:.4f}")
        else:
            # small grid example
            grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10],
                'clf__max_features': ['sqrt']
            }
            gs = GridSearchCV(pipeline, param_grid=grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            print(f"Grid search best params: {best_params}")
            print(f"Grid search best CV score: {gs.best_score_:.4f}")

    # Evaluate on holdout test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_le.classes_))

    # Save artifacts (pipeline includes preprocessing)
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    artifacts = {
        'model': best_model,
        'target_le': target_le,
        'feature_names': X.columns.tolist(),
        'best_params': best_params
    }
    model_path = os.path.join(save_dir, save_name)
    joblib.dump(artifacts, model_path)
    print(f"Saved gym recommender to {model_path}")

    return {
        'accuracy': acc,
        'model_path': model_path,
        'best_params': best_params
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', choices=['random', 'grid', 'none'], default='random')
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--cv', type=int, default=4)
    args = parser.parse_args()
    train_and_save_gym_recommender(search=args.search, n_iter=args.n_iter, cv=args.cv)
