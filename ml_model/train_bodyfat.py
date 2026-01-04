import os
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint as sp_randint


def train_and_save_bodyfat(filepath=None, save_name='bodyfat_regressor.pkl', search='random', n_iter=20, cv=4, random_state=42):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if filepath is None:
        filepath = os.path.join(os.path.dirname(base_dir), 'data', 'bodyfat.csv')

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Basic cleaning and standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Target variable: Percent body fat (column named 'BodyFat' or 'Percent body fat')
    # In this dataset it appears as 'BodyFat' (second column)
    if 'BodyFat' in df.columns:
        y = df['BodyFat']
    elif 'Body Fat' in df.columns:
        y = df['Body Fat']
    elif 'Percent body fat' in df.columns:
        y = df['Percent body fat']
    else:
        # fallback: try second column
        y = df.iloc[:, 1]

    # Derive additional features (BMI, Waist-to-Hip)
    # Dataset uses Weight (lbs) and Height (inches). Convert to metric for BMI calculation
    try:
        df['Weight_kg'] = df['Weight'] * 0.45359237
        df['Height_m'] = df['Height'] * 0.0254
        df['BMI'] = df['Weight_kg'] / (df['Height_m'] ** 2)
    except Exception:
        # If columns missing or non-numeric, skip
        pass

    # Waist-to-hip ratio (Abdomen/Hip) — useful indicator of central adiposity
    try:
        df['WHR'] = df['Abdomen'] / df['Hip']
    except Exception:
        pass

    # Drop target and any non-feature cols
    X = df.drop(columns=[y.name])

    # Use only numeric columns for a simple baseline
    X = X.select_dtypes(include=[np.number]).copy()

    # Impute and scale inside pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=random_state))
    ])

    param_dist = {
        'rf__n_estimators': sp_randint(100, 500),
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['auto', 'sqrt']
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    if search == 'random':
        print('Running RandomizedSearchCV...')
        rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=random_state, verbose=1)
        rs.fit(X_train, y_train)
        best = rs.best_estimator_
        best_params = rs.best_params_
    elif search == 'none' or search is None:
        print('Training without hyperparameter search...')
        pipeline.fit(X_train, y_train)
        best = pipeline
        best_params = {}
    else:
        # default fallback to simple pipeline fit
        pipeline.fit(X_train, y_train)
        best = pipeline
        best_params = {}

    preds = best.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Train a simple linear baseline for comparison
    try:
        from sklearn.linear_model import LinearRegression
        baseline_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        baseline_pipe.fit(X_train, y_train)
        baseline_preds = baseline_pipe.predict(X_test)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    except Exception:
        baseline_rmse = None

    # Extract feature importances if available (RandomForest)
    feature_importances = None
    try:
        rf = best.named_steps.get('rf')
        if hasattr(rf, 'feature_importances_'):
            fi = rf.feature_importances_
            feature_importances = dict(zip(X.columns.tolist(), fi.tolist()))
    except Exception:
        feature_importances = None

    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    artifacts = {
        'model': best,
        'feature_names': X.columns.tolist(),
        'best_params': best_params,
        'feature_importances': feature_importances,
        'baseline': {'linear_rmse': baseline_rmse}
    }
    model_path = os.path.join(save_dir, save_name)
    joblib.dump(artifacts, model_path)
    print(f"Saved bodyfat model to {model_path}")

    return {'rmse': rmse, 'r2': r2, 'model_path': model_path, 'best_params': best_params, 'baseline_rmse': baseline_rmse, 'feature_importances': feature_importances}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', choices=['random', 'none'], default='random')
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--cv', type=int, default=4)
    args = parser.parse_args()

    train_and_save_bodyfat(search=args.search, n_iter=args.n_iter, cv=args.cv)
