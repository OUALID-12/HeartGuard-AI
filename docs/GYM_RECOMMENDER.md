# Gym Workout Recommender

This project includes a basic Gym Workout recommender trained on `data/gym_members_exercise_tracking.csv`.

Files:
- `ml_model/train_gym_model.py` — training script (RandomForest) that saves `ml_model/saved_models/gym_recommender.pkl`.
- `ml_model/predict_gym.py` — `GymRecommender` class with `recommend(input_dict, top_k=1)`.
- `predictions/recommend/` endpoint and template at `predictions/templates/predictions/recommend.html`.

How to (re)train:
1. Place CSV at `data/gym_members_exercise_tracking.csv` (already added in repository).
2. Run with optional hyperparameter tuning:

- Randomized search (recommended):

  `python ml_model/train_gym_model.py --search random --n-iter 20 --cv 4`

- Grid search (small example):

  `python ml_model/train_gym_model.py --search grid --cv 4`

- No search (fast):

  `python ml_model/train_gym_model.py --search none`

A successful run will save `ml_model/saved_models/gym_recommender.pkl` containing a pipeline (preprocessing + classifier) and metadata (`target_le`, `feature_names`, `best_params`).

How to use (programmatically):

from ml_model.predict_gym import GymRecommender
rec = GymRecommender()
rec.load()
res = rec.recommend({ 'Age': 30, 'Gender': 'Male', 'Weight (kg)': 80 }, top_k=3)

Web UI:
- Visit `/predictions/recommend/` (login required) to use the simple form. You may select "Top 3" to show multiple suggestions.

Programmatic API:
- POST to `/predictions/api/recommend/` with JSON `{ "features": { ... }, "top_k": 3 }` and auth cookie to receive JSON recommendations.

Notes & Next steps:
- Baseline model accuracy is modest (around ~25-28% in CV). Recommended improvements: additional feature engineering, more training data, trying boosted trees (XGBoost/LightGBM), and larger randomized search runs.
- Consider adding model versioning and reproducible evaluation reports to track improvements.

