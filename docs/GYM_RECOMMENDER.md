# Gym Workout Recommender

This project includes a basic Gym Workout recommender trained on `data/gym_members_exercise_tracking.csv`.

Files:
- `ml_model/train_gym_model.py` — training script (RandomForest) that saves `ml_model/saved_models/gym_recommender.pkl`.
- `ml_model/predict_gym.py` — `GymRecommender` class with `recommend(input_dict, top_k=1)`.
- `predictions/recommend/` endpoint and template at `predictions/templates/predictions/recommend.html`.

How to (re)train:
1. Place CSV at `data/gym_members_exercise_tracking.csv` (already added in repository).
2. Run: `python ml_model/train_gym_model.py` — this will create `ml_model/saved_models/gym_recommender.pkl`.

How to use (programmatically):

from ml_model.predict_gym import GymRecommender
rec = GymRecommender()
res = rec.recommend({ 'Age': 30, 'Gender': 'Male', 'Weight (kg)': 80 }, top_k=1)

Web UI:
- Visit `/predictions/recommend/` (login required) to use the simple form.

Notes & Next steps:
- Baseline model accuracy is low (~25%). Improvements: feature engineering, more training data, model tuning, class weighting, and returning top-3 suggestions.
- Add integration tests for the view and form validation.
