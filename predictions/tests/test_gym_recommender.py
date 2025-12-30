from django.test import TestCase
from ml_model.predict_gym import GymRecommender

class GymRecommenderTest(TestCase):
    def test_recommend_returns_label(self):
        rec = GymRecommender()
        # Ensure model loads
        rec.load()
        # Sample input (reasonable defaults)
        sample = {
            'Age': 30,
            'Gender': 'Male',
            'Weight (kg)': 75,
            'Height (m)': 1.8,
            'Avg_BPM': 145,
            'Session_Duration (hours)': 1.0,
            'Calories_Burned': 800,
            'Workout_Frequency (days/week)': 3,
            'Experience_Level': 2,
            'BMI': 23.15
        }
        res = rec.recommend(sample, top_k=1)
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        label, prob = res[0]
        self.assertIsInstance(label, str)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
