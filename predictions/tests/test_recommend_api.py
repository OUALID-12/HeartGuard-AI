from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
import json

class RecommendApiTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='tester', password='pass')
        self.client = Client()
        self.client.login(username='tester', password='pass')

    def test_recommend_api_returns_recommendations(self):
        url = reverse('predictions:recommend_api')
        payload = {
            'features': {
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
            },
            'top_k': 3
        }
        resp = self.client.post(url, data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get('success'))
        recs = data.get('recommendations', [])
        self.assertTrue(isinstance(recs, list))
        self.assertTrue(len(recs) <= 3)
        if len(recs) > 0:
            self.assertIn('label', recs[0])
            self.assertIn('probability', recs[0])
