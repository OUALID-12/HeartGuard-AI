from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from .models import GymRecommendation

class HistoryRecommendationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='tester', password='pass')
        self.client = Client()
        self.client.login(username='tester', password='pass')

    def test_recommendation_shows_in_history(self):
        # create a recommendation with multiple suggestions in inputs
        inputs = {
            'features': {'Age': 30},
            'recommendations': [
                {'label': 'Cardio', 'probability': 0.75},
                {'label': 'Strength', 'probability': 0.10}
            ]
        }
        GymRecommendation.objects.create(user=self.user, recommended_workout='Cardio', confidence=0.75, inputs=inputs)
        resp = self.client.get(reverse('predictions:history'))
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode('utf-8')
        self.assertIn('Workout Recommendation', content)
        self.assertIn('Cardio', content)
        self.assertIn('Strength', content)
