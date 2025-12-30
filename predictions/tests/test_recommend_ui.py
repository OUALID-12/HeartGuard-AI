from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from predictions.models import GymRecommendation

class RecommendUIIntegrationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='tester', password='pass')
        self.client = Client()
        self.client.login(username='tester', password='pass')

    def test_recommend_form_saves_recommendation(self):
        url = reverse('predictions:recommend')
        form_data = {
            'age': '30',
            'gender': 'Male',
            'weight': '75',
            'height': '1.8',
            'avg_bpm': '140',
            'duration': '1.0',
            'top_k': '3'
        }
        resp = self.client.post(url, data=form_data)
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode('utf-8')
        self.assertIn('Recommendations', content)
        # Ensure a GymRecommendation was created
        recs = GymRecommendation.objects.filter(user=self.user)
        self.assertTrue(recs.exists())
        r = recs.first()
        self.assertIsNotNone(r.inputs)
        self.assertIn('recommendations', r.inputs)
