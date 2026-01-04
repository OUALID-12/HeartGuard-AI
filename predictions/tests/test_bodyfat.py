from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from ml_model.train_bodyfat import train_and_save_bodyfat
from ml_model.predict_bodyfat import BodyFatPredictor
import os

class BodyFatTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='tester', password='pass')
        self.client = Client()
        self.client.login(username='tester', password='pass')

    def test_train_script_runs(self):
        res = train_and_save_bodyfat(search='none')
        self.assertIn('model_path', res)
        self.assertTrue(os.path.exists(res['model_path']))
        # Newer train script returns baseline and feature importances
        self.assertIn('baseline_rmse', res)
        self.assertIn('feature_importances', res)

    def test_predictor_loads_and_predicts(self):
        pred = BodyFatPredictor()
        pred.load()
        sample = {'Age': 30, 'Weight': 170, 'Height': 70, 'Neck': 38.0, 'Abdomen': 90.0}
        out = pred.predict(sample)
        # backward-compat: model returns 'bodyfat_percent_model'
        self.assertIn('bodyfat_percent_model', out)
        self.assertIsInstance(out['bodyfat_percent_model'], float)

    def test_predictor_with_density_returns_siri(self):
        pred = BodyFatPredictor()
        pred.load()
        sample = {'Density': 1.0708}
        out = pred.predict(sample)
        self.assertIn('bodyfat_percent_siri', out)
        # Check Siri equation roughly
        self.assertAlmostEqual(out['bodyfat_percent_siri'], 495.0/1.0708 - 450.0, places=3)

    def test_bodyfat_ui_saves_prediction(self):
        url = reverse('predictions:bodyfat')
        data = {'Age': '30', 'Weight': '170', 'Height': '70', 'Neck': '38', 'Abdomen': '90'}
        resp = self.client.post(url, data=data)
        self.assertEqual(resp.status_code, 200)
        # ensure DB entry
        from predictions.models import BodyFatPrediction
        self.assertTrue(BodyFatPrediction.objects.filter(user=self.user).exists())
