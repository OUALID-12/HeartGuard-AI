from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from predictions.models import PushSubscription
import json

User = get_user_model()

class PWATests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='pwatest', password='pass')

    def test_manifest_exists(self):
        resp = self.client.get('/static/manifest.json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('name', resp.json())

    def test_service_worker_served(self):
        resp = self.client.get('/service-worker.js')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('CACHE_NAME', resp.content.decode('utf-8'))

    def test_push_public_key_endpoint(self):
        resp = self.client.get('/api/push/public-key/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('publicKey', data)

    def test_subscribe_requires_login(self):
        resp = self.client.post('/api/push/subscribe/', data=json.dumps({'endpoint':'https://example.test/endpoint','keys':{}}), content_type='application/json')
        # Should redirect to login
        self.assertIn(resp.status_code, (302, 403))

    def test_subscribe_flow(self):
        self.client.force_login(self.user)
        payload = {'endpoint':'https://example.test/endpoint','keys':{'p256dh':'abc','auth':'xyz'}}
        resp = self.client.post('/api/push/subscribe/', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(PushSubscription.objects.filter(endpoint=payload['endpoint']).exists())
