from django.test import TestCase
from django.contrib.auth.models import User
from django.urls import reverse

class PendingRegistrationsAccessTests(TestCase):
    def setUp(self):
        # Create users
        self.superuser = User.objects.create_superuser('admin', 'admin@example.com', 'pass')
        self.staff = User.objects.create_user('staff', 'staff@example.com', 'pass')
        self.staff.is_staff = True
        self.staff.save()
        self.normal = User.objects.create_user('normal', 'normal@example.com', 'pass')

    def test_superuser_can_access(self):
        self.client.login(username='admin', password='pass')
        resp = self.client.get(reverse('predictions:pending_registrations'))
        self.assertEqual(resp.status_code, 200)

    def test_staff_gets_forbidden(self):
        self.client.login(username='staff', password='pass')
        resp = self.client.get(reverse('predictions:pending_registrations'))
        self.assertEqual(resp.status_code, 403)

    def test_anonymous_redirects_to_login(self):
        resp = self.client.get(reverse('predictions:pending_registrations'))
        self.assertEqual(resp.status_code, 302)
        # Should redirect to login with next param
        expected = '/login/?next=' + reverse('predictions:pending_registrations')
        self.assertTrue(resp['Location'].startswith('/login/'))

    def test_admin_index_shows_only_pending_registrations(self):
        # Superuser should see the minimal admin index with a link to pending registrations
        self.client.login(username='admin', password='pass')
        resp = self.client.get('/admin/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(reverse('predictions:pending_registrations'), resp.content.decode())
        # Ensure that common model labels are not present to keep the admin minimal
        self.assertNotIn('Patient', resp.content.decode())
        self.assertNotIn('Prediction', resp.content.decode())


class PWATests(TestCase):
    def setUp(self):
        self.client = self.client
        self.user = User.objects.create_user(username='pwatest', password='pass')

    def test_manifest_exists(self):
        import json, os
        static_manifest = os.path.join('static', 'manifest.json')
        self.assertTrue(os.path.exists(static_manifest))
        with open(static_manifest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn('name', data)

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
        import json
        resp = self.client.post('/api/push/subscribe/', data=json.dumps({'endpoint':'https://example.test/endpoint','keys':{}}), content_type='application/json')
        # Should redirect to login or forbidden
        self.assertIn(resp.status_code, (302, 403))

    def test_subscribe_flow(self):
        import json
        self.client.force_login(self.user)
        payload = {'endpoint':'https://example.test/endpoint','keys':{'p256dh':'abc','auth':'xyz'}}
        resp = self.client.post('/api/push/subscribe/', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        from predictions.models import PushSubscription
        self.assertTrue(PushSubscription.objects.filter(endpoint=payload['endpoint']).exists())

    def test_unsubscribe_api_and_ui_flow(self):
        import json
        self.client.force_login(self.user)
        payload = {'endpoint':'https://example.test/to_delete','keys':{'p256dh':'a','auth':'b'}}
        self.client.post('/api/push/subscribe/', data=json.dumps(payload), content_type='application/json')
        # Unsubscribe via API
        resp = self.client.post('/api/push/unsubscribe/', data=json.dumps({'endpoint': payload['endpoint']}), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        from predictions.models import PushSubscription
        self.assertFalse(PushSubscription.objects.filter(endpoint=payload['endpoint']).exists())


