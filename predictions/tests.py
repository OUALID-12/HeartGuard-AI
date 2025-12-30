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


