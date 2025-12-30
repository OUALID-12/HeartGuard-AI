import os
import django
import traceback

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'heart_disease_project.settings')
django.setup()

from django.db import IntegrityError, transaction
from django.contrib.auth.models import User
from predictions.forms import RegistrationForm
from predictions.models import UserProfile

TEST_USERNAME = 'hatim'
TEST_EMAIL = 'oualidmansour25+test@icloud.com'

def cleanup():
    try:
        u = User.objects.filter(username=TEST_USERNAME).first()
        if u:
            print('Cleaning up existing user and profile...')
            u.delete()
    except Exception:
        traceback.print_exc()


def run_test():
    cleanup()

    data = {
        'username': TEST_USERNAME,
        'email': TEST_EMAIL,
        'password1': 'Oualid..11',
        'password2': 'Oualid..11',
        'specialty': 'cardiologie',
        'city': 'Marrakech',
        'birth_date': '2000-12-19'
    }

    try:
        f = RegistrationForm(data)
        print('First submission: is_valid=', f.is_valid(), 'errors=', f.errors)
        if f.is_valid():
            with transaction.atomic():
                u = f.save()
            print('First save: created user', u.username, 'is_active=', u.is_active)
            p = UserProfile.objects.filter(user=u).first()
            print('Profile created: specialty=', getattr(p, 'specialty', None), 'city=', getattr(p, 'city', None))

        # Second submission simulating a retry that previously caused UNIQUE constraint
        data2 = data.copy()
        data2['city'] = 'Rabat'  # change one field to see update
        f2 = RegistrationForm(data2)
        print('Second submission: is_valid=', f2.is_valid(), 'errors=', f2.errors)
        if f2.is_valid():
            with transaction.atomic():
                u2 = f2.save()
            print('Second save: user', u2.username, 'is_active=', u2.is_active)
            p2 = UserProfile.objects.get(user=u2)
            print('Profile after second save: specialty=', p2.specialty, 'city=', p2.city)

    except IntegrityError as e:
        print('IntegrityError occurred:')
        traceback.print_exc()
    except Exception as e:
        print('Unexpected exception:')
        traceback.print_exc()


if __name__ == '__main__':
    run_test()
