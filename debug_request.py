import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','heart_disease_project.settings')
import django
django.setup()
from django.test import Client
c=Client()
try:
    r=c.get('/en/')
    print('STATUS', r.status_code)
    print('CONTENT', r.content.decode('utf-8', errors='ignore'))
except Exception as e:
    import traceback
    traceback.print_exc()