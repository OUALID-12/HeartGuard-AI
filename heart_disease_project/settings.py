
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-test-key-for-dev'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictions',  # Our new app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # Session/privacy: log out users after inactivity
    'predictions.middleware.InactivityLogoutMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'heart_disease_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'heart_disease_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [
    BASE_DIR / 'predictions/static',
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = 'predictions:home'
LOGOUT_REDIRECT_URL = 'predictions:home'
## Auth behavior: redirect to login by default and expire sessions at browser close for added privacy
LOGIN_URL = '/login/'
# When True, the session cookie will expire when the user closes their browser (session cookie).
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# AI Chatbot Configuration
# Provider selection and credentials (prefer environment variables in production)
CHATBOT_PROVIDER = os.getenv('CHATBOT_PROVIDER', 'gemini')  # 'deepseek' or 'gemini'
# Default to the Gemini key you provided for local testing; set CHATBOT_API_KEY in env for production
CHATBOT_API_KEY = os.getenv('CHATBOT_API_KEY', 'AIzaSyCzHGTHKuSquxSuYsxxFcq1Th-avXXZ6AE')
# When using Gemini with a simple API key, set this to 'true' to send the key as a URL parameter (?key=...).
# If you have an OAuth2 access token instead, set it to 'false' and the code will send the token in the Authorization header.
CHATBOT_GEMINI_USE_API_KEY = os.getenv('CHATBOT_GEMINI_USE_API_KEY', 'true').lower() == 'true'
# Default URLs (override via CHATBOT_API_URL env var if needed)
# DeepSeek: https://api.deepseek.com/chat/completions
# Gemini / Google Generative Language example endpoint (may require different path depending on API version)
CHATBOT_API_URL = os.getenv('CHATBOT_API_URL', 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText')

# Session & privacy settings
# Session cookie lifetime (seconds) — short default for increased privacy (30 minutes)
SESSION_COOKIE_AGE = int(os.getenv('SESSION_COOKIE_AGE', 1800))
# Set to True to make the cookie expire at browser close (already set earlier); keep for privacy
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
# Optional sliding session cookie behavior (can be enabled if desired). We use explicit inactivity middleware below
SESSION_SAVE_EVERY_REQUEST = False
# Inactivity timeout (seconds) — middleware will log out users after this period (default 15 minutes)
INACTIVITY_TIMEOUT = int(os.getenv('INACTIVITY_TIMEOUT', 900))

# Email config for development environment
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'no-reply@heartguard.ai')

# VAPID keys for Web Push (set these in your environment for production):
VAPID_PUBLIC_KEY = os.getenv('VAPID_PUBLIC_KEY', '')
VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY', '')
VAPID_EMAIL = os.getenv('VAPID_EMAIL', '') or f"mailto:{DEFAULT_FROM_EMAIL}"


