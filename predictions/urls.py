
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .forms import CustomAuthenticationForm

app_name = 'predictions'

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='predictions/login.html', authentication_form=CustomAuthenticationForm), name='login'),
    path('pending-registrations/', views.pending_registrations, name='pending_registrations'),
    path('logout/', auth_views.LogoutView.as_view(next_page='predictions:home'), name='logout'),
    path('predict/', views.predict_view, name='predict'),
    path('recommend/', views.recommend_view, name='recommend'),
    path('api/recommend/', views.recommend_json, name='recommend_api'),
    path('bodyfat/', views.bodyfat_view, name='bodyfat'),
    path('api/bodyfat/', views.bodyfat_json, name='bodyfat_api'),
    path('history/', views.history_view, name='history'),
    path('patient/<int:pk>/', views.patient_detail, name='patient_detail'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('chatbot/api/', views.chatbot_api, name='chatbot_api'),
    path('trends/', views.trends_view, name='trends'),
    path('assessment/', views.assessment_view, name='assessment'),
    path('assessment/result/', views.assessment_result, name='assessment_result'),
    path('lab/', views.lab_upload, name='lab_upload'),
    path('profile/', views.profile_update, name='profile_update'),
    path('simulator/', views.simulator_view, name='simulator'),
    path('api/simulate/', views.api_simulate_risk, name='api_simulate'),
    path('export-pdf/', views.export_report_pdf, name='export_pdf'),
    path('roadmap/', views.roadmap_view, name='roadmap'),
    path('api/ai-health-check/', views.ai_health_check, name='ai_health_check'),
    # PWA offline fallback
    path('offline.html', views.offline, name='offline'),
    # Service worker served at root
    path('service-worker.js', views.service_worker, name='service_worker'),

    # Push API endpoints
    path('api/push/public-key/', views.push_public_key, name='push_public_key'),
    path('api/push/subscribe/', views.push_subscribe, name='push_subscribe'),
    path('api/push/unsubscribe/', views.push_unsubscribe, name='push_unsubscribe'),
    path('api/push/send-test/', views.push_send_test, name='push_send_test'),
]
