
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Patient(models.Model):
    SEX_CHOICES = [
        (1, 'Male'),
        (0, 'Female'),
    ]
    
    CHEST_PAIN_CHOICES = [
        (1, 'Typical Angina'),
        (2, 'Atypical Angina'),
        (3, 'Non-anginal Pain'),
        (4, 'Asymptomatic'),
    ]
    
    FASTING_BS_CHOICES = [
        (0, 'False (< 120 mg/dl)'),
        (1, 'True (> 120 mg/dl)'),
    ]
    
    RESTING_ECG_CHOICES = [
        (0, 'Normal'),
        (1, 'ST-T Wave Abnormality'),
        (2, 'Left Ventricular Hypertrophy'),
    ]
    
    EXERCISE_ANGINA_CHOICES = [
        (0, 'No'),
        (1, 'Yes'),
    ]
    
    SLOPE_CHOICES = [
        (1, 'Upsloping'),
        (2, 'Flat'),
        (3, 'Downsloping'),
    ]

    THAL_CHOICES = [
        (3, 'Normal'),
        (6, 'Fixed Defect'),
        (7, 'Reversable Defect'),
    ]

    # Patient Demographics & Basic Vitals
    patient_name = models.CharField(max_length=100, default="My Self", help_text="Name or Identifier for this record")
    # Optional separate fields for better patient identification
    first_name = models.CharField(max_length=50, blank=True, null=True, help_text="Patient's first name")
    last_name = models.CharField(max_length=50, blank=True, null=True, help_text="Patient's last name")
    birth_date = models.DateField(blank=True, null=True, help_text="Patient date of birth")
    age = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(120)], help_text="Age in years")
    sex = models.IntegerField(choices=SEX_CHOICES, help_text="Patient's gender")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Prefer using first/last name if provided; otherwise use the generic patient_name
        name_parts = []
        if self.first_name:
            name_parts.append(self.first_name)
        if self.last_name:
            name_parts.append(self.last_name)
        name_display = " ".join(name_parts) if name_parts else self.patient_name
        dob = f" - Born {self.birth_date.isoformat()}" if self.birth_date else ""
        return f"Patient #{self.id} - {name_display}{dob} - {self.age}y/o {self.get_sex_display()}"
    
    # Medical Features
    chest_pain_type = models.IntegerField(choices=CHEST_PAIN_CHOICES, verbose_name="Chest Pain Type (cp)")
    resting_bp = models.IntegerField(validators=[MinValueValidator(50), MaxValueValidator(250)], verbose_name="Resting Blood Pressure (trestbps)", help_text="in mm Hg")
    cholesterol = models.IntegerField(validators=[MinValueValidator(100), MaxValueValidator(600)], verbose_name="Serum Cholesterol (chol)", help_text="in mg/dl")
    fasting_bs = models.IntegerField(choices=FASTING_BS_CHOICES, verbose_name="Fasting Blood Sugar (fbs)")
    resting_ecg = models.IntegerField(choices=RESTING_ECG_CHOICES, verbose_name="Resting ECG Results (restecg)")
    max_heart_rate = models.IntegerField(validators=[MinValueValidator(60), MaxValueValidator(220)], verbose_name="Max Heart Rate (thalach)")
    exercise_angina = models.IntegerField(choices=EXERCISE_ANGINA_CHOICES, verbose_name="Exercise Induced Angina (exang)")
    oldpeak = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(10.0)], help_text="ST depression induced by exercise relative to rest")
    st_slope = models.IntegerField(choices=SLOPE_CHOICES, verbose_name="ST Slope (slope)")
    ca = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(4)], verbose_name="Number of Major Vessels (ca)", help_text="0-3 colored by flourosopy")
    thal = models.IntegerField(choices=THAL_CHOICES, verbose_name="Thalassemia (thal)")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # Prefer using first/last name if provided; otherwise use the generic patient_name
        name_parts = []
        if self.first_name:
            name_parts.append(self.first_name)
        if self.last_name:
            name_parts.append(self.last_name)
        name_display = " ".join(name_parts) if name_parts else self.patient_name
        dob = f" - Born {self.birth_date.isoformat()}" if self.birth_date else ""
        return f"Patient #{self.id} - {name_display}{dob} - {self.age}y/o {self.get_sex_display()}"

from django.contrib.auth.models import User

class UserProfile(models.Model):
    """Extended profile for additional user data and admin validation status."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    specialty = models.CharField(max_length=120, blank=True)
    city = models.CharField(max_length=120, blank=True)
    birth_date = models.DateField(blank=True, null=True)
    is_validated = models.BooleanField(default=False, help_text='Set to true when an admin validates the user registration')
    validated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='validations')
    validated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Profile for {self.user.username}"

class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('USER_APPROVED', 'User Approved'),
        ('USER_REJECTED', 'User Rejected'),
        ('USER_CREATED', 'User Created'),
        ('USER_DEACTIVATED', 'User Deactivated'),
        ('OTHER', 'Other'),
    ]

    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    performed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='performed_actions')
    target_user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_targets')
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.TextField(blank=True)

    def __str__(self):
        return f"{self.get_action_display()} by {self.performed_by} on {self.target_user} at {self.timestamp}"

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='predictions')
    prediction_result = models.IntegerField(choices=[(0, 'No Disease'), (1, 'Heart Disease')], verbose_name="Prediction")
    probability = models.FloatField(help_text="Probability of heart disease (0-1)")
    model_version = models.CharField(max_length=50, default="v1.0")
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.patient} - {self.get_prediction_result_display()}"

class Assessment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    patient_name = models.CharField(max_length=100, default="My Self")
    score_phq = models.IntegerField()
    score_gad = models.IntegerField()
    wellness_score = models.IntegerField()
    risk_level = models.CharField(max_length=20)
    taken_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Assessment for {self.user.username} - {self.risk_level} Risk"


# Signals to ensure a UserProfile exists for every User
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def ensure_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)

class LabSearch(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    patient_name = models.CharField(max_length=100, default="My Self")
    query = models.TextField()
    results_count = models.IntegerField()
    searched_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search by {self.user.username}: {self.query[:30]}"

class Alert(models.Model):
    ALERT_TYPES = [
        ('TREND', 'Critical Trend'),
        ('CHECKUP', 'Check-up Reminder'),
        ('MEDICATION', 'Medication Reminder'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alerts')
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    is_sent_via_email = models.BooleanField(default=False)
    is_sent_via_sms = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_alert_type_display()} for {self.user.username}"


class PushSubscription(models.Model):
    """Stores a browser push subscription for a user."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='push_subscriptions')
    endpoint = models.TextField(unique=True)
    keys = models.JSONField(blank=True, null=True, help_text='p256dh and auth keys')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        owner = self.user.username if self.user else 'anonymous'
        return f"PushSubscription for {owner} ({self.endpoint[:50]}...)"
