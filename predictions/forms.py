from django import forms
from .models import Patient
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

class PatientDataForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = [
            'patient_name', 'first_name', 'last_name', 'birth_date', 'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_bs', 'resting_ecg', 'max_heart_rate', 'exercise_angina',
            'oldpeak', 'st_slope', 'ca', 'thal'
        ]
        widgets = {
            'patient_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Optional: Name (e.g. Dad, Self)'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First name (optional)'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last name (optional)'}),
            'birth_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 45'}),
            'sex': forms.Select(attrs={'class': 'form-select'}),
            'chest_pain_type': forms.Select(attrs={'class': 'form-select'}),
            'resting_bp': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 120'}),
            'cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 200'}),
            'fasting_bs': forms.Select(attrs={'class': 'form-select'}),
            'resting_ecg': forms.Select(attrs={'class': 'form-select'}),
            'max_heart_rate': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 150'}),
            'exercise_angina': forms.Select(attrs={'class': 'form-select'}),
            'oldpeak': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'e.g. 1.0'}),
            'st_slope': forms.Select(attrs={'class': 'form-select'}),
            'ca': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0-3'}),
            'thal': forms.Select(attrs={'class': 'form-select'}),
        }
        widgets = {
            'patient_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Optional: Name (e.g. Dad, Self)'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 45'}),
            'sex': forms.Select(attrs={'class': 'form-select'}),
            'chest_pain_type': forms.Select(attrs={'class': 'form-select'}),
            'resting_bp': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 120'}),
            'cholesterol': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 200'}),
            'fasting_bs': forms.Select(attrs={'class': 'form-select'}),
            'resting_ecg': forms.Select(attrs={'class': 'form-select'}),
            'max_heart_rate': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 150'}),
            'exercise_angina': forms.Select(attrs={'class': 'form-select'}),
            'oldpeak': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'e.g. 1.0'}),
            'st_slope': forms.Select(attrs={'class': 'form-select'}),
            'ca': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '0-3'}),
            'thal': forms.Select(attrs={'class': 'form-select'}),
        }

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))
    specialty = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
    city = forms.CharField(required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))
    birth_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2', 'specialty', 'city', 'birth_date')

    def save(self, commit=True):
        from django.db import transaction, IntegrityError
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        # New users are created inactive until admin validation
        user.is_active = False
        if commit:
            with transaction.atomic():
                user.save()
                # Use update_or_create to avoid UNIQUE constraint errors if a profile was auto-created
                from .models import UserProfile
                try:
                    UserProfile.objects.update_or_create(
                        user=user,
                        defaults={
                            'specialty': self.cleaned_data.get('specialty', ''),
                            'city': self.cleaned_data.get('city', ''),
                            'birth_date': self.cleaned_data.get('birth_date')
                        }
                    )
                except IntegrityError:
                    # If a race condition still causes an integrity error, fetch and update the existing profile
                    profile = UserProfile.objects.get(user=user)
                    profile.specialty = self.cleaned_data.get('specialty', '')
                    profile.city = self.cleaned_data.get('city', '')
                    profile.birth_date = self.cleaned_data.get('birth_date')
                    profile.save()
        return user

from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError

class CustomAuthenticationForm(AuthenticationForm):
    """Custom auth form that provides a clearer message for users pending admin validation."""
    def confirm_login_allowed(self, user):
        # Let Django's default checks (is_active) run first
        if not user.is_active:
            # If the account is inactive and has a profile, explain pending validation
            profile = getattr(user, 'profile', None)
            if profile and not profile.is_validated:
                raise ValidationError('Your account is pending administrator approval. You will receive an email when approved.', code='inactive')
            raise ValidationError('This account is inactive.', code='inactive')
        return super().confirm_login_allowed(user)
