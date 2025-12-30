
from django.contrib import admin
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from .models import Patient, Prediction, Assessment, LabSearch, UserProfile, AuditLog


def approve_users(modeladmin, request, queryset):
    """Admin action to approve (validate) selected user profiles."""
    count = 0
    for user in queryset:
        try:
            profile = getattr(user, 'profile', None)
            if profile and not profile.is_validated:
                profile.is_validated = True
                profile.validated_by = request.user
                profile.validated_at = timezone.now()
                profile.save()
                user.is_active = True
                user.save()
                # Send approval email
                try:
                    send_mail(
                        'Your HeartGuard AI account has been approved',
                        'Your account has been validated by an administrator. You can now log in.',
                        getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@example.com'),
                        [user.email],
                        fail_silently=True
                    )
                except Exception as e:
                    print(f"Failed to send user approval email: {e}")
                # Log audit
                AuditLog.objects.create(action='USER_APPROVED', performed_by=request.user, target_user=user, details='Approved via admin action')
                count += 1
        except Exception as e:
            print(f"Error approving user {user}: {e}")
    modeladmin.message_user(request, f"{count} user(s) approved")


def reject_users(modeladmin, request, queryset):
    """Admin action to reject (deactivate/remove) selected users."""
    count = 0
    for user in queryset:
        try:
            profile = getattr(user, 'profile', None)
            if profile and not profile.is_validated:
                profile.is_validated = False
                profile.validated_by = request.user
                profile.validated_at = timezone.now()
                profile.save()
                user.is_active = False
                user.save()
                # Send rejection email
                try:
                    send_mail(
                        'Your HeartGuard AI registration was not approved',
                        'Your registration was reviewed and not approved. Contact support for more info.',
                        getattr(settings, 'DEFAULT_FROM_EMAIL', 'no-reply@example.com'),
                        [user.email],
                        fail_silently=True
                    )
                except Exception as e:
                    print(f"Failed to send user rejection email: {e}")
                AuditLog.objects.create(action='USER_REJECTED', performed_by=request.user, target_user=user, details='Rejected via admin action')
                count += 1
        except Exception as e:
            print(f"Error rejecting user {user}: {e}")
    modeladmin.message_user(request, f"{count} user(s) rejected")


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    fk_name = 'user'
    can_delete = False
    verbose_name_plural = 'Profile'


class CustomUserAdmin(admin.ModelAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'is_active', 'is_staff')
    actions = [approve_users, reject_users]


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('id', 'age', 'sex', 'created_at')
    list_filter = ('sex', 'chest_pain_type', 'created_at')
    search_fields = ('age',)
    date_hierarchy = 'created_at'


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('patient', 'prediction_result', 'probability', 'predicted_at', 'model_version')
    list_filter = ('prediction_result', 'predicted_at')
    search_fields = ('patient__age',)
    readonly_fields = ('predicted_at',)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('action', 'performed_by', 'target_user', 'timestamp')
    readonly_fields = ('action', 'performed_by', 'target_user', 'details', 'timestamp')

# Replace default User admin with our custom admin that includes the profile inline and approval actions
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

try:
    admin.site.unregister(User)
except Exception:
    pass

@admin.register(User)
class CustomUserAdmin(DjangoUserAdmin):
    inlines = (UserProfileInline,)
    list_display = ('username', 'email', 'is_active', 'is_staff', 'is_validated')
    list_filter = ('is_active', 'is_staff', 'is_superuser', 'profile__is_validated')
    actions = [approve_users, reject_users]

    def is_validated(self, obj):
        return getattr(obj.profile, 'is_validated', False)
    is_validated.short_description = 'Validated'

admin.site.register(Assessment)
admin.site.register(LabSearch)

# Allow admin management of push subscriptions
from .models import PushSubscription

@admin.register(PushSubscription)
class PushSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at')
    readonly_fields = ('endpoint', 'keys', 'created_at')
    search_fields = ('user__username', 'endpoint')

# Make the admin interface minimal: show only a link to Pending Registrations from the admin index.
# We keep the registrations as-is but replace the index template to a minimal one that only exposes the pending registrations feature.
admin.site.site_header = "HeartGuard AI Admin"
admin.site.site_title = "HeartGuard AI Admin"
admin.site.index_title = "Admin"
admin.site.index_template = 'admin/minimal_index.html'
# Disable the standard navigation sidebar so the admin index only shows our minimal content
admin.site.enable_nav_sidebar = False
