import time
from django.conf import settings
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth import logout


class InactivityLogoutMiddleware:
    """Middleware to log out users after a period of inactivity.

    It stores a 'last_activity' timestamp in the session and compares it to
    settings.INACTIVITY_TIMEOUT (seconds). If exceeded, the user is logged out
    and redirected to the login page with an informational message.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            timeout = getattr(settings, 'INACTIVITY_TIMEOUT', None)
            if timeout and request.user.is_authenticated:
                now = int(time.time())
                last = request.session.get('last_activity')
                if last and (now - int(last)) > int(timeout):
                    # Inactivity exceeded -> log out and clear session
                    logout(request)
                    request.session.flush()
                    messages.info(request, 'You have been logged out due to inactivity. Please log in again.')
                    return redirect(settings.LOGIN_URL)

                # Update last activity timestamp
                request.session['last_activity'] = now
        except Exception as exc:
            # Fail silently — do not break the site if middleware errors
            print(f"InactivityLogoutMiddleware error: {exc}")

        response = self.get_response(request)
        return response
