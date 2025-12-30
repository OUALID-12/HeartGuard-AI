from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from predictions.models import PushSubscription
from predictions.utils import send_push

User = get_user_model()

class Command(BaseCommand):
    help = 'Send a test push to all subscriptions or to subscriptions for a given username'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, help='Optional username to send only to that user')
        parser.add_argument('--message', type=str, default='Test notification from HeartGuard AI')

    def handle(self, *args, **options):
        username = options.get('username')
        message = options.get('message')

        subs = PushSubscription.objects.all()
        if username:
            subs = subs.filter(user__username=username)

        if not subs.exists():
            self.stdout.write('No subscriptions found')
            return

        for s in subs:
            sub_info = {'endpoint': s.endpoint, 'keys': s.keys or {}}
            res = send_push(sub_info, {'title': 'HeartGuard AI', 'body': message})
            self.stdout.write(f"Subscription {s.id} ({s.user}): {res}")
