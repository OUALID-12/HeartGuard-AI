from django.core.management.base import BaseCommand
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode('ascii')


class Command(BaseCommand):
    help = 'Generate VAPID keypair (public & private) formatted for web-push (base64url)

    The command prints two values: PUBLIC_KEY and PRIVATE_KEY which you can set as env variables
    VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY respectively.'

    def handle(self, *args, **kwargs):
        # Generate P-256 key
        private_key = ec.generate_private_key(ec.SECP256R1())
        private_value = private_key.private_numbers().private_value
        private_bytes = private_value.to_bytes(32, 'big')

        public_numbers = private_key.public_key().public_numbers()
        x = public_numbers.x.to_bytes(32, 'big')
        y = public_numbers.y.to_bytes(32, 'big')
        # Uncompressed point, 0x04 || X || Y
        public_bytes = b"\x04" + x + y

        public_key_b64 = base64url_encode(public_bytes)
        private_key_b64 = base64url_encode(private_bytes)

        self.stdout.write("VAPID_PUBLIC_KEY=" + public_key_b64)
        self.stdout.write("VAPID_PRIVATE_KEY=" + private_key_b64)
        self.stdout.write("")
        self.stdout.write("Save these to your environment (e.g., .env) as VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY.")
        self.stdout.write("Example (Linux/macOS): export VAPID_PUBLIC_KEY=... && export VAPID_PRIVATE_KEY=...")
