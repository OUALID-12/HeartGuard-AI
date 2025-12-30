# Progressive Web App (PWA) - HeartGuard AI

This document explains how to configure and test the PWA features: offline support, service worker, and push notifications.

## Files added

- `static/manifest.json` - Web App Manifest
- `static/service-worker.js` - Service worker with precaching and runtime caching
- `static/icons/*` - App icons
- `predictions/templates/predictions/offline.html` - Offline fallback page
- `predictions/models.py` - `PushSubscription` model
- `predictions/views.py` - endpoints for push (`/api/push/*`) and service worker
- `predictions/utils.py` - `send_push` helper wrapper around `pywebpush`

## VAPID keys (Push notifications)

Push notifications require VAPID keys. Generate a key pair locally for testing using the management command:

```bash
python manage.py generate_vapid
```

The command prints two environment variables you should set:

- `VAPID_PUBLIC_KEY` (used by client to subscribe)
- `VAPID_PRIVATE_KEY` (used by server to sign push)

For local testing set them in your `.env` or export in shell. DO NOT use production keys in public repos.

Example (Linux/Mac):

```bash
export VAPID_PUBLIC_KEY=...
export VAPID_PRIVATE_KEY=...
export VAPID_EMAIL="mailto:admin@example.com"
```

## Sending test notifications

- Web UI: Click the bell icon at the bottom right (visible to logged-in users) and allow notifications in the browser.
- Admin/CLI: As staff, you can POST to `/api/push/send-test/` with JSON `{ "message": "Hello" }` to send test notifications to all subscriptions.
- CLI management command: `python manage.py send_push_test --username oualid --message 'Hello'`

## HTTPS requirement

Push notifications and service workers require HTTPS except on `localhost`. Ensure you deploy behind TLS in production.

## Troubleshooting

- If the install prompt doesn't appear, try clearing site data and reloading the page with the service worker active.
- If push doesn't arrive, check that `VAPID_PUBLIC_KEY` and `VAPID_PRIVATE_KEY` are configured and valid.
- Use `python manage.py send_push_test` to debug server-sent notifications.

## Tests

- Tests were added to `predictions/tests/test_pwa.py` to verify manifest, service worker, public key endpoint, and subscription flow.

