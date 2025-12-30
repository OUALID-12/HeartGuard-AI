#!/usr/bin/env bash
set -e

# Run Django migrations and collectstatic at container start.
# Any failures should not crash the container startup (useful for optional steps)
if [ -f manage.py ]; then
  echo "Running migrations..."
  python manage.py migrate --noinput || echo "Migrations failed or require interactive action"

  echo "Collecting static files..."
  python manage.py collectstatic --noinput || echo "Collectstatic failed"
fi

# Exec container CMD
exec "$@"
