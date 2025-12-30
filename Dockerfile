# syntax=docker/dockerfile:1

# ---- Builder: install wheels ----
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir /wheels -r requirements.txt

# ---- Runtime: lightweight image ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:$PATH"

# runtime deps (keep minimal; add libpq only if using postgres)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd -m appuser
WORKDIR /app

# install wheels produced in the builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# copy app code
COPY . /app

# add entrypoint
COPY ./entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ensure files owned by non-root user
RUN chown -R appuser:appuser /app
USER appuser

ENV DJANGO_SETTINGS_MODULE=heart_disease_project.settings
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "heart_disease_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]

# simple healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/ || exit 1
