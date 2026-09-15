# syntax=docker/dockerfile:1

# Industrial Maintenance Agent.
#
# The service is LLM-free by default (PLANNER_MODE=rule), so the image needs no
# provider credential and none is baked into it. Credentials, when the optional
# LLM planner is enabled, are supplied through the environment at run time.
#
# Nothing secret is copied. ".env" and every ".env.*" variant are excluded by
# .dockerignore; only ".env.example", which holds placeholders, is readable in
# the build context and it is not copied into the image either.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged runtime account. The application never needs root.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first, so editing application source does not invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

# Application source and the data the tools read. No tests and no caches.
COPY pyproject.toml README.md ./
COPY app ./app
COPY data ./data
COPY evaluation ./evaluation
COPY scripts ./scripts

# SQLite defaults to sqlite:///./data/industrial_maintenance.db, so this
# directory must be writable by the runtime user. Ownership is set here so a
# fresh named volume mounted at /app/data inherits it.
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# Liveness probe against the real endpoint. A non-zero exit marks the container
# unhealthy; there is no TCP-only check that can pass while the app is broken.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

# Deterministic planner by default. Set PLANNER_MODE=llm plus the LLM_* variables
# to enable the optional LLM planner inside the container.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PLANNER_MODE=rule

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
