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
#
# Build shape is selected by two independent build arguments:
#
#   INSTALL_RAG_DEPS   0 (default)  core agent only
#                      1            additionally installs
#                                   requirements-rag-local.txt, which the local
#                                   RAG provider needs to import the retrieval
#                                   engine in-process
#
#   INSTALL_OTEL_DEPS  0 (default)  no OpenTelemetry SDK in the image
#                      1            additionally installs requirements-otel.txt,
#                                   which OTEL_ENABLED=true needs in order to
#                                   record spans
#
# The default shape stays small: numpy, scikit-learn, pypdf and the
# OpenTelemetry SDK are never installed unless the matching argument is passed
# explicitly. The external Industrial Knowledge RAG checkout is mounted at run
# time and is never copied into the image.
#
# Why the tracing dependency is a build shape rather than always installed:
# OTEL_ENABLED defaults to false and a disabled tracer imports nothing, so the
# default image does not need the SDK. A deployment that sets OTEL_ENABLED=true
# without this argument still starts and logs
# "otel_sdk_missing tracing_disabled=true"; it simply produces no spans. Build
# with INSTALL_OTEL_DEPS=1 when you intend to trace.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unprivileged runtime account. The application never needs root.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first, so editing application source does not invalidate this
# layer. Every optional requirement file is copied so the conditional installs
# below can reach them.
COPY requirements.txt ./
COPY requirements-rag-local.txt ./
COPY requirements-otel.txt ./

# Select the build shape. Declared next to their use so a change in a value
# invalidates only the install layer.
ARG INSTALL_RAG_DEPS=0
ARG INSTALL_OTEL_DEPS=0

# The core set is always installed. Each optional set is installed only when its
# argument is 1, so the default image never carries numpy, scikit-learn, pypdf or
# the OpenTelemetry SDK.
RUN pip install --no-cache-dir --requirement requirements.txt \
    && if [ "${INSTALL_RAG_DEPS}" = "1" ]; then \
           pip install --no-cache-dir --requirement requirements-rag-local.txt; \
       fi \
    && if [ "${INSTALL_OTEL_DEPS}" = "1" ]; then \
           pip install --no-cache-dir --requirement requirements-otel.txt; \
       fi

# Application source and the data the tools read. No tests and no caches.
#
# The data copy names its two files instead of taking the directory. An external
# dataset placed under data/ would otherwise be baked into the image, which is
# both a size problem and a redistribution problem: the MetroPT-3 CSV is about
# 208 MiB and this project does not ship it. .dockerignore excludes that
# directory as well, so the context stays small even before the copy runs.
COPY pyproject.toml README.md ./
COPY app ./app
COPY data/devices.json data/alarms.json ./data/
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
