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
# Two build shapes come out of this one file, selected by a build argument:
#
#   INSTALL_RAG_DEPS=0  (default)  core agent only.
#   INSTALL_RAG_DEPS=1             additionally installs
#                                  requirements-rag-local.txt, which the local
#                                  RAG provider needs to import the retrieval
#                                  engine in-process.
#
# The default shape stays small: numpy, scikit-learn and pypdf are never
# installed unless INSTALL_RAG_DEPS=1 is passed explicitly. The external
# Industrial Knowledge RAG checkout is mounted at run time and is never copied
# into the image.
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
# layer. Both requirement files are copied so the optional set is available to
# the conditional install below.
COPY requirements.txt ./
COPY requirements-rag-local.txt ./

# Selects the build shape. Declared next to its use so a change in the value
# invalidates only the install layer.
ARG INSTALL_RAG_DEPS=0

# The core set is always installed. The optional RAG set is installed only when
# INSTALL_RAG_DEPS=1, so the default image never carries numpy, scikit-learn or
# pypdf.
RUN pip install --no-cache-dir --requirement requirements.txt \
    && if [ "${INSTALL_RAG_DEPS}" = "1" ]; then \
           pip install --no-cache-dir --requirement requirements-rag-local.txt; \
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
