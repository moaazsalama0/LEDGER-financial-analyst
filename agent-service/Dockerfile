# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY agent-service/requirements-light.txt agent-service/requirements-gemini.txt agent-service/requirements-groq.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements-gemini.txt -r requirements-groq.txt

COPY docker/check_requirements.py /opt/ledger/check_requirements.py
COPY docker/entrypoint.sh /usr/local/bin/ledger-entrypoint
COPY agent-service/app ./app

ENV REQUIREMENTS_FILES=/app/requirements-gemini.txt:/app/requirements-groq.txt
ENTRYPOINT ["/bin/sh", "/usr/local/bin/ledger-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
