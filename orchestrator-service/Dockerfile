# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY orchestrator-service/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

COPY docker/check_requirements.py /opt/ledger/check_requirements.py
COPY docker/entrypoint.sh /usr/local/bin/ledger-entrypoint
COPY orchestrator-service/app ./app

ENV REQUIREMENTS_FILES=/app/requirements.txt
ENTRYPOINT ["/bin/sh", "/usr/local/bin/ledger-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
