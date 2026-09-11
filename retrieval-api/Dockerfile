# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/models/huggingface
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY retrieval-api/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r requirements.txt

COPY docker/check_requirements.py /opt/ledger/check_requirements.py
COPY docker/entrypoint.sh /usr/local/bin/ledger-entrypoint
COPY retrieval-api/app ./app

ENV REQUIREMENTS_FILES=/app/requirements.txt
ENTRYPOINT ["/bin/sh", "/usr/local/bin/ledger-entrypoint"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
