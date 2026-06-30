FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATEFUL_AI_DATA_DIR=/data/stateful_ai \
    STATEFUL_AI_EMBEDDING_BACKEND=mock

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# This image runs the self-contained Flask demo, so it uses the Flask deps set.
COPY requirements-flask-demo.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-flask-demo.txt

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin stateful_ai \
    && mkdir -p /data/stateful_ai \
    && chown -R stateful_ai:stateful_ai /data/stateful_ai /app

USER stateful_ai

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "apps.flask_app:app"]
