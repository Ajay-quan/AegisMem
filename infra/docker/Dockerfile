FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    sqlalchemy>=2.0.0 \
    alembic>=1.13.0 \
    asyncpg>=0.29.0 \
    psycopg2-binary>=2.9.9 \
    redis>=5.0.3 \
    celery>=5.3.6 \
    openai>=1.30.0 \
    anthropic>=0.28.0 \
    qdrant-client>=1.9.0 \
    neo4j>=5.20.0 \
    langgraph>=0.1.0 \
    langchain-core>=0.2.0 \
    opentelemetry-api>=1.24.0 \
    opentelemetry-sdk>=1.24.0 \
    prometheus-client>=0.20.0 \
    python-json-logger>=2.0.7 \
    tenacity>=8.3.0 \
    python-dotenv>=1.0.1 \
    rich>=13.7.1 \
    numpy>=1.26.0 \
    scipy>=1.13.0

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
