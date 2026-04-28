# Stage 1: Build - Dependency Resolution
FROM python:3.11-slim as builder

# Optimize environment for containerized builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal OS dependencies for shared libraries (e.g., OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install application requirements (Optimized for CPU)
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# Stage 2: Runtime - Final 4GB-Safe Image
FROM python:3.11-slim

WORKDIR /app

# Copy only the compiled binaries and site-packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code (honoring .dockerignore)
COPY . .

# Deep cleanup to stay under Railway's 4GB limit
RUN find . -type d -name "__pycache__" -exec rm -rf {} + && \
    rm -rf .git .github .vscode tests test_data .venv

# Railway dynamic port binding configuration
ENV PORT=8000
EXPOSE 8000
# WRONG (exec form - no shell expansion)
CMD ["uvicorn", "crop:app", "--host", "0.0.0.0", "--port", "$PORT"]

# CORRECT (shell form)
CMD uvicorn crop:app --host 0.0.0.0 --port $PORT

# OR CORRECT (exec form via sh)
CMD ["sh", "-c", "uvicorn crop:app --host 0.0.0.0 --port $PORT"]
