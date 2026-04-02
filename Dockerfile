# Stage 1: Builder - Dependency Compilation & Resolution
FROM python:3.11-slim as builder

# Optimize environment for containerized builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# STEP 1: Optimize Deep Learning footprint (Saves ~2.5GB)
# We specifically request the CPU-only wheels to avoid bundling unnecessary NVIDIA CUDA binaries
RUN pip install --prefix=/install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install application-specific requirements
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# Stage 2: Runtime - Final Lightweight Production Image
FROM python:3.11-slim

WORKDIR /app

# Copy only the compiled binaries and site-packages from the builder stage
# This ensures the final image is < 4GB and stays within Railway's limits
COPY --from=builder /install /usr/local

# Copy application source code (honoring .dockerignore)
COPY . .

# Railway dynamic port binding configuration
ENV PORT=8000
EXPOSE 8000

# Start the FastAPI application via Uvicorn
# Using exec form to handle signals correctly
CMD ["sh", "-c", "uvicorn crop:app --host 0.0.0.0 --port ${PORT}"]
