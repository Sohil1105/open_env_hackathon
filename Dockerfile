# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for Loan Underwriting OpenEnv
# Exposes a FastAPI HTTP server on port 7860 (HF Spaces default)
# ─────────────────────────────────────────────────────────────────────────────

# Use python:3.10-slim as required by the hackathon spec
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install minimal system dependencies (needed by some pip packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Remove any stale __pycache__ from the build context
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Expose the port that FastAPI will run on
EXPOSE 7860

# Health check using curl (reliable in container environments)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the FastAPI server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
