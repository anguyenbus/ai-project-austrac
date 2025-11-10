# Multi-stage Dockerfile for instantinsight Lambda Container
# Uses uv for ultra-fast dependency installation from pyproject.toml

# Stage 1: Build dependencies with uv (official uv image)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set working directory for build
WORKDIR /build

# Copy project files needed for build
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install dependencies using uv to /opt/python
# uv is 10-100x faster than pip for dependency resolution
RUN uv pip install --system --no-cache --target /opt/python .

# Stage 2: Runtime image (AWS Lambda Python base)
FROM public.ecr.aws/lambda/python:3.12

# Copy installed dependencies from builder
COPY --from=builder /opt/python /var/task

# Copy application code
COPY src/ /var/task/src/
COPY lambda/lambda_handler.py /var/task/

# Set Python path for module imports
ENV PYTHONPATH=/var/task

# Set Lambda runtime environment variables
ENV AWS_LAMBDA_FUNCTION_NAME=instantinsight-query-processor
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Lambda handler
CMD ["lambda_handler.lambda_handler"]