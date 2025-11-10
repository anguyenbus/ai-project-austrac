# Lambda Container Deployment

Query processor Lambda function deployed as a container image.

## Directory Structure

```
lambda/
├── Dockerfile              # Multi-stage container build
├── lambda_handler.py       # Lambda entry point
├── Makefile               # Build and deployment automation
├── README.md              # This file
├── infrastructure/
│   └── cloudformation/    # CloudFormation IaC templates
├── scripts/               # Helper scripts for local testing
└── tests/                 # Lambda handler tests
```

## Local Testing

### Prerequisites

- Docker installed and running
- AWS SSO configured with `your-account` profile
- Project dependencies installed via `uv`

### AWS SSO Setup

The Lambda function requires AWS credentials to access S3, Athena, and Bedrock services.

**Configure SSO:**
```bash
aws configure sso --profile your-account
```

**Login to SSO:**
```bash
aws sso login --profile your-account
```

The test scripts automatically fetch temporary credentials from your SSO cache.

### Run Local Tests

Start the Lambda runtime emulator:
```bash
make test-local
```

This command:
- Builds the container image
- Extracts AWS SSO credentials automatically
- Starts the Lambda runtime on port 9000
- Maps PostgreSQL and Redis to `host.docker.internal`

In another terminal, invoke the function:
```bash
make test-invoke
```

Or manually with curl:
```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me top 10 products by revenue"}'
```

## Build Image

Build the container image:
```bash
make build
```

Build with custom tag:
```bash
make build IMAGE_TAG=v1.0.0
```

The build process:
- Uses multi-stage Docker build
- Installs dependencies via `uv` from `pyproject.toml`
- Creates optimized runtime image based on AWS Lambda Python 3.12
- Compiles native extensions (psycopg, pandas, numpy)
