#!/usr/bin/env bash
# Run Lambda container locally with environment loaded

set -e

# Load environment variables
source "$(dirname "$0")/load-env.sh"

# Get AWS SSO credentials if using SSO
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${AWS_ACCESS_KEY_ID}" ] && [ -d "$HOME/.aws/sso" ]; then
    echo ""
    echo -e "${COLOR_BOLD}🔐 Detected AWS SSO - fetching credentials...${COLOR_RESET}"
    eval $(bash "${SCRIPT_DIR}/get-sso-credentials.sh")
fi

# Show environment for debugging
echo ""
echo -e "${COLOR_BOLD}🚀 Starting Lambda container with environment:${COLOR_RESET}"
echo -e "  POSTGRES_HOST: ${POSTGRES_HOST}"
echo -e "  REDIS_HOST: ${REDIS_HOST}"
echo -e "  AWS_REGION: ${AWS_REGION}"
if [ -n "${AWS_ACCESS_KEY_ID}" ]; then
    echo -e "  AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:8}..."
fi
echo ""

# Run Docker container with all environment variables
docker run --rm -it -p 9000:8080 \
    --name instantinsight-lambda-test \
    --add-host=host.docker.internal:host-gateway \
    -e "AWS_REGION=${AWS_REGION}" \
    -e "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" \
    -e "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" \
    -e "AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}" \
    -e "POSTGRES_HOST=${POSTGRES_HOST}" \
    -e "POSTGRES_PORT=${POSTGRES_PORT}" \
    -e "POSTGRES_USER=${POSTGRES_USER}" \
    -e "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
    -e "POSTGRES_DATABASE=${POSTGRES_DATABASE}" \
    -e "REDIS_HOST=${REDIS_HOST}" \
    -e "REDIS_PORT=${REDIS_PORT}" \
    -e "LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}" \
    -e "LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}" \
    -e "LANGFUSE_HOST=${LANGFUSE_HOST}" \
    -e "ATHENA_DATABASE=${ATHENA_DATABASE}" \
    -e "ATHENA_WORK_GROUP=${ATHENA_WORK_GROUP}" \
    -e "RESULTS_BUCKET=${RESULTS_BUCKET}" \
    -e "RESULTS_PREFIX=${RESULTS_PREFIX}" \
    -e "LOG_LEVEL=${LOG_LEVEL}" \
    "$1"