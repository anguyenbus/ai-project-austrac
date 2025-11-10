#!/usr/bin/env bash
# Lambda Local Testing Environment Loader
#
# This script loads environment variables from the project's .env file
# and exports them for use with local Lambda testing.
#
# Usage:
#   source lambda/scripts/load-env.sh
#   # Or it's called automatically by: make test-local

set -a  # Automatically export all variables

# Colors for output
COLOR_RESET='\033[0m'
COLOR_BOLD='\033[1m'
COLOR_GREEN='\033[32m'
COLOR_YELLOW='\033[33m'
COLOR_BLUE='\033[34m'

echo -e "${COLOR_BOLD}🔧 Loading environment for Lambda local testing...${COLOR_RESET}"

# Find project root (go up from lambda/scripts/ to project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

# Check if .env file exists
if [ ! -f "${ENV_FILE}" ]; then
    echo -e "${COLOR_YELLOW}⚠️  Warning: .env file not found at ${ENV_FILE}${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}   Using default values...${COLOR_RESET}"
else
    echo -e "${COLOR_GREEN}✅ Loading from: ${ENV_FILE}${COLOR_RESET}"
    # Load .env file, ignoring comments and empty lines
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        if [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]]; then
            # Remove inline comments
            line="${line%%#*}"
            # Export the variable
            export "$line"
        fi
    done < "${ENV_FILE}"
fi

# Override with Docker-friendly defaults for local testing
# NOTE: Force override for Docker networking (not :-default syntax)
export POSTGRES_HOST="host.docker.internal"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_DATABASE="${POSTGRES_DATABASE:-instantinsight}"

export REDIS_HOST="host.docker.internal"
export REDIS_PORT="${REDIS_PORT:-6379}"

export LANGFUSE_HOST="http://host.docker.internal:3000"

export ATHENA_DATABASE="${ATHENA_DATABASE:-text_to_sql}"
export ATHENA_WORK_GROUP="${ATHENA_WORK_GROUP:-primary}"

export RESULTS_BUCKET="${RESULTS_BUCKET:-instantinsight-query-results}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-query-results/}"

export AWS_REGION="${AWS_REGION:-ap-southeast-2}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Validation: Check required variables
echo ""
echo -e "${COLOR_BOLD}Environment Configuration:${COLOR_RESET}"
echo -e "  ${COLOR_BLUE}POSTGRES_HOST:${COLOR_RESET}     ${POSTGRES_HOST}"
echo -e "  ${COLOR_BLUE}POSTGRES_PORT:${COLOR_RESET}     ${POSTGRES_PORT}"
echo -e "  ${COLOR_BLUE}POSTGRES_USER:${COLOR_RESET}     ${POSTGRES_USER}"
echo -e "  ${COLOR_BLUE}POSTGRES_DATABASE:${COLOR_RESET} ${POSTGRES_DATABASE}"
echo -e "  ${COLOR_BLUE}REDIS_HOST:${COLOR_RESET}        ${REDIS_HOST}"
echo -e "  ${COLOR_BLUE}AWS_REGION:${COLOR_RESET}        ${AWS_REGION}"

# Check critical missing variables
MISSING_VARS=()

if [ -z "${POSTGRES_PASSWORD}" ]; then
    MISSING_VARS+=("POSTGRES_PASSWORD")
fi

if [ -z "${AWS_ACCESS_KEY_ID}" ]; then
    MISSING_VARS+=("AWS_ACCESS_KEY_ID (optional for local testing)")
fi

if [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    MISSING_VARS+=("AWS_SECRET_ACCESS_KEY (optional for local testing)")
fi

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo ""
    echo -e "${COLOR_YELLOW}⚠️  Missing environment variables:${COLOR_RESET}"
    for var in "${MISSING_VARS[@]}"; do
        echo -e "  - ${var}"
    done
    echo ""
    echo -e "${COLOR_YELLOW}Add these to your .env file or export them manually.${COLOR_RESET}"
fi

echo ""
echo -e "${COLOR_GREEN}✅ Environment loaded successfully!${COLOR_RESET}"
echo ""

set +a  # Stop automatically exporting