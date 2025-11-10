#!/usr/bin/env bash
# Get AWS SSO credentials and export them as environment variables
#
# Usage:
#   eval $(bash lambda/scripts/get-sso-credentials.sh)
#   # Or source it:
#   source lambda/scripts/get-sso-credentials.sh

set -e

AWS_PROFILE="${AWS_PROFILE:-your-account}"

# Colors for output
COLOR_RESET='\033[0m'
COLOR_BOLD='\033[1m'
COLOR_GREEN='\033[32m'
COLOR_YELLOW='\033[33m'
COLOR_RED='\033[31m'

echo -e "${COLOR_BOLD}🔐 Getting AWS SSO credentials for profile: ${AWS_PROFILE}${COLOR_RESET}" >&2

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo -e "${COLOR_RED}❌ AWS CLI not found. Please install it first.${COLOR_RESET}" >&2
    exit 1
fi

# Check if SSO session is valid
if ! aws sts get-caller-identity --profile "${AWS_PROFILE}" &> /dev/null; then
    echo -e "${COLOR_YELLOW}⚠️  SSO session expired or not logged in${COLOR_RESET}" >&2
    echo -e "${COLOR_YELLOW}Logging in to AWS SSO...${COLOR_RESET}" >&2
    aws sso login --profile "${AWS_PROFILE}"
fi

# Get temporary credentials using aws configure export-credentials
echo -e "${COLOR_GREEN}✅ Fetching temporary credentials...${COLOR_RESET}" >&2

# Use aws configure export-credentials (requires AWS CLI v2.13+)
CREDS=$(aws configure export-credentials --profile "${AWS_PROFILE}" --format env 2>/dev/null || echo "")

if [ -z "$CREDS" ]; then
    # Fallback: manually extract from SSO cache
    echo -e "${COLOR_YELLOW}⚠️  Falling back to manual credential extraction${COLOR_RESET}" >&2
    
    # Get the SSO cache file (most recent)
    SSO_CACHE_FILE=$(ls -t ~/.aws/sso/cache/*.json 2>/dev/null | head -1)
    
    if [ -z "$SSO_CACHE_FILE" ]; then
        echo -e "${COLOR_RED}❌ No SSO cache found. Please run: aws sso login --profile ${AWS_PROFILE}${COLOR_RESET}" >&2
        exit 1
    fi
    
    # Extract credentials from cache
    ACCESS_TOKEN=$(jq -r '.accessToken' "$SSO_CACHE_FILE" 2>/dev/null)
    
    if [ -z "$ACCESS_TOKEN" ] || [ "$ACCESS_TOKEN" = "null" ]; then
        echo -e "${COLOR_RED}❌ Invalid SSO cache. Please run: aws sso login --profile ${AWS_PROFILE}${COLOR_RESET}" >&2
        exit 1
    fi
    
    # Get account and role from config
    SSO_ACCOUNT_ID=$(aws configure get sso_account_id --profile "${AWS_PROFILE}")
    SSO_ROLE_NAME=$(aws configure get sso_role_name --profile "${AWS_PROFILE}")
    SSO_REGION=$(aws configure get sso_region --profile "${AWS_PROFILE}")
    
    # Get role credentials
    ROLE_CREDS=$(aws sso get-role-credentials \
        --role-name "${SSO_ROLE_NAME}" \
        --account-id "${SSO_ACCOUNT_ID}" \
        --access-token "${ACCESS_TOKEN}" \
        --region "${SSO_REGION}" \
        --output json 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        echo -e "${COLOR_RED}❌ Failed to get role credentials. Please run: aws sso login --profile ${AWS_PROFILE}${COLOR_RESET}" >&2
        exit 1
    fi
    
    # Extract credentials
    export AWS_ACCESS_KEY_ID=$(echo "$ROLE_CREDS" | jq -r '.roleCredentials.accessKeyId')
    export AWS_SECRET_ACCESS_KEY=$(echo "$ROLE_CREDS" | jq -r '.roleCredentials.secretAccessKey')
    export AWS_SESSION_TOKEN=$(echo "$ROLE_CREDS" | jq -r '.roleCredentials.sessionToken')
    export AWS_REGION=$(aws configure get region --profile "${AWS_PROFILE}")
    
    # Output for eval
    echo "export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}'"
    echo "export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}'"
    echo "export AWS_SESSION_TOKEN='${AWS_SESSION_TOKEN}'"
    echo "export AWS_REGION='${AWS_REGION}'"
else
    # AWS CLI v2.13+ export-credentials worked
    echo "$CREDS"
    eval "$CREDS"
fi

echo -e "${COLOR_GREEN}✅ AWS credentials exported successfully${COLOR_RESET}" >&2
echo -e "${COLOR_BOLD}Identity:${COLOR_RESET}" >&2
aws sts get-caller-identity --no-cli-pager 2>&1 | head -5 >&2
echo "" >&2