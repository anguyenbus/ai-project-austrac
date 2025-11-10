#!/bin/bash

# Add Model to Langfuse API
# Usage: ./add_langfuse_model.sh <model_name> <input_price> <output_price>

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -E '^LANGFUSE_' | xargs)
fi

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model_name> <input_price> <output_price>"
    echo ""
    echo "Examples:"
    echo "  $0 'anthropic.claude-3-haiku-20240307-v1:0' 0.0008 0.0032"
    echo "  $0 'gpt-4' 0.01 0.03"
    echo "  $0 'claude-3.5-sonnet' 0.003 0.015"
    exit 1
fi

# Configuration
LANGFUSE_HOST="${LANGFUSE_HOST:-http://localhost:3000}"
MODEL_NAME="$1"
INPUT_PRICE="$2"
OUTPUT_PRICE="$3"

# Auto-generate match pattern from model name
SIMPLE_NAME=$(echo "$MODEL_NAME" | sed 's/[^a-zA-Z0-9-]/-/g')
MATCH_PATTERN="(?i)^(${MODEL_NAME//./\\.}|${SIMPLE_NAME})$"

# Check credentials
if [ -z "$LANGFUSE_PUBLIC_KEY" ] || [ -z "$LANGFUSE_SECRET_KEY" ]; then
    echo "Error: Missing Langfuse credentials in .env file"
    exit 1
fi

# Generate auth token
AUTH_TOKEN=$(echo -n "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" | base64)

# Create request
JSON_PAYLOAD=$(cat <<EOF
{
  "modelName": "${MODEL_NAME}",
  "matchPattern": "${MATCH_PATTERN}",
  "startDate": null,
  "unit": "CHARACTERS",
  "inputPrice": ${INPUT_PRICE},
  "outputPrice": ${OUTPUT_PRICE},
  "totalPrice": null,
  "tokenizerId": "openai",
  "tokenizerConfig": null
}
EOF
)

echo "Adding ${MODEL_NAME} to Langfuse..."

# Send request
RESPONSE=$(curl -s -w "\n%{http_code}" "${LANGFUSE_HOST}/api/public/models" \
    --request POST \
    --header 'Content-Type: application/json' \
    --header "Authorization: Basic ${AUTH_TOKEN}" \
    --data "${JSON_PAYLOAD}")

# Check result
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 201 ]; then
    echo "✅ Success!"
    echo "$BODY" | jq -r '.id, .modelName' 2>/dev/null
else
    echo "❌ Failed (HTTP ${HTTP_CODE})"
    echo "$BODY"
    exit 1
fi