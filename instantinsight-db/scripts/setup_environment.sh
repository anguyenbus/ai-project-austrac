#!/bin/bash

# Environment Setup Script for Athena Schema Extraction
# This script helps set up the necessary environment variables

echo "🚀 Athena Schema Extraction Environment Setup"
echo "============================================="
echo ""

# Function to prompt for input with default value
prompt_with_default() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -n "$prompt [$default]: "
    read input
    if [ -z "$input" ]; then
        input="$default"
    fi
    
    export $var_name="$input"
    echo "export $var_name=\"$input\"" >> .env.athena
}

# Create or clear the .env.athena file
> .env.athena
echo "# Athena Schema Extraction Environment Variables" >> .env.athena
echo "# Generated on $(date)" >> .env.athena
echo "" >> .env.athena

echo "Please provide the following configuration:"
echo ""

# AWS Configuration
echo "📡 AWS Configuration:"
prompt_with_default "AWS Default Region" "ap-southeast-2" "AWS_DEFAULT_REGION"
prompt_with_default "AWS Profile" "default" "AWS_PROFILE"
echo ""

# Athena Configuration
echo "🗄️  Athena Configuration:"
echo "⚠️  IMPORTANT: You need an S3 bucket for Athena query results"
echo "   Example: s3://my-company-athena-results/"
prompt_with_default "Athena S3 Staging Directory" "s3://your-bucket/athena-results/" "ATHENA_S3_STAGING_DIR"
prompt_with_default "Athena Database Name" "your_database" "ATHENA_DATABASE"
prompt_with_default "Athena Work Group" "primary" "ATHENA_WORK_GROUP"
prompt_with_default "Athena Query Timeout (seconds)" "60" "ATHENA_QUERY_TIMEOUT"
echo ""

# Integration Configuration
echo "🔗 Integration Configuration:"
prompt_with_default "Enable Athena Integration" "true" "ENABLE_ATHENA"
echo ""

# Add additional configuration to .env.athena
cat >> .env.athena << EOF

# PostgreSQL Configuration (for local RAG system)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=your_db_name

# RAG System Configuration
LLM_PROVIDER=bedrock
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
FORCE_REBUILD_VECTOR_STORE=false

# Neo4j support removed - using PostgreSQL with pgvector only
EOF

echo ""
echo "✅ Environment configuration saved to .env.athena"
echo ""
echo "Next steps:"
echo "1. Load the environment variables:"
echo "   source .env.athena"
echo ""
echo "2. Ensure AWS credentials are configured:"
echo "   aws configure"
echo ""
echo "3. Verify your S3 bucket exists and is accessible:"
echo "   aws s3 ls $ATHENA_S3_STAGING_DIR"
echo ""
echo "4. Start the local databases:"
echo "   docker-compose up -d"
echo ""
echo "5. Validate the setup:"
echo "   poetry run python scripts/setup_athena_rag.py --validate-only"
echo ""
echo "6. Run the complete setup:"
echo "   poetry run python scripts/setup_athena_rag.py --setup-all"
echo ""
echo "🎉 Setup complete! Your configuration is ready for Athena schema extraction."