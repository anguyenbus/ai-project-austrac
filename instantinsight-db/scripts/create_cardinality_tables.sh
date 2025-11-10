#!/bin/bash
#
# Setup RAG Cardinality System
#
# PURPOSE:
#   Initialises and populates the high-cardinality column processing system for RAG.
#   This script sets up database tables for storing categorical values from high-cardinality
#   columns and generates vector embeddings for semantic search capabilities.
#
# USAGE:
#   ./create_cardinality_tables.sh --config config/high_cardinality_columns.yaml
#
# OPTIONS:
#   --config <file>  Path to YAML configuration file specifying columns to process
#   -h, --help       Display this help message
#
# WORKFLOW:
#   1. Validates environment and configuration
#   2. Initialises database schema (creates rag_cardinality tables)
#   3. Extracts unique values from AWS Athena
#   4. Stores values in PostgreSQL with pgvector
#   5. Generates embeddings using AWS Bedrock
#
# REQUIREMENTS:
#   - PostgreSQL with pgvector extension
#   - AWS credentials configured for Athena access
#   - AWS Bedrock access for embedding generation
#   - Poetry for Python dependency management
#   - .env file with database configuration
#
# CONFIGURATION FILE FORMAT:
#   columns:
#     - table: table_name
#       column: column_name
#
# EXAMPLE:
#   ./create_cardinality_tables.sh --config config/high_cardinality_columns.yaml
#
# ERROR HANDLING:
#   - Validates configuration file existence
#   - Checks for required environment variables
#   - Attempts schema fixes if initial setup fails
#   - Provides detailed error messages for troubleshooting
#
# SEE ALSO:
#   - scripts/create_rag_cardinality.py - Python implementation
#   - scripts/init_cardinality_schema.py - Schema initialisation
#   - docs/data/cardinality/CARDINALITY_COLUMNS.md - Full documentation
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly SCHEMA_SQL="${PROJECT_ROOT}/schemas/pgvector-cardinality.sql"

# Logging
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
error() { log "ERROR: $*" >&2; exit 1; }

# Parse arguments
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --config <config_file.yaml>"
            echo "Create and populate RAG cardinality tables from configuration"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validation
[[ -z "$CONFIG_FILE" ]] && error "Config file required: --config <file>"
[[ ! -f "$CONFIG_FILE" ]] && error "Config file not found: $CONFIG_FILE"

# Navigate to project root
cd "$PROJECT_ROOT"

# Load environment
[[ -f ".env" ]] || error ".env file not found"
source .env

# Create/update database schema
log "Initialising database schema..."
poetry run python scripts/init_cardinality_schema.py || {
    log "Schema initialisation failed, attempting fix..."
    poetry run python scripts/fix_schema.py || error "Schema setup failed"
}

# Process cardinality data
log "Processing cardinality data from: $CONFIG_FILE"
poetry run python scripts/create_rag_cardinality.py --config "$CONFIG_FILE" || \
    error "Cardinality processing failed"

log "✓ RAG cardinality system ready"