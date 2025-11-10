#!/bin/bash

# Generate SQL examples for specific tables in the RAG system
#
# Usage:
#   ./scripts/generate_table_examples.sh table1,table2,table3
#   ./scripts/generate_table_examples.sh "table1, table2, table3"
#   ./scripts/generate_table_examples.sh reconciliation_account_group_ledger_transaction
#
# Examples:
#   ./scripts/generate_table_examples.sh reconciliation_account_group_ledger_transaction,expense_line_system_type
#   ./scripts/generate_table_examples.sh "agreement_charges_billing, agreement_charges_billing_rate"

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${CYAN}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if table names were provided
if [ $# -eq 0 ]; then
    print_error "No table names provided!"
    echo ""
    echo "Usage: $0 <table_names>"
    echo ""
    echo "Examples:"
    echo "  $0 reconciliation_account_group_ledger_transaction"
    echo "  $0 expense_line_system_type,agreement_charges_billing"
    echo "  $0 \"table1, table2, table3\""
    echo ""
    echo "Available options:"
    echo "  --database <name>    Specify database (default: from .env ATHENA_DATABASE)"
    echo "  --no-examples        Skip example generation (only add schemas)"
    echo "  --help               Show this help message"
    exit 1
fi

# Parse command line arguments
TABLE_NAMES=""
DATABASE=""
GENERATE_EXAMPLES="--generate-examples"

while [[ $# -gt 0 ]]; do
    case $1 in
        --database)
            DATABASE="$2"
            shift 2
            ;;
        --no-examples)
            GENERATE_EXAMPLES=""
            shift
            ;;
        --help|-h)
            echo "Generate SQL examples for specific tables"
            echo ""
            echo "Usage: $0 [OPTIONS] <table_names>"
            echo ""
            echo "Options:"
            echo "  --database <name>    Specify database (default: from .env)"
            echo "  --no-examples        Skip example generation"
            echo "  --help, -h          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 table1,table2"
            echo "  $0 --database mydb table1"
            echo "  $0 --no-examples table1,table2"
            exit 0
            ;;
        *)
            # Assume this is the table names argument
            TABLE_NAMES="$1"
            shift
            ;;
    esac
done

# Validate table names were provided
if [ -z "$TABLE_NAMES" ]; then
    print_error "No table names provided!"
    exit 1
fi

# Remove spaces from table names for consistent formatting
TABLE_NAMES=$(echo "$TABLE_NAMES" | tr -d ' ')

print_info "Table Example Generator"
echo ""

# Load environment variables
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Please create .env with ATHENA_DATABASE variable."
    exit 1
fi

source .env

# Use provided database or fall back to environment variable
if [ -z "$DATABASE" ]; then
    DATABASE="$ATHENA_DATABASE"
fi

# Validate database is set
if [ -z "$DATABASE" ]; then
    print_error "Database not specified and ATHENA_DATABASE not set in .env"
    echo "Please either:"
    echo "  1. Add ATHENA_DATABASE=\"your_database\" to .env"
    echo "  2. Use --database flag: $0 --database mydb $TABLE_NAMES"
    exit 1
fi

print_success "Configuration validated"
print_info "Database: $DATABASE"
print_info "Tables: $TABLE_NAMES"
if [ -n "$GENERATE_EXAMPLES" ]; then
    print_info "Mode: Generate SQL examples using LLM"
else
    print_info "Mode: Schema only (no examples)"
fi
echo ""

# Build the command
COMMAND="poetry run python scripts/setup_schema_logic.py"
COMMAND="$COMMAND --databases \"$DATABASE\""
COMMAND="$COMMAND --table-names \"$TABLE_NAMES\""
COMMAND="$COMMAND $GENERATE_EXAMPLES"
COMMAND="$COMMAND --demo-mode"  # Skip S3 staging directory validation

print_info "Executing command:"
echo "  $COMMAND"
echo ""

# Track start time
START_TIME=$(date +%s)

# Execute the command
if eval $COMMAND; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    print_success "Example generation completed successfully!"
    print_info "Duration: ${DURATION} seconds"
    
    # Show where examples were saved
    echo ""
    print_info "Generated examples have been:"
    print_info "  • Stored in PostgreSQL vector database (RAG system)"
    print_info "  • Exported to src/training/data/sql_examples_${DATABASE}_*.yaml"
    
    # Count generated files
    FILE_COUNT=$(ls -1 src/training/data/sql_examples_${DATABASE}_*.yaml 2>/dev/null | wc -l)
    if [ $FILE_COUNT -gt 0 ]; then
        echo ""
        print_info "Recent example files:"
        ls -lt src/training/data/sql_examples_${DATABASE}_*.yaml 2>/dev/null | head -5 | awk '{print "  • "$9}'
    fi
else
    echo ""
    print_error "Example generation failed!"
    print_info "Check the logs above for details"
    print_info "Common issues:"
    print_info "  • Table names not found in database"
    print_info "  • AWS credentials not configured"
    print_info "  • Database connection issues"
    exit 1
fi