#!/bin/bash

# Setup SchemaLogic with environment-driven configuration
# Requires: ATHENA_DATABASE in .env and datasets/tables.yml
#
# This script uses the refactored modular setup system.

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

print_info "SchemaLogic Setup Script (Refactored Version)"
print_info "Using modular components for better maintainability"
echo ""

# Parse command line arguments
GENERATE_EXAMPLES=""
SKIP_METADATA_UPDATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-examples)
            GENERATE_EXAMPLES="--generate-examples"
            shift
            ;;
        --validate-only)
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS --validate-only"
            shift
            ;;
        --skip-metadata-update)
            SKIP_METADATA_UPDATE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Usage: $0 [--generate-examples] [--validate-only] [--skip-metadata-update]"
            echo "  --generate-examples    Generate SQL examples using LLM"
            echo "  --validate-only        Only validate prerequisites"
            echo "  --skip-metadata-update Skip updating analyser metadata (join safety rules)"
            exit 1
            ;;
    esac
done

# Load environment variables from .env
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Please create .env with ATHENA_DATABASE variable."
    exit 1
fi

source .env

# Validate ATHENA_DATABASE is set
if [ -z "$ATHENA_DATABASE" ]; then
    print_error "ATHENA_DATABASE not set in .env"
    echo "Please add: ATHENA_DATABASE=\"your_database_name\""
    exit 1
fi

print_success "Environment validation passed"
print_info "ATHENA_DATABASE = '$ATHENA_DATABASE'"
echo ""

# Set force rebuild if not in validation mode
if [[ "$ADDITIONAL_ARGS" == *"--validate-only"* ]]; then
    FORCE_REBUILD=""
    print_info "Running in validation-only mode"
else
    FORCE_REBUILD="--force-rebuild"
    # Show what will happen based on arguments
    if [ "$GENERATE_EXAMPLES" = "--generate-examples" ]; then
        print_warning "SQL examples will be generated using LLM (requires Bedrock access)"
    else
        print_info "Will load existing examples from src/training/data/"
    fi
fi

# Use the setup script
print_info "Using SchemaLogic setup script"
COMMAND="python scripts/setup_schema_logic.py"

# Build the complete command
FULL_COMMAND="uv run $COMMAND --databases \"$ATHENA_DATABASE\" $FORCE_REBUILD $GENERATE_EXAMPLES $ADDITIONAL_ARGS"

print_info "Executing: $FULL_COMMAND"
echo ""

# Check if we're doing a dry run for testing
if [ "$DRY_RUN" = "true" ]; then
    print_info "DRY RUN MODE - Command would be executed but stopping here"
    exit 0
fi

# Run the setup
eval $FULL_COMMAND

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    print_success "Schema setup complete!"
    
    # Update analyser metadata after successful schema setup
    if [ "$SKIP_METADATA_UPDATE" = true ]; then
        print_info "Skipping analyser metadata update (--skip-metadata-update flag)"
    else
        echo ""
        print_info "Updating analyser metadata with join safety rules..."
        
        # Check if analyser_definitions.yaml exists
        if [ -f "config/analyser_definitions.yaml" ]; then
            uv run python scripts/update_analyser_metadata.py
            
            if [ $? -eq 0 ]; then
                print_success "Analyser metadata updated successfully"
            else
                print_warning "Failed to update analyser metadata (non-critical)"
                print_info "You can manually run: poetry run python scripts/update_analyser_metadata.py"
            fi
        else
            print_warning "config/analyser_definitions.yaml not found - skipping metadata update"
            print_info "To enable join safety, create config/analyser_definitions.yaml and run:"
            print_info "  uv run python scripts/update_analyser_metadata.py"
        fi
    fi
    
    echo ""
    print_success "Setup complete!"
    print_info "You can now use the hybrid database architecture with intelligent query routing."
    echo ""
    print_info "For more options run:"
    print_info "  python scripts/setup_schema_logic.py --help"
else
    echo ""
    print_error "Setup failed. Check the logs above for details."
    echo ""
    print_info "For troubleshooting, try running with verbose mode:"
    print_info "  $FULL_COMMAND --verbose"
    exit 1
fi
