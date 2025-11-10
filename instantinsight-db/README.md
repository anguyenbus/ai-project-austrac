# instantinsight-DB: Universal Database RAG System

A production-ready RAG (Retrieval-Augmented Generation) system that enables intelligent SQL generation from natural language queries across any database platform. Built with universal database connectivity, the system extracts schemas from multiple backends (Athena, PostgreSQL, Snowflake, BigQuery, etc.), creates searchable vector embeddings, and generates accurate SQL using LLMs.

## Overview

instantinsight-DB provides a complete pipeline for text-to-SQL conversion with semantic understanding:

- **Universal Database Support**: Connect to any Ibis-supported backend with a single connection string
- **Intelligent Schema Extraction**: Automatically discover and vectorize database schemas
- **LLM-Powered SQL Generation**: Generate and validate SQL queries using AWS Bedrock Claude models
- **Advanced Vector Search**: PostgreSQL + pgvector for high-performance semantic retrieval
- **Semantic Caching**: Redis-based intelligent query result caching
- **Production-Ready**: Comprehensive testing, type safety, and security controls

## Architecture

### Core Technologies

- **PostgreSQL + pgvector**: Vector embeddings storage with optimized similarity search
- **Redis Stack**: Semantic caching with intelligent deduplication
- **Ibis Framework**: Universal database connectivity layer supporting 6+ backends
- **AWS Bedrock**: LLM integration (Claude Sonnet 4) and embeddings (Titan v2)
- **Alembic**: Version-controlled database migrations
- **Docker Compose**: Containerized development environment

### Key Components

#### 1. Universal Database Connectivity

**Backend-Agnostic Architecture** powered by Ibis Framework:

- **AnalyticsConnector** ([`src/connectors/analytics_backend.py`](src/connectors/analytics_backend.py)): Universal database backend supporting Athena, PostgreSQL, Snowflake, BigQuery, Redshift, Databricks, and more
- **SchemaIntrospector** ([`src/utils/schema_introspector.py`](src/utils/schema_introspector.py)): Platform-independent schema extraction with optional cloud provider enrichment
- **GlueEnricher** ([`src/utils/glue_enricher.py`](src/utils/glue_enricher.py)): Optional AWS Glue metadata enrichment for Athena backends (S3 locations, partition keys, descriptions)

**Supported Backends**:

- AWS Athena
- PostgreSQL
- Snowflake
- BigQuery
- Redshift
- Databricks
- Any Ibis-compatible database

See [Universal Database Connectivity Guide](docs/architecture/UNIVERSAL_DATABASE_CONNECTIVITY.md) for complete architecture details.

#### 2. RAG System

**Enhanced PostgreSQL-Based RAG** ([`src/rag/pgvector_rag.py`](src/rag/pgvector_rag.py)):

- Document chunking strategies for improved retrieval granularity
- Hybrid search combining vector similarity with keyword matching
- Namespace support for multiple embedding strategies
- Optimized database functions for performance
- Backward compatibility with legacy document-level mode

**RAG Engine** ([`src/rag/rag_engine.py`](src/rag/rag_engine.py)):

- Clean interface for schema and training example management
- Integration with Bedrock embeddings and LLMs
- Comprehensive statistics and monitoring

#### 3. Setup Orchestration

**Modular Setup Pipeline** ([`src/setup/setup_orchestrator.py`](src/setup/setup_orchestrator.py)):

- **SetupOrchestrator**: Main workflow coordination with error handling
- **ConfigLoader**: Environment-based configuration management
- **PrerequisiteValidator**: System health checks and dependency validation
- **AnalyserFilter**: Intelligent table filtering and selection logic

#### 4. Schema Vectorization

**Schema Processing Pipeline** ([`src/utils/vectorizer/`](src/utils/vectorizer/)):

- **SchemaVectorizerOrchestrator**: Coordinates DDL generation, SQL examples, and RAG integration
- **AthenaDDLGenerator**: Creates comprehensive schema documentation
- **SQLExampleGenerator**: LLM-powered SQL example generation with validation
- **YAMLDataExporter**: Training data export and management
- **TemplateLoader**: Structured prompt templates for consistent LLM interactions

#### 5. Database Models

**Alembic-Managed Schema** ([`src/database/models.py`](src/database/models.py)):

- `rag_documents`: Main document storage with metadata
- `rag_chunks`: Document chunks with type classification
- `rag_embeddings`: Vector embeddings with namespace support
- `rag_training_examples`: Generated SQL examples with validation metadata
- `rag_cardinality_columns`: High-cardinality categorical column tracking
- `rag_cardinality`: Categorical value embeddings for semantic search

## Quick Start

### Prerequisites

- Python 3.10+ (3.11+ recommended)
- Docker and Docker Compose
- AWS credentials configured (for Athena/Bedrock)
- uv package manager ([installation guide](https://docs.astral.sh/uv/))

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd instantinsight-db
```

2. **Install dependencies** using `uv`:

```bash
uv sync --all-extras
```

3. **Start infrastructure** services:

```bash
chmod +x start_up_container.sh
./start_up_container.sh
```

4. **Configure environment** variables:

```bash
cp .env.example .env
# Edit .env with your settings (see Configuration section)
```

5. **Run schema setup** pipeline:

```bash
chmod +x scripts/setup_schema_logic.sh
./scripts/setup_schema_logic.sh
```

### Configuration

Copy `.env.example` to `.env` and configure:

#### Core Settings

```env
# Universal Analytics Database URL (supports any backend)
# Athena:
ANALYTICS_DB_URL=athena://awsdatacatalog?region=ap-southeast-2&database=warehouse&work_group=primary&s3_staging_dir=s3://bucket/results/

# PostgreSQL:
# ANALYTICS_DB_URL=postgres://user:password@localhost:5432/analytics

# Snowflake:
# ANALYTICS_DB_URL=snowflake://account/database?warehouse=compute

# BigQuery:
# ANALYTICS_DB_URL=bigquery://project/dataset

# AWS Configuration
AWS_PROFILE=default
AWS_REGION=ap-southeast-2

# PostgreSQL Vector Store
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=instantinsight
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/instantinsight

# AWS Bedrock Models
BEDROCK_MODEL=apac.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Usage

### Basic Schema Setup

```bash
# Setup with default database
uv run python scripts/setup_schema_logic.py

# Setup specific databases
uv run python scripts/setup_schema_logic.py --databases analytics,warehouse

# Force rebuild everything
uv run python scripts/setup_schema_logic.py --force-rebuild
```

### Generate SQL Examples with LLM

```bash
# Generate examples for all tables
uv run python scripts/setup_schema_logic.py --setup-all --generate-examples

# Generate examples for specific tables
uv run python scripts/setup_schema_logic.py --table-names customers,orders --generate-examples

# Update with new tables only (efficient)
uv run python scripts/setup_schema_logic.py --update-new-tables-only --generate-examples
```

### Incremental Updates

```bash
# Add new tables without affecting existing data
uv run python scripts/setup_schema_logic.py --update

# Update only tables not in system
uv run python scripts/setup_schema_logic.py --update-new-tables-only
```

### High-Cardinality Column Processing

```bash
# Setup cardinality tables and process categorical columns
./scripts/create_cardinality_tables.sh --config config/high_cardinality_columns.yaml

# Or use Python script for more control
uv run python scripts/create_rag_cardinality.py --config config/high_cardinality_columns.yaml
```

## Database Schema

The system uses Alembic migrations to manage these core tables:

### Vector Storage Tables

- **`rag_documents`**: Main document storage with full content and metadata
- **`rag_chunks`**: Document chunks with type classification for granular retrieval
- **`rag_embeddings`**: 1024-dimensional Titan v2 embeddings with namespace support

### Training and Metadata Tables

- **`rag_training_examples`**: LLM-generated SQL examples with validation status
- **`rag_cardinality_columns`**: Tracks high-cardinality categorical columns
- **`rag_cardinality`**: Categorical values with vector embeddings for semantic search

### Migration Management

```bash
# Apply pending migrations
uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "description"

# Rollback migration
uv run alembic downgrade -1
```

## Development

### Code Quality Standards

This project enforces strict code quality through:

- **Ruff**: Fast Python linter and formatter (rules: E, F, I, B, UP, D)
- **Type Hints**: Comprehensive type annotations with `beartype` runtime validation
- **Contracts**: `icontract` for preconditions and postconditions
- **Google-style Docstrings**: Complete documentation for all public interfaces

```bash
# Run linting
uv run ruff check --select E,F,I,B,UP,D --ignore E501,D203,D212

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test suites
uv run pytest tests/connectors/
uv run pytest tests/integration/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```
