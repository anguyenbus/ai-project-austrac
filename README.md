# InstantInsight: Natural Language to SQL Query System

A production-ready system that converts natural language questions into executable SQL queries using a sophisticated multi-agent architecture powered by AWS Bedrock. The system combines intelligent schema understanding, semantic caching, and automatic error recovery to deliver accurate results across any database platform.

## Project Structure

This repository contains two integrated components:

- **[`instantinsight/`](instantinsight/)**: Core application layer with multi-agent query processing, Flask UI, Lambda deployment, and session management
- **[`instantinsight-db/`](instantinsight-db/)**: Database layer providing universal connectivity, RAG-based schema management, and vector search capabilities

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
│                "How many products are discontinued?"             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────┐
│                   instantinsight (Application Layer)             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Multi-Agent Pipeline Coordinator                        │   │
│  │  • Query validation & normalization                      │   │
│  │  • Semantic cache lookup (pgvector)                      │   │
│  │  • 11 specialized agents working together                │   │
│  │  • SQL generation with error recovery                    │   │
│  │  • Visualization recommendations                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Session Management (S3-backed)                          │   │
│  │  • Multi-turn conversations                              │   │
│  │  • Context preservation                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────┐
│               instantinsight-db (Database Layer)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Universal Database Connectivity (Ibis)                  │   │
│  │  • Athena, PostgreSQL, Snowflake, BigQuery, Redshift     │   │
│  │  • Schema introspection & enrichment                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RAG System (PostgreSQL + pgvector)                      │   │
│  │  • Vector embeddings (AWS Titan v2)                      │   │
│  │  • Document chunking strategies                          │   │
│  │  • Hybrid search (semantic + keyword)                    │   │
│  │  • Training example generation                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────┐
│                     Analytics Database                           │
│         (Athena, PostgreSQL, Snowflake, BigQuery, etc.)          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Multi-Agent Intelligence

- **11 Specialized Agents**: Query validation, schema understanding, SQL generation, error correction, and visualization
- **Intelligent Collaboration**: Agents work in coordinated stages with feedback loops
- **Cost Optimization**: Tiered model selection (Nova Micro → Haiku → Sonnet → Nova Pro)

### Universal Database Support

- **Platform Agnostic**: Connect to any Ibis-supported backend with a single connection string
- **6+ Database Platforms**: Athena, PostgreSQL, Snowflake, BigQuery, Redshift, Databricks
- **Cloud Provider Enrichment**: Optional AWS Glue metadata integration

### RAG-Powered Schema Understanding

- **Automatic Discovery**: Extract and vectorize database schemas
- **Semantic Search**: pgvector-based similarity matching for relevant tables and columns
- **Training Examples**: LLM-generated SQL examples with validation
- **High-Cardinality Handling**: Categorical column embeddings for semantic search

### Production-Ready Features

- **Semantic Caching**: 70%+ hit rate with Redis-based intelligent deduplication
- **Multi-Turn Conversations**: S3-backed session management with contextual understanding
- **Error Recovery**: Automatic SQL correction and retry logic with validation
- **Observability**: Full tracing with Langfuse integration
- **Security**: SQL injection prevention, schema validation, and guardrails

## Quick Start

### Prerequisites

- **Python 3.12+** (3.11+ recommended)
- **Docker & Docker Compose** (for local infrastructure)
- **AWS Credentials** (for Bedrock LLMs and optionally Athena)
- **uv Package Manager** ([installation guide](https://docs.astral.sh/uv/))

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd ai-project-austrac
```

2. **Install dependencies** for both projects:

```bash
# instantinsight-db
cd instantinsight-db
uv sync --all-extras --dev

# instantinsight
cd ../instantinsight
uv sync --all-extras --dev
```

3. **Start infrastructure services**:

```bash
# From instantinsight-db directory
chmod +x start_up_container.sh
./start_up_container.sh
```

This starts:

- PostgreSQL with pgvector extension (port 5432)
- Redis Stack for semantic caching (port 6379)
- Langfuse for observability (port 3000)

4. **Configure environment variables**:

**instantinsight-db/.env**:

```bash
# Analytics Database (supports any Ibis backend)
ANALYTICS_DB_URL=athena://awsdatacatalog?region=ap-southeast-2&database=warehouse&work_group=primary&s3_staging_dir=s3://bucket/results/

# PostgreSQL Vector Store
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=instantinsight
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/instantinsight

# AWS Bedrock
AWS_PROFILE=default
AWS_REGION=ap-southeast-2
BEDROCK_MODEL=apac.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
```

**instantinsight/.env**:

```bash
# Copy similar settings from instantinsight-db/.env
# Add additional settings for session management:
S3_SESSION_BUCKET=your-session-bucket
```

5. **Initialize the RAG knowledge base**:

```bash
# From instantinsight-db directory
chmod +x scripts/setup_schema_logic.sh
./scripts/setup_schema_logic.sh
```

This will:

- Extract your database schema
- Create vector embeddings
- Generate training SQL examples
- Store everything in PostgreSQL

### Basic Usage

**Python API**:

```python
from src.query_processor import NL2SQLProcessor

# Initialize processor
processor = NL2SQLProcessor(
    use_pipeline=True,    # Enable multi-agent pipeline
    enable_cache=True     # Enable semantic caching
)

# Process query
result = processor.process_query("How many products are discontinued?")
processor.display_results(result)
```

**Flask Web Interface**:

```bash
# From instantinsight directory
cd flask_app
uv run python run_flask.py
```

Visit http://localhost:5000 for the interactive web interface.

**Lambda Deployment**:

```bash
# From instantinsight/lambda directory
make build
make test-local
```

See [`instantinsight/lambda/README.md`](instantinsight/lambda/README.md) for deployment details.

## Example Queries

```python
# Analytics
"Show top 10 products by revenue"
"List employees by department with salary"
"Calculate average order value by month"

# Reporting
"How many active users in the last 30 days?"
"What is the conversion rate by traffic source?"
"Show inventory levels below reorder point"

# Data Exploration
"What tables contain customer information?"
"Show sample data from orders table"
"List all columns in the products table"
```

## Performance & Cost

### Latency

- **Cache Hit**: 10-50ms (no LLM calls)
- **Simple Query**: 1-3 seconds (3-5 agents)
- **Complex Query**: 3-8 seconds (8-12 agents)

### Cost Optimization

- **Semantic Cache**: 70% hit rate = 70% cost reduction
- **Tiered Models**: Use cheapest model for each task
  - Nova Micro: Formatting ($0.035/1M tokens)
  - Claude Haiku: Validation ($0.25/1M tokens)
  - Nova Pro: Schema mapping ($3/1M tokens)
  - Claude Sonnet: Complex SQL ($3/1M tokens)

### Typical Query Costs

| Query Type | Without Cache          | With Cache (70% hit) |
| ---------- | ---------------------- | -------------------- |
| Simple     | $0.05         | $0.015 |                      |
| Complex    | $0.15         | $0.045 |                      |
| Multi-join | $0.30         | $0.090 |                      |

## Technology Stack

### Core Technologies

- **Python 3.12+** with uv package manager
- **PostgreSQL + pgvector** - Vector storage and RAG engine
- **Redis Stack** - Semantic caching
- **Ibis Framework** - Universal database connectivity
- **AWS Bedrock** - LLM integration (Claude Sonnet, Haiku, Nova)

### AI/ML Frameworks

- **LangChain** - LLM orchestration
- **Strands** - Agent framework
- **Instructor** - Structured outputs
- **AWS Titan v2** - Text embeddings

### Infrastructure

- **Docker Compose** - Local development environment
- **Alembic** - Database migrations
- **AWS Lambda** - Serverless deployment
- **S3** - Session storage

### Monitoring & Observability

- **Langfuse** - LLM tracing and cost tracking
- **Loguru** - Structured logging

## Development

### Code Quality Standards

Both projects enforce strict quality through:

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
# Run all tests (from each project directory)
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Specific test suites
uv run pytest tests/agents/
uv run pytest tests/integration/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Documentation

### instantinsight (Application Layer)

- [Architecture Overview](instantinsight/docs/architecture.md)
- [Agent Workflow](instantinsight/docs/agent-workflow.md)
- [Cache System](instantinsight/docs/cache-system.md)
- [Multi-Turn Conversations](instantinsight/docs/multi-turn-conversations.md)
- [Lambda Deployment](instantinsight/lambda/README.md)

### instantinsight-db (Database Layer)

- [Universal Database Connectivity](instantinsight-db/docs/architecture/UNIVERSAL_DATABASE_CONNECTIVITY.md)
- [RAG System Architecture](instantinsight-db/README.md#architecture)
- [Setup Orchestration](instantinsight-db/README.md#setup-orchestration)
- [Schema Vectorization](instantinsight-db/README.md#schema-vectorization)

## Project Structure

```
ai-project-austrac/
├── README.md                      # This file
├── instantinsight/                # Application layer
│   ├── src/
│   │   ├── agents/               # 11 specialized agents
│   │   ├── rag/                  # RAG engine & pipeline
│   │   ├── cache/                # Semantic caching
│   │   ├── session/              # S3-backed sessions
│   │   ├── config/               # Configuration
│   │   └── utils/                # Utilities
│   ├── flask_app/                # Web interface
│   ├── lambda/                   # AWS Lambda deployment
│   ├── tests/                    # Comprehensive test suite
│   └── pyproject.toml            # Dependencies
│
└── instantinsight-db/            # Database layer
    ├── src/
    │   ├── connectors/           # Universal database backends
    │   ├── rag/                  # RAG components
    │   ├── setup/                # Setup orchestration
    │   └── utils/                # Schema processing
    ├── alembic/                  # Database migrations
    ├── config/                   # Analyser definitions
    ├── schemas/                  # SQL schemas
    ├── scripts/                  # Setup scripts
    ├── tests/                    # Test suite
    └── pyproject.toml            # Dependencies
```
