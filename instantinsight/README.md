# instantinsight

**Natural Language to SQL with Multi-Agent Architecture**

instantinsight converts natural language queries into SQL statements using a sophisticated multi-agent system powered by AWS Bedrock models. The system features semantic caching, intelligent error recovery, and automatic visualization recommendations.

## Overview

```
User: "How many products are discontinued?"
   |
   v
___________________________________________________
|  Multi-Agent Pipeline                            |
|  • Query validation and normalization           |
|  • Semantic cache lookup                        |
|  • Schema understanding                         |
|  • SQL generation and validation                |
|  • Execution with error recovery                |
|  • Visualization recommendations                |
|__________________________________________________|
   |
   v
Result: SELECT COUNT(*) FROM products WHERE discontinued = 1
        -> 29 products
```

## Key Features

- **Multi-Agent System**: 11 specialized agents working together
- **S3-Backed Session Management**: Server-managed conversation history with automatic persistence
- **Multi-Turn Conversations**: Natural follow-up queries with contextual understanding
- **Semantic Caching**: 70%+ cache hit rate reduces costs and latency
- **Intelligent Error Recovery**: Automatic SQL correction and retry logic
- **Schema Understanding**: Automatically maps natural language to database schema
- **Cost Optimized**: Tiered model selection (Nova Micro -> Haiku -> Sonnet)
- **Visualization**: Automatic chart recommendations with Plotly JSON
- **Observability**: Full tracing with Langfuse integration

## Architecture

### System Components

```
_________________________________________________________________
|                         User Query                             |
|________________________________________________________________|
                             |
                             v
_________________________________________________________________
|                    Pipeline Coordinator                        |
|                                                                |
|  Stage 1: Query Validation    (QueryNormalizer, IntentValidator)
|  Stage 2: Cache Lookup        (Semantic Cache with pgvector)  |
|  Stage 3: SQL Generation      (Multiple specialized agents)   |
|  Stage 4: Execution           (RAG Engine + error recovery)   |
|________________________________________________________________|
                             |
            _________________|_________________
            |                |                |
            v                v                v
    _______________  _______________  _______________
    |   Agents    |  |  RAG Engine |  |    Cache    |
    |  (11 types) |  |  (Custom)   |  |  (Semantic) |
    |_____________|  |_____________|  |_____________|
```

### Agent Collaboration

instantinsight uses 11 specialized agents that collaborate in stages:

**Query Processing**
1. `QueryIntentValidator` - Validates intent and security
2. `QueryNormalizer` - Normalizes to canonical form
3. `ClarificationAgent` - Generates help messages

**Schema Understanding**
4. `SchemaTableSelector` - Selects relevant tables
5. `SchemaColumnMapper` - Maps terms to columns
6. `SchemaFilterBuilder` - Builds WHERE clauses

**SQL Generation**
7. `SQLGenerator` - Main SQL generation (Claude Sonnet)
8. `SchemaValidator` - Validates table/column existence
9. `SQLFormatter` - Formats SQL for readability

**Execution & Output**
10. `SQLCorrector` - Fixes errors and retries
11. `OutputVisualizer` - Generates chart recommendations

See [Agent Workflow Documentation](docs/agent-workflow.md) for detailed interaction diagrams.

## Quick Start

### Installation

**Prerequisites**: Ensure these services are running locally:
- PostgreSQL with pgvector extension (port 5432)
- Redis (port 6379)
- Langfuse (port 3000)

```bash
# Install dependencies
uv sync --all-extras --dev

# Configure environment
cp .env.example .env
# Edit .env with your database and AWS credentials

# Initialize RAG knowledge base
uv run python scripts/init_database.py
```

### Basic Usage

```python
from simple_app import NL2SQLProcessor

# Initialize processor
processor = NL2SQLProcessor(
    use_pipeline=True,    # Enable multi-agent pipeline
    enable_cache=True     # Enable semantic caching
)

# Process query
result = processor.process_query("How many products are discontinued?")
processor.display_results(result)
```

### Example Output

```
Cache Hit (confidence: 0.987):
   Retrieved in 23ms

Generated SQL:
----------------------------------------
SELECT COUNT(*) FROM products WHERE discontinued = 1

Results (1 rows):
----------------------------------------
   count
0     29

Visualization: bar chart (confidence: 85%)
```

## Performance

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

| Query Type | Without Cache | With Cache (70% hit) |
|------------|---------------|----------------------|
| Simple | $0.05 | $0.015 |
| Complex | $0.15 | $0.045 |
| Multi-join | $0.30 | $0.090 |

## Configuration

### Environment Variables

```bash
# RAG Database (PostgreSQL with pgvector)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Analytics Database (Universal Backend via Ibis)
# Supports: Athena, PostgreSQL, Snowflake, BigQuery, etc.
ANALYTICS_DB_URL=athena://?schema_name=text_to_sql&work_group=primary&region_name=ap-southeast-2&s3_staging_dir=s3://your-bucket/

# Alternative examples:
# PostgreSQL: ANALYTICS_DB_URL=postgres://user:password@localhost:5432/database
# Snowflake: ANALYTICS_DB_URL=snowflake://account/database?warehouse=compute
# BigQuery: ANALYTICS_DB_URL=bigquery://project-id/dataset

# AWS Bedrock
AWS_REGION=ap-southeast-2
BEDROCK_MODEL_ID=apac.anthropic.claude-sonnet-4-20250514-v1:0

# Optional: Override specific agents
BEDROCK_MODEL_SQLGENERATOR=amazon.nova-pro-v1:0
BEDROCK_TEMP_SQLGENERATOR=0.3

# Monitoring
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
```

### Model Selection

Modify agent models in `src/config/agent_model_config.py`:

```python
AGENT_CONFIGS = {
    "SQLGenerator": {
        "model_id": "amazon.nova-pro-v1:0",  # Change to Nova Pro (cheaper)
        "temperature": 0.3,
        "max_tokens": 6000,
    },
    "QueryNormalizer": {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",  # Change to Haiku
        "temperature": 0.1,
        "max_tokens": 5000,
    },
}
```

## Documentation

- **[Architecture Overview](docs/architecture.md)** - System design and components
- **[Agent Workflow](docs/agent-workflow.md)** - How agents collaborate (with diagrams)
- **[Getting Started](docs/getting-started.md)** - Installation and setup guide
- **[Cache System](docs/cache-system.md)** - Semantic caching deep dive
- **[Multi-Turn Conversations](docs/multi-turn-conversations.md)** - Session management and contextual queries
- **[S3 Session Plan](docs/s3-session-plan.md)** - S3-backed session architecture details

## Technology Stack

### Core
- **Python 3.12+** with uv
- **PostgreSQL** with pgvector extension (for RAG database)
- **Ibis Framework** - Universal database connectivity (Athena, PostgreSQL, Snowflake, BigQuery, etc.)
- **AWS Bedrock** (Claude Sonnet, Claude Haiku, Nova Pro/Micro)

### AI/ML
- **LangChain** - LLM orchestration
- **Strands** - Agent framework
- **Instructor** - Structured outputs

### Monitoring
- **Langfuse** - Observability and tracing
- **Loguru** - Structured logging

### Lambda Functions

This project uses AWS Lambda for serverless computation. The lambdas are organized into a modular structure that promotes code reuse and simplified maintenance.

- **`lambdas/shared/`**: Contains common infrastructure, base handlers, and utility code shared across multiple Lambda functions. See the [Shared Lambda Infrastructure README](lambdas/shared/README.md) for more details.
- **`lambdas/<lambda-name>/`**: Each individual Lambda function has its own directory, containing its specific handler code and any unique dependencies.

This structure allows for a clear separation of concerns and streamlines development and deployment. For a detailed explanation of the architecture, see the [Lambda Architecture documentation](docs/lambda-architecture.md).

## Development

### Code Style

- Maximum function length: 20 lines
- Maximum file length: 200 lines
- Simple, direct solutions
- No complex abstractions
- Type hints where helpful

### Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src tests/

# Linting
uv run ruff check .
uv run ruff format .
```

### Adding New Agents

1. Create agent in `src/agents/strand_agents/`
2. Add configuration in `src/config/agent_model_config.py`
3. Create prompts in `src/agents/prompt_builders/`
4. Integrate into pipeline handlers

See [Architecture Documentation](docs/architecture.md) for detailed extension points.

## Monitoring

### Langfuse Integration

All queries are traced in Langfuse:
- Agent invocations
- Token usage per agent
- Costs per query
- Cache hit/miss rates
- Error tracking

### Debug Mode

```python
processor = NL2SQLProcessor(debug_mode=True)
```

Logs:
- Agent inputs/outputs
- Pipeline stage transitions
- Cache lookups
- SQL generation steps

## Common Use Cases

### Analytics Queries

```python
"Show top 10 products by revenue"
"List employees by department with salary"
"Calculate average order value by month"
```

### Reporting

```python
"How many active users in the last 30 days?"
"What is the conversion rate by traffic source?"
"Show inventory levels below reorder point"
```

### Data Exploration

```python
"What tables contain customer information?"
"Show sample data from orders table"
"List all columns in the products table"
```

## Contributing

Contributions welcome! Please:
1. Follow code style guidelines
2. Add tests for new features
3. Update documentation
4. Keep functions under 20 lines
