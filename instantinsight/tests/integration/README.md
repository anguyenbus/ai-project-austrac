# Integration Tests

This directory contains integration tests and demonstration scripts for the NL2SQL pipeline.

## SQL Pipeline Evaluator

**File:** `sql_pipeline_evaluator.py`

### Purpose
The SQL Pipeline Evaluator tests the complete Natural Language to SQL generation pipeline by:
- Loading test queries from YAML files in `tests/data/validation/`
- Generating SQL from natural language questions using the Pipeline
- Executing and validating the generated SQL against Athena
- Calculating RAG metrics and performance statistics
- Saving detailed evaluation results to `pipeline_evaluation_results/`

### Usage
```bash
# Run the evaluation
poetry run python tests/integration/sql_pipeline_evaluator.py

# Import for custom testing
from tests.integration.sql_pipeline_evaluator import SQLPipelineEvaluator

evaluator = SQLPipelineEvaluator(
    use_pipeline=True,      # Use Pipeline with recovery refinement
    enable_rag_metrics=True # Calculate RAG evaluation metrics
)
results_df = evaluator.process_all_queries()
```

### Test Data
Test queries are stored in YAML files under `tests/data/validation/`:
- `first_10.yaml` - First 10 test queries
- `middle_10.yaml` - Middle 10 test queries  
- `sql_examples_custom_30_test.yaml` - 30 custom test examples

### Results
Evaluation results are saved to `pipeline_evaluation_results/` (gitignored):
- `validation_summary.csv` - Main evaluation results
- `query_{i}_expected.csv` - Expected query results
- `query_{i}_generated.csv` - Generated query results
- `rag_evaluation_results.csv` - Detailed RAG metrics

## Phase Demos

This directory also contains demonstration scripts that showcase the complete instantinsight query processing pipeline, from natural language input to SQL generation and execution.

## 🚀 Quick Start

### Run the Complete Query Flow Demo

```bash
# From the project root directory
./tests/integration/demo_query_flow.sh
```

## 🔍 Understanding the Output

The demo provides detailed output for each phase:

```
🚀 QUERY FLOW DEMONSTRATION
===============================================================================
📝 Input Query: 'show me sales by company'
⏰ Started at: 2024-01-15 10:30:45
===============================================================================

📊 PHASE 1: QUERY ANALYSIS & PREPROCESSING
============================================================
🔍 Step 1.1: Extracting table mentions...
   • Detected tables: ['orders', 'customers']
🎯 Step 1.2: Analyzing query intent...
   • Detected intents: ['aggregation', 'filtering']
⏱️ Phase 1 completed in 0.045s

🔄 PHASE 2: HYBRID RAG RETRIEVAL
============================================================
🔍 Step 2.1: pgvector Vector Similarity Search...
   • Retrieved 5 pgvector results
🕸️ Step 2.2: Neo4j Graph Traversal...
   • Retrieved 4 graph relationships
⏱️ Phase 2 completed in 0.123s

[... and so on for all 5 phases]
```

## ⚠️ Troubleshooting

### Common Issues

1. **"No database connection established"**
   - Ensure PostgreSQL is running: `pg_isready -h localhost -p 5432`
   - Check database credentials in `.env` file
   - Verify your_db_name database exists

2. **"Neo4j connection failed"**
   - Start Neo4j: `neo4j start` (or via Neo4j Desktop)
   - Check Neo4j is accessible: `curl http://localhost:7474`
   - Verify credentials in `.env` file

3. **"LLM provider error"**
   - For AWS Bedrock: Check AWS SSO login and profile
   - For Claude API: Verify ANTHROPIC_API_KEY
   - Check LLM_PROVIDER setting in `.env`

4. **"Permission denied" when running shell script**
   ```bash
   chmod +x tests/integration/demo_query_flow.sh
   ```

### Debug Mode

For more detailed debugging, run the Python script directly:

```bash
poetry run python tests/integration/demo_query_flow.py
# Then type your query when prompted
```

## 🧪 Testing

The demo includes built-in tests for:
- Database connectivity
- Neo4j availability
- SQL execution capabilities
- RAG system initialization

## 📈 Performance Metrics

The demo tracks and reports:
- Phase-by-phase execution times
- Retrieval result counts
- Context quality scores
- SQL generation success rates

## 🔗 Related Documentation

- [Knowledge Graph Setup](../README_KNOWLEDGE_GRAPH.md) - Complete setup guide
- [Project README](../README.md) - Main project documentation
- [Memory Bank](../memory-bank/) - Project context and progress
