#!/usr/bin/env python3
"""
Test Script for Chunked PgVector Architecture.

This script validates the new chunked architecture by:
1. Testing chunking strategies
2. Validating database operations
3. Testing search functionality
4. Performance benchmarking

Usage:
    poetry run python scripts/test_chunked_architecture.py --test-chunking
    poetry run python scripts/test_chunked_architecture.py --test-database
    poetry run python scripts/test_chunked_architecture.py --test-search
    poetry run python scripts/test_chunked_architecture.py --benchmark
    poetry run python scripts/test_chunked_architecture.py --all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger  # noqa: E402

from src.config.database_config import DATABASE_ROUTING_CONFIG  # noqa: E402
from src.rag.components.chunking_strategies import (  # noqa: E402
    chunk_document,
)
from src.rag.pgvector_rag import PgvectorRAG  # noqa: E402


class ChunkedArchitectureTester:
    """Test suite for the chunked pgvector architecture."""

    def __init__(self):
        """Initialize the test suite with empty result containers."""
        self.test_results = {
            "chunking_tests": {},
            "database_tests": {},
            "search_tests": {},
            "benchmark_results": {},
        }

        # Test data
        self.test_documents = {
            "schema": {
                "content": """
CREATE EXTERNAL TABLE `text_to_sql`.`sales_transactions` (
  `transaction_id` bigint COMMENT 'Unique identifier for each transaction',
  `customer_id` bigint COMMENT 'Customer identifier',
  `product_id` bigint COMMENT 'Product identifier', 
  `transaction_date` date COMMENT 'Date of transaction',
  `amount` double COMMENT 'Transaction amount',
  `currency` string COMMENT 'Currency code',
  `payment_method` string COMMENT 'Payment method used',
  `status` string COMMENT 'Transaction status',
  `created_at` timestamp COMMENT 'Record creation timestamp',
  `updated_at` timestamp COMMENT 'Record update timestamp',
  `discount_amount` double COMMENT 'Discount applied',
  `tax_amount` double COMMENT 'Tax amount',
  `total_amount` double COMMENT 'Total amount including tax'
)
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://my-bucket/sales-transactions/'
TBLPROPERTIES (
  'has_encrypted_data'='false',
  'classification'='sales_data',
  'department'='finance'
)
                """,
                "metadata": {"table_name": "sales_transactions", "source": "athena"},
            },
            "sql_example": {
                "content": json.dumps(
                    {
                        "question": "Show me total sales by customer for the last 30 days with customer details",
                        "sql": """
WITH recent_sales AS (
    SELECT 
        customer_id,
        SUM(amount) as total_amount,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_transaction
    FROM sales_transactions 
    WHERE transaction_date >= date_sub(current_date(), 30)
        AND status = 'completed'
    GROUP BY customer_id
    HAVING SUM(amount) > 100
),
customer_details AS (
    SELECT 
        customer_id,
        name,
        email,
        tier
    FROM customers
    WHERE active = true
)
SELECT 
    cd.name,
    cd.email,
    cd.tier,
    rs.total_amount,
    rs.transaction_count,
    rs.avg_transaction,
    CASE 
        WHEN rs.total_amount > 1000 THEN 'High Value'
        WHEN rs.total_amount > 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment
FROM recent_sales rs
JOIN customer_details cd ON rs.customer_id = cd.customer_id
ORDER BY rs.total_amount DESC
LIMIT 100
                    """,
                    }
                ),
                "metadata": {
                    "complexity": "high",
                    "tables": ["sales_transactions", "customers"],
                },
            },
            "documentation": {
                "content": """
# Sales Data Analysis Guide

## Overview
The sales transactions table contains detailed information about all customer purchases. This data is crucial for understanding customer behavior, revenue trends, and business performance.

## Key Fields
- transaction_id: Unique identifier for each sale
- customer_id: Links to customer information
- amount: The base transaction amount before tax and discounts
- total_amount: Final amount after all adjustments

## Common Queries
When analyzing sales data, consider these common patterns:

### Revenue Analysis
- Daily, weekly, monthly revenue trends
- Revenue by product category
- Revenue by customer segment

### Customer Behavior
- Purchase frequency analysis
- Customer lifetime value calculations
- Repeat purchase patterns

## Data Quality Notes
- All amounts are in USD unless otherwise specified
- Status field indicates transaction completion
- Timestamps are in UTC
                """,
                "metadata": {"topic": "sales_analysis", "author": "data_team"},
            },
        }

    def test_chunking_strategies(self) -> dict[str, Any]:
        """Test all chunking strategies."""
        logger.info("🧪 Testing chunking strategies...")
        results = {}

        for doc_type, doc_data in self.test_documents.items():
            try:
                # Test chunking
                chunks = chunk_document(
                    doc_type=doc_type,
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                )

                # Validate chunks
                chunk_info = []
                total_tokens = 0

                for i, chunk in enumerate(chunks):
                    chunk_tokens = chunk.estimate_tokens()
                    total_tokens += chunk_tokens

                    chunk_info.append(
                        {
                            "index": i,
                            "type": chunk.chunk_type,
                            "tokens": chunk_tokens,
                            "content_length": len(chunk.content),
                            "metadata_keys": list(chunk.metadata.keys()),
                        }
                    )

                results[doc_type] = {
                    "success": True,
                    "chunk_count": len(chunks),
                    "total_tokens": total_tokens,
                    "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
                    "chunks": chunk_info,
                }

                logger.info(
                    f"  ✓ {doc_type}: {len(chunks)} chunks, {total_tokens} tokens"
                )

            except Exception as e:
                results[doc_type] = {"success": False, "error": str(e)}
                logger.error(f"  ✗ {doc_type}: {e}")

        self.test_results["chunking_tests"] = results
        return results

    def test_database_operations(self) -> dict[str, Any]:
        """Test database operations with chunked architecture."""
        logger.info("🗄️ Testing database operations...")

        # Get database config
        db_config = DATABASE_ROUTING_CONFIG.get("postgresql", {})
        if not db_config:
            return {"error": "No PostgreSQL configuration found"}

        # Build connection string
        from src.connectors.database import DatabaseConnectionManager

        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            db_config
        )

        results = {}
        rag = None

        try:
            # Initialize RAG system
            rag = PgvectorRAG(connection_string=connection_string, enable_chunking=True)

            # Test connection
            if not rag.connect_to_database():
                return {"error": "Failed to connect to database"}

            # Test embeddings
            if not rag.initialize_embeddings():
                return {"error": "Failed to initialize embeddings"}

            results["connection"] = {"success": True}

            # Test document addition
            doc_results = {}
            for doc_type, doc_data in self.test_documents.items():
                try:
                    start_time = time.time()
                    add_result = rag.add_document(
                        content=doc_data["content"],
                        doc_type=doc_type,
                        source="test",
                        metadata=doc_data["metadata"],
                    )
                    end_time = time.time()

                    if add_result.get("success"):
                        doc_results[doc_type] = {
                            "success": True,
                            "document_id": add_result["document_id"],
                            "chunks_created": add_result["chunks_created"],
                            "embeddings_created": add_result["embeddings_created"],
                            "processing_time": end_time - start_time,
                        }
                        logger.info(
                            f"  ✓ Added {doc_type}: {add_result['chunks_created']} chunks"
                        )
                    else:
                        doc_results[doc_type] = {
                            "success": False,
                            "error": add_result.get("error", "Unknown error"),
                        }
                        logger.error(f"  ✗ Failed to add {doc_type}")

                except Exception as e:
                    doc_results[doc_type] = {"success": False, "error": str(e)}
                    logger.error(f"  ✗ Error adding {doc_type}: {e}")

            results["document_addition"] = doc_results

            # Test statistics
            try:
                stats = rag.get_statistics()
                results["statistics"] = {"success": True, "stats": stats}
                logger.info(f"  ✓ Statistics: {stats.get('totals', {})}")
            except Exception as e:
                results["statistics"] = {"success": False, "error": str(e)}

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Database operations failed: {e}")

        finally:
            if rag:
                rag.close()

        self.test_results["database_tests"] = results
        return results

    def test_search_functionality(self) -> dict[str, Any]:
        """Test search functionality."""
        logger.info("🔍 Testing search functionality...")

        # Get database config
        db_config = DATABASE_ROUTING_CONFIG.get("postgresql", {})
        if not db_config:
            return {"error": "No PostgreSQL configuration found"}

        from src.connectors.database import DatabaseConnectionManager

        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            db_config
        )

        results = {}
        rag = None

        try:
            rag = PgvectorRAG(connection_string=connection_string, enable_chunking=True)

            if not rag.connect_to_database() or not rag.initialize_embeddings():
                return {"error": "Failed to initialize RAG system"}

            # Test queries
            test_queries = [
                {
                    "query": "Show me customer sales data",
                    "expected_types": ["schema", "sql_example"],
                    "description": "General sales query",
                },
                {
                    "query": "What columns are in the sales table?",
                    "expected_types": ["schema"],
                    "description": "Schema-specific query",
                },
                {
                    "query": "How to calculate customer lifetime value?",
                    "expected_types": ["sql_example", "documentation"],
                    "description": "Analysis methodology query",
                },
            ]

            query_results = {}

            for i, test_query in enumerate(test_queries):
                try:
                    # Similarity search
                    start_time = time.time()
                    sim_results = rag.similarity_search(
                        query=test_query["query"],
                        k=5,
                        similarity_threshold=0.1,  # Lower threshold for testing
                    )
                    sim_time = time.time() - start_time

                    # Hybrid search
                    start_time = time.time()
                    hybrid_results = rag.hybrid_search(query=test_query["query"], k=5)
                    hybrid_time = time.time() - start_time

                    query_results[f"query_{i + 1}"] = {
                        "query": test_query["query"],
                        "description": test_query["description"],
                        "similarity_search": {
                            "results_count": len(sim_results),
                            "processing_time": sim_time,
                            "top_similarity": (
                                sim_results[0]["similarity"] if sim_results else 0
                            ),
                            "doc_types_found": list(
                                set(r["doc_type"] for r in sim_results)
                            ),
                        },
                        "hybrid_search": {
                            "results_count": len(hybrid_results),
                            "processing_time": hybrid_time,
                            "top_score": (
                                hybrid_results[0]["combined_score"]
                                if hybrid_results
                                else 0
                            ),
                        },
                    }

                    logger.info(
                        f"  ✓ Query {i + 1}: {len(sim_results)} sim, {len(hybrid_results)} hybrid results"
                    )

                except Exception as e:
                    query_results[f"query_{i + 1}"] = {
                        "query": test_query["query"],
                        "error": str(e),
                    }
                    logger.error(f"  ✗ Query {i + 1} failed: {e}")

            results["queries"] = query_results

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Search testing failed: {e}")

        finally:
            if rag:
                rag.close()

        self.test_results["search_tests"] = results
        return results

    def benchmark_performance(self) -> dict[str, Any]:
        """Benchmark the performance of the chunked architecture."""
        logger.info("⚡ Running performance benchmarks...")

        # Get database config
        db_config = DATABASE_ROUTING_CONFIG.get("postgresql", {})
        if not db_config:
            return {"error": "No PostgreSQL configuration found"}

        from src.connectors.database import DatabaseConnectionManager

        connection_string = DatabaseConnectionManager.build_postgres_connection_string(
            db_config
        )

        results = {}

        try:
            # Test with different chunking settings
            chunking_configs = [
                {"enable_chunking": False, "name": "no_chunking"},
                {"enable_chunking": True, "name": "default_chunking"},
                {
                    "enable_chunking": True,
                    "name": "optimized_chunking",
                    "chunk_strategies": {
                        "schema": {"max_columns_per_chunk": 8},
                        "sql_example": {"max_sql_length": 500},
                    },
                },
            ]

            config_results = {}

            for config in chunking_configs:
                logger.info(f"  Testing {config['name']}...")

                rag = PgvectorRAG(
                    connection_string=connection_string,
                    **{k: v for k, v in config.items() if k not in ["name"]},
                )

                try:
                    if not rag.connect_to_database() or not rag.initialize_embeddings():
                        continue

                    # Benchmark document addition
                    add_times = []
                    chunk_counts = []

                    for doc_type, doc_data in self.test_documents.items():
                        start_time = time.time()
                        result = rag.add_document(
                            content=doc_data["content"],
                            doc_type=doc_type,
                            source="benchmark",
                            metadata=doc_data["metadata"],
                        )
                        end_time = time.time()

                        if result.get("success"):
                            add_times.append(end_time - start_time)
                            chunk_counts.append(result["chunks_created"])

                    # Benchmark search
                    search_times = []
                    for _ in range(3):  # Multiple runs for average
                        start_time = time.time()
                        rag.similarity_search("customer sales analysis", k=5)
                        search_times.append(time.time() - start_time)

                    config_results[config["name"]] = {
                        "avg_add_time": (
                            sum(add_times) / len(add_times) if add_times else 0
                        ),
                        "total_chunks": sum(chunk_counts),
                        "avg_search_time": (
                            sum(search_times) / len(search_times) if search_times else 0
                        ),
                        "documents_processed": len(add_times),
                    }

                except Exception as e:
                    config_results[config["name"]] = {"error": str(e)}

                finally:
                    rag.close()

            results["chunking_comparison"] = config_results

            # Generate performance summary
            if len(config_results) >= 2:
                no_chunk = config_results.get("no_chunking", {})
                default_chunk = config_results.get("default_chunking", {})

                if "error" not in no_chunk and "error" not in default_chunk:
                    results["performance_impact"] = {
                        "chunking_overhead_percent": (
                            (
                                (
                                    default_chunk["avg_add_time"]
                                    - no_chunk["avg_add_time"]
                                )
                                / no_chunk["avg_add_time"]
                                * 100
                            )
                            if no_chunk["avg_add_time"] > 0
                            else 0
                        ),
                        "search_performance_change": (
                            (
                                (
                                    default_chunk["avg_search_time"]
                                    - no_chunk["avg_search_time"]
                                )
                                / no_chunk["avg_search_time"]
                                * 100
                            )
                            if no_chunk["avg_search_time"] > 0
                            else 0
                        ),
                        "chunk_granularity_factor": (
                            (default_chunk["total_chunks"] / no_chunk["total_chunks"])
                            if no_chunk["total_chunks"] > 0
                            else 1
                        ),
                    }

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Benchmarking failed: {e}")

        self.test_results["benchmark_results"] = results
        return results

    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("CHUNKED PGVECTOR ARCHITECTURE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Chunking tests
        if self.test_results["chunking_tests"]:
            report.append("CHUNKING STRATEGY TESTS")
            report.append("-" * 40)
            for doc_type, result in self.test_results["chunking_tests"].items():
                if result.get("success"):
                    report.append(
                        f"✓ {doc_type.upper()}: {result['chunk_count']} chunks, {result['total_tokens']} tokens"
                    )
                else:
                    report.append(
                        f"✗ {doc_type.upper()}: {result.get('error', 'Failed')}"
                    )
            report.append("")

        # Database tests
        if self.test_results["database_tests"]:
            report.append("DATABASE OPERATION TESTS")
            report.append("-" * 40)
            db_tests = self.test_results["database_tests"]

            if db_tests.get("connection", {}).get("success"):
                report.append("✓ Database Connection: Success")
            else:
                report.append("✗ Database Connection: Failed")

            if "document_addition" in db_tests:
                for doc_type, result in db_tests["document_addition"].items():
                    if result.get("success"):
                        report.append(
                            f"✓ {doc_type.upper()} Addition: {result['chunks_created']} chunks in {result['processing_time']:.3f}s"
                        )
                    else:
                        report.append(
                            f"✗ {doc_type.upper()} Addition: {result.get('error', 'Failed')}"
                        )
            report.append("")

        # Search tests
        if self.test_results["search_tests"]:
            report.append("SEARCH FUNCTIONALITY TESTS")
            report.append("-" * 40)
            search_tests = self.test_results["search_tests"]

            if "queries" in search_tests:
                for query_id, result in search_tests["queries"].items():
                    if "error" not in result:
                        sim_count = result["similarity_search"]["results_count"]
                        hybrid_count = result["hybrid_search"]["results_count"]
                        report.append(
                            f"✓ {query_id.upper()}: {sim_count} similarity, {hybrid_count} hybrid results"
                        )
                    else:
                        report.append(f"✗ {query_id.upper()}: {result['error']}")
            report.append("")

        # Benchmark results
        if self.test_results["benchmark_results"]:
            report.append("PERFORMANCE BENCHMARKS")
            report.append("-" * 40)
            benchmark = self.test_results["benchmark_results"]

            if "chunking_comparison" in benchmark:
                for config_name, result in benchmark["chunking_comparison"].items():
                    if "error" not in result:
                        report.append(f"{config_name.upper()}:")
                        report.append(
                            f"  - Avg Add Time: {result['avg_add_time']:.3f}s"
                        )
                        report.append(f"  - Total Chunks: {result['total_chunks']}")
                        report.append(
                            f"  - Avg Search Time: {result['avg_search_time']:.3f}s"
                        )
                    else:
                        report.append(
                            f"{config_name.upper()}: Error - {result['error']}"
                        )

            if "performance_impact" in benchmark:
                impact = benchmark["performance_impact"]
                report.append("CHUNKING IMPACT:")
                report.append(
                    f"  - Processing Overhead: {impact['chunking_overhead_percent']:.1f}%"
                )
                report.append(
                    f"  - Search Performance Change: {impact['search_performance_change']:.1f}%"
                )
                report.append(
                    f"  - Granularity Factor: {impact['chunk_granularity_factor']:.1f}x"
                )
            report.append("")

        report.append("=" * 80)

        return "\\n".join(report)

    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            filename = f"chunked_architecture_test_results_{int(time.time())}.json"

        filepath = Path(__file__).parent.parent / "test_results" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        logger.info(f"Test results saved to: {filepath}")
        return filepath


def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description="Test chunked pgvector architecture")
    parser.add_argument(
        "--test-chunking", action="store_true", help="Test chunking strategies"
    )
    parser.add_argument(
        "--test-database", action="store_true", help="Test database operations"
    )
    parser.add_argument(
        "--test-search", action="store_true", help="Test search functionality"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to file"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate and display report"
    )

    args = parser.parse_args()

    if not any(
        [
            args.test_chunking,
            args.test_database,
            args.test_search,
            args.benchmark,
            args.all,
        ]
    ):
        parser.print_help()
        sys.exit(1)

    tester = ChunkedArchitectureTester()

    try:
        logger.info("🚀 Starting chunked architecture tests...")

        if args.all or args.test_chunking:
            tester.test_chunking_strategies()

        if args.all or args.test_database:
            tester.test_database_operations()

        if args.all or args.test_search:
            tester.test_search_functionality()

        if args.all or args.benchmark:
            tester.benchmark_performance()

        if args.save_results:
            tester.save_results()

        if args.report or args.all:
            report = tester.generate_report()
            print(report)

        logger.info("✅ All tests completed!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
