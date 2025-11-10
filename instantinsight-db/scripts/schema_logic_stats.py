#!/usr/bin/env python3
"""
Generate statistics for SchemaLogic system and save to CSV.

Analyzes documents, schemas, tables, and other metrics.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.config.database_config import ATHENA_CONFIG, POSTGRES_CONFIG
from src.connectors.database import DatabaseConnectionManager


class AthenaRAGStats:
    """Generate comprehensive statistics for Athena RAG system."""

    def __init__(self):
        """Initialize the stats generator."""
        self.engine = DatabaseConnectionManager.create_postgres_engine(POSTGRES_CONFIG)
        # Don't initialize PgvectorRAG as it auto-clears data when < 5 documents
        # self.rag = PgvectorRAG(DATABASE_URL)
        self.stats = defaultdict(dict)

    def collect_document_stats(self):
        """Collect statistics about documents in the RAG system."""
        logger.info("Collecting document statistics...")

        # Query to get document counts by type - updated for chunked schema
        query = """
        SELECT 
            doc_type,
            COUNT(*) as count,
            AVG(LENGTH(full_content)) as avg_content_length,
            MIN(LENGTH(full_content)) as min_content_length,
            MAX(LENGTH(full_content)) as max_content_length
        FROM rag_documents
        GROUP BY doc_type
        ORDER BY count DESC;
        """

        df = pd.read_sql(query, self.engine)

        # Store basic stats
        self.stats["document_types"] = df.to_dict("records")
        self.stats["total_documents"] = df["count"].sum()

        # Get database/catalog statistics
        catalog_query = """
        SELECT 
            metadata->>'catalog' as catalog,
            metadata->>'database' as database,
            COUNT(*) as document_count,
            COUNT(DISTINCT metadata->>'table_name') as table_count
        FROM rag_documents
        WHERE metadata IS NOT NULL AND metadata->>'catalog' IS NOT NULL
        GROUP BY metadata->>'catalog', metadata->>'database';
        """

        catalog_df = pd.read_sql(catalog_query, self.engine)
        self.stats["catalog_stats"] = catalog_df.to_dict("records")

    def collect_table_stats(self):
        """Collect statistics about tables in the system."""
        logger.info("Collecting table statistics...")

        # First, check what doc_type values actually exist
        check_types_query = "SELECT DISTINCT doc_type FROM rag_documents"
        types_df = pd.read_sql(check_types_query, self.engine)
        logger.info(f"Available doc_types: {types_df['doc_type'].tolist()}")

        # Count tables based on schema documents (the correct way)
        # Schema documents represent individual tables
        schema_count_query = """
        SELECT COUNT(*) as total_schema_tables
        FROM rag_documents
        WHERE doc_type = 'schema'
        """

        schema_count_df = pd.read_sql(schema_count_query, self.engine)
        total_tables = schema_count_df["total_schema_tables"].iloc[0]
        logger.info(f"Found {total_tables} schema documents (tables)")

        # Also check what metadata fields exist for schema documents
        metadata_sample_query = """
        SELECT metadata
        FROM rag_documents
        WHERE doc_type = 'schema'
        LIMIT 3
        """
        metadata_sample = pd.read_sql(metadata_sample_query, self.engine)
        if not metadata_sample.empty:
            logger.info(
                f"Sample schema metadata keys: {list(metadata_sample.iloc[0]['metadata'].keys()) if metadata_sample.iloc[0]['metadata'] else 'None'}"
            )

        # Try to count unique tables by full_table_name if available
        unique_table_query = """
        SELECT COUNT(DISTINCT metadata->>'full_table_name') as unique_tables
        FROM rag_documents
        WHERE doc_type = 'schema'
        AND metadata IS NOT NULL 
        AND metadata->>'full_table_name' IS NOT NULL
        """
        unique_df = pd.read_sql(unique_table_query, self.engine)
        unique_tables = unique_df["unique_tables"].iloc[0]
        if unique_tables > 0:
            logger.info(f"Found {unique_tables} unique tables by full_table_name")
            total_tables = unique_tables  # Use unique count if available

        # Analyze ALL schema content to identify unique tables - updated for chunked schema
        all_content_query = """
        SELECT full_content as content
        FROM rag_documents
        WHERE doc_type = 'schema'
        AND full_content IS NOT NULL
        """

        all_content_df = pd.read_sql(all_content_query, self.engine)

        # Extract table names from DDL content to get unique tables
        unique_tables = set()
        ddl_count = 0
        documentation_count = 0

        import re

        for _, row in all_content_df.iterrows():
            content = row["content"]
            if content:
                if "CREATE EXTERNAL TABLE" in content:
                    ddl_count += 1
                    # Extract table name from DDL
                    lines = content.split("\n")
                    for line in lines:
                        if "CREATE EXTERNAL TABLE" in line:
                            # Extract table name (format: CREATE EXTERNAL TABLE `database`.`table`)
                            match = re.search(r"`([^`]+)`\.`([^`]+)`", line)
                            if match:
                                database, table = match.groups()
                                unique_tables.add(f"{database}.{table}")
                            break
                elif "awsdatacatalog." in content and "Table:" in content:
                    # This looks like table documentation
                    documentation_count += 1
                    # Extract from documentation format: "Athena Table: awsdatacatalog.database.table"
                    match = re.search(r"awsdatacatalog\.([^.]+)\.([^\s]+)", content)
                    if match:
                        database, table = match.groups()
                        unique_tables.add(f"{database}.{table}")

        actual_unique_tables = len(unique_tables)
        logger.info(
            f"Processed {ddl_count} DDL documents and {documentation_count} documentation documents"
        )
        logger.info(f"Unique tables extracted from content: {actual_unique_tables}")
        if actual_unique_tables > 0:
            total_tables = actual_unique_tables
            logger.info(f"Using unique table count: {total_tables}")

        # Get sample content for debugging (limit to 3 for brevity) - updated for chunked schema
        sample_query = """
        SELECT 
            full_content as content,
            metadata,
            created_at,
            LENGTH(full_content) as content_length
        FROM rag_documents
        WHERE doc_type = 'schema'
        ORDER BY created_at DESC
        LIMIT 3;
        """
        content_df = pd.read_sql(sample_query, self.engine)

        # Sample the content to show what we're working with
        self.stats["schema_content_samples"] = []
        for _i, row in content_df.head(3).iterrows():
            sample = {
                "content_preview": (
                    row["content"][:200] + "..." if row["content"] else "No content"
                ),
                "metadata": row["metadata"],
                "content_length": row["content_length"],
            }
            self.stats["schema_content_samples"].append(sample)

        # Store basic stats about schema documents
        self.stats["schema_analysis"] = {
            "total_schema_docs": len(content_df),
            "avg_content_length": (
                content_df["content_length"].mean() if not content_df.empty else 0
            ),
            "ddl_docs": sum(
                1
                for _, row in content_df.iterrows()
                if row["content"] and "CREATE EXTERNAL TABLE" in row["content"]
            ),
            "extracted_unique_tables": actual_unique_tables,
        }

        self.stats["total_tables"] = total_tables  # Use the correct count
        self.stats[
            "table_stats"
        ] = []  # Simplified since metadata doesn't contain table details
        self.stats["top_10_tables"] = []  # Not applicable without proper metadata

    def collect_sql_example_stats(self):
        """Collect statistics about SQL examples."""
        logger.info("Collecting SQL example statistics...")

        sql_query = """
        SELECT 
            metadata->>'question' as question,
            full_content as sql_query,
            LENGTH(full_content) as content_length,
            created_at as added_at
        FROM rag_documents
        WHERE doc_type = 'sql_example'
        ORDER BY created_at DESC;
        """

        df = pd.read_sql(sql_query, self.engine)
        self.stats["sql_examples_count"] = len(df)
        self.stats["avg_sql_length"] = (
            df["content_length"].mean() if not df.empty else 0
        )

        # Get complexity metrics (rough estimate based on keywords)
        if not df.empty:
            complexity_keywords = [
                "JOIN",
                "GROUP BY",
                "HAVING",
                "UNION",
                "SUBQUERY",
                "WITH",
            ]
            df["complexity_score"] = df["sql_query"].apply(
                lambda x: (
                    sum(1 for kw in complexity_keywords if kw in str(x).upper())
                    if x
                    else 0
                )
            )
            self.stats["avg_sql_complexity"] = df["complexity_score"].mean()
            self.stats["complex_queries_count"] = len(df[df["complexity_score"] >= 2])

    def collect_chunk_stats(self):
        """Collect detailed statistics about chunks."""
        logger.info("Collecting chunk statistics...")

        # Chunk distribution by type and document type
        chunk_distribution_query = """
        SELECT 
            d.doc_type,
            c.chunk_type,
            COUNT(c.id) as chunk_count,
            AVG(LENGTH(c.content)) as avg_content_length,
            MIN(LENGTH(c.content)) as min_content_length,
            MAX(LENGTH(c.content)) as max_content_length,
            COUNT(e.id) as embeddings_count
        FROM rag_documents d
        JOIN rag_chunks c ON d.id = c.document_id
        LEFT JOIN rag_embeddings e ON c.id = e.chunk_id
        GROUP BY d.doc_type, c.chunk_type
        ORDER BY d.doc_type, chunk_count DESC;
        """

        chunk_df = pd.read_sql(chunk_distribution_query, self.engine)
        self.stats["chunk_distribution"] = chunk_df.to_dict("records")

        # Chunks per document statistics
        chunks_per_doc_query = """
        SELECT 
            d.doc_type,
            COUNT(c.id) as chunks_per_document,
            COUNT(DISTINCT d.id) as document_count
        FROM rag_documents d
        JOIN rag_chunks c ON d.id = c.document_id
        GROUP BY d.doc_type, d.id
        ORDER BY chunks_per_document DESC;
        """

        chunks_per_doc_df = pd.read_sql(chunks_per_doc_query, self.engine)

        # Calculate statistics per document type
        chunking_efficiency = {}
        for doc_type in chunks_per_doc_df["doc_type"].unique():
            type_data = chunks_per_doc_df[chunks_per_doc_df["doc_type"] == doc_type]
            chunking_efficiency[doc_type] = {
                "avg_chunks_per_doc": type_data["chunks_per_document"].mean(),
                "min_chunks_per_doc": type_data["chunks_per_document"].min(),
                "max_chunks_per_doc": type_data["chunks_per_document"].max(),
                "total_documents": len(type_data),
                "total_chunks": type_data["chunks_per_document"].sum(),
            }

        self.stats["chunking_efficiency"] = chunking_efficiency

        # Namespace distribution
        namespace_query = """
        SELECT 
            e.namespace,
            COUNT(e.id) as embedding_count,
            COUNT(DISTINCT c.document_id) as document_count
        FROM rag_embeddings e
        JOIN rag_chunks c ON e.chunk_id = c.id
        GROUP BY e.namespace
        ORDER BY embedding_count DESC;
        """

        namespace_df = pd.read_sql(namespace_query, self.engine)
        self.stats["namespace_distribution"] = namespace_df.to_dict("records")

    def collect_embedding_stats(self):
        """Collect statistics about embeddings."""
        logger.info("Collecting embedding statistics...")

        # Enhanced embedding statistics with chunk details
        embedding_query = """
        SELECT 
            COUNT(DISTINCT c.id) as total_chunks,
            COUNT(e.id) as total_embeddings,
            COUNT(e.id) as non_null_embeddings,
            1024 as avg_embedding_dimension,
            COUNT(DISTINCT c.document_id) as documents_with_chunks,
            COUNT(DISTINCT e.namespace) as unique_namespaces
        FROM rag_chunks c
        LEFT JOIN rag_embeddings e ON c.id = e.chunk_id;
        """

        result = pd.read_sql(embedding_query, self.engine)
        self.stats["embedding_stats"] = result.to_dict("records")[0]

        # Embedding coverage by chunk type
        coverage_query = """
        SELECT 
            c.chunk_type,
            COUNT(c.id) as total_chunks_of_type,
            COUNT(e.id) as chunks_with_embeddings,
            ROUND(COUNT(e.id)::decimal / COUNT(c.id) * 100, 2) as coverage_percentage
        FROM rag_chunks c
        LEFT JOIN rag_embeddings e ON c.id = e.chunk_id
        GROUP BY c.chunk_type
        ORDER BY coverage_percentage DESC;
        """

        coverage_df = pd.read_sql(coverage_query, self.engine)
        self.stats["embedding_coverage_by_type"] = coverage_df.to_dict("records")

    def collect_metadata_stats(self):
        """Collect statistics about metadata fields."""
        logger.info("Collecting metadata statistics...")

        # Get all unique metadata keys
        metadata_query = """
        SELECT 
            jsonb_object_keys(metadata) as metadata_key,
            COUNT(*) as usage_count
        FROM rag_documents
        WHERE metadata IS NOT NULL
        GROUP BY metadata_key
        ORDER BY usage_count DESC;
        """

        df = pd.read_sql(metadata_query, self.engine)
        self.stats["metadata_fields"] = df.to_dict("records")

    def collect_performance_analysis(self):
        """Analyze chunking performance and efficiency."""
        logger.info("Collecting performance analysis...")

        # Chunking effectiveness analysis
        effectiveness_query = """
        WITH document_chunk_counts AS (
            SELECT 
                d.id,
                d.doc_type,
                d.full_content,
                COUNT(c.id) as chunks_per_doc
            FROM rag_documents d
            JOIN rag_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.doc_type, d.full_content
        )
        SELECT 
            doc_type,
            AVG(LENGTH(full_content)) as avg_document_length,
            AVG(chunks_per_doc) as avg_chunks_per_document,
            SUM(chunks_per_doc) as total_chunks,
            COUNT(DISTINCT id) as total_documents
        FROM document_chunk_counts
        GROUP BY doc_type;
        """

        effectiveness_df = pd.read_sql(effectiveness_query, self.engine)
        self.stats["chunking_effectiveness"] = effectiveness_df.to_dict("records")

        # Separate query for chunk length statistics
        chunk_length_query = """
        SELECT 
            d.doc_type,
            AVG(LENGTH(c.content)) as avg_chunk_length,
            MIN(LENGTH(c.content)) as min_chunk_length,
            MAX(LENGTH(c.content)) as max_chunk_length
        FROM rag_documents d
        JOIN rag_chunks c ON d.id = c.document_id
        GROUP BY d.doc_type;
        """

        chunk_length_df = pd.read_sql(chunk_length_query, self.engine)
        self.stats["chunk_length_analysis"] = chunk_length_df.to_dict("records")

        # Search optimization metrics
        search_optimization_query = """
        SELECT 
            c.chunk_type,
            COUNT(c.id) as chunk_count,
            AVG(LENGTH(c.content)) as avg_content_length,
            CASE 
                WHEN c.chunk_type IN ('table_overview', 'example_overview') THEN 'High Priority'
                WHEN c.chunk_type IN ('columns', 'example_main') THEN 'Medium Priority' 
                ELSE 'Low Priority'
            END as search_priority
        FROM rag_chunks c
        GROUP BY c.chunk_type
        ORDER BY 
            CASE 
                WHEN c.chunk_type IN ('table_overview', 'example_overview') THEN 1
                WHEN c.chunk_type IN ('columns', 'example_main') THEN 2 
                ELSE 3
            END, chunk_count DESC;
        """

        search_df = pd.read_sql(search_optimization_query, self.engine)
        self.stats["search_optimization"] = search_df.to_dict("records")

        # Storage efficiency
        storage_query = """
        SELECT 
            SUM(LENGTH(d.full_content)) as total_document_content_size,
            SUM(LENGTH(c.content)) as total_chunk_content_size,
            COUNT(DISTINCT d.id) as unique_documents,
            COUNT(c.id) as total_chunks,
            COUNT(e.id) as total_embeddings,
            ROUND(SUM(LENGTH(c.content))::decimal / SUM(LENGTH(d.full_content)) * 100, 2) as storage_efficiency_percentage
        FROM rag_documents d
        JOIN rag_chunks c ON d.id = c.document_id
        LEFT JOIN rag_embeddings e ON c.id = e.chunk_id;
        """

        storage_df = pd.read_sql(storage_query, self.engine)
        self.stats["storage_efficiency"] = storage_df.to_dict("records")[0]

    def generate_summary_stats(self):
        """Generate enhanced summary statistics."""
        logger.info("Generating enhanced summary statistics...")

        # Basic statistics
        embedding_stats = self.stats.get("embedding_stats", {})
        chunking_efficiency = self.stats.get("chunking_efficiency", {})
        storage_efficiency = self.stats.get("storage_efficiency", {})

        # Calculate average chunks per document across all types
        total_chunks = sum(
            [eff.get("total_chunks", 0) for eff in chunking_efficiency.values()]
        )
        total_docs = sum(
            [eff.get("total_documents", 0) for eff in chunking_efficiency.values()]
        )
        avg_chunks_per_doc = round(total_chunks / max(total_docs, 1), 2)

        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "athena_database": ATHENA_CONFIG.get("database", "Unknown"),
            # Document Statistics
            "total_documents": self.stats.get("total_documents", 0),
            "total_tables": self.stats.get("total_tables", 0),
            "sql_examples_count": self.stats.get("sql_examples_count", 0),
            "document_types_count": len(self.stats.get("document_types", [])),
            "avg_sql_complexity": round(self.stats.get("avg_sql_complexity", 0), 2),
            # Chunk Statistics
            "total_chunks": embedding_stats.get("total_chunks", 0),
            "avg_chunks_per_document": avg_chunks_per_doc,
            "unique_namespaces": embedding_stats.get("unique_namespaces", 0),
            "documents_with_chunks": embedding_stats.get("documents_with_chunks", 0),
            # Embedding Statistics
            "total_embeddings": embedding_stats.get("total_embeddings", 0),
            "embedding_coverage": round(
                embedding_stats.get("non_null_embeddings", 0)
                / max(embedding_stats.get("total_embeddings", 1), 1)
                * 100,
                2,
            ),
            "embedding_dimension": embedding_stats.get("avg_embedding_dimension", 1024),
            # Performance Metrics
            "storage_efficiency_percentage": storage_efficiency.get(
                "storage_efficiency_percentage", 0
            ),
            "chunk_types_available": len(self.stats.get("chunk_distribution", [])),
            # Chunking Efficiency by Document Type
            "chunking_by_type": {
                doc_type: {
                    "avg_chunks": round(eff.get("avg_chunks_per_doc", 0), 2),
                    "documents": eff.get("total_documents", 0),
                    "total_chunks": eff.get("total_chunks", 0),
                }
                for doc_type, eff in chunking_efficiency.items()
            },
        }

        self.stats["summary"] = summary

    def save_to_csv(self, output_dir="./rag_stats"):
        """Save all statistics to CSV files."""
        logger.info(f"Saving statistics to CSV files in {output_dir}...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save summary stats
        summary_df = pd.DataFrame([self.stats["summary"]])
        summary_df.to_csv(output_path / "athena_rag_summary.csv", index=False)
        logger.info("✓ Saved summary stats to athena_rag_summary.csv")

        # Save document type stats
        if self.stats.get("document_types"):
            doc_types_df = pd.DataFrame(self.stats["document_types"])
            doc_types_df.to_csv(output_path / "document_types_stats.csv", index=False)
            logger.info("✓ Saved document type stats to document_types_stats.csv")

        # Save table stats
        if self.stats.get("table_stats"):
            table_df = pd.DataFrame(self.stats["table_stats"])
            table_df.to_csv(output_path / "table_stats.csv", index=False)
            logger.info("✓ Saved table stats to table_stats.csv")

        # Save top tables
        if self.stats.get("top_10_tables"):
            top_tables_df = pd.DataFrame(self.stats["top_10_tables"])
            top_tables_df.to_csv(output_path / "top_10_tables.csv", index=False)
            logger.info("✓ Saved top 10 tables to top_10_tables.csv")

        # Save catalog stats
        if self.stats.get("catalog_stats"):
            catalog_df = pd.DataFrame(self.stats["catalog_stats"])
            catalog_df.to_csv(output_path / "catalog_stats.csv", index=False)
            logger.info("✓ Saved catalog stats to catalog_stats.csv")

        # Save metadata field stats
        if self.stats.get("metadata_fields"):
            metadata_df = pd.DataFrame(self.stats["metadata_fields"])
            metadata_df.to_csv(output_path / "metadata_fields_stats.csv", index=False)
            logger.info("✓ Saved metadata field stats to metadata_fields_stats.csv")

        # Save chunk distribution stats
        if self.stats.get("chunk_distribution"):
            chunk_dist_df = pd.DataFrame(self.stats["chunk_distribution"])
            chunk_dist_df.to_csv(
                output_path / "chunk_distribution_stats.csv", index=False
            )
            logger.info(
                "✓ Saved chunk distribution stats to chunk_distribution_stats.csv"
            )

        # Save chunking efficiency stats
        if self.stats.get("chunking_efficiency"):
            chunking_eff_data = []
            for doc_type, efficiency in self.stats["chunking_efficiency"].items():
                efficiency["doc_type"] = doc_type
                chunking_eff_data.append(efficiency)
            chunking_eff_df = pd.DataFrame(chunking_eff_data)
            chunking_eff_df.to_csv(
                output_path / "chunking_efficiency_stats.csv", index=False
            )
            logger.info(
                "✓ Saved chunking efficiency stats to chunking_efficiency_stats.csv"
            )

        # Save namespace distribution stats
        if self.stats.get("namespace_distribution"):
            namespace_df = pd.DataFrame(self.stats["namespace_distribution"])
            namespace_df.to_csv(
                output_path / "namespace_distribution_stats.csv", index=False
            )
            logger.info(
                "✓ Saved namespace distribution stats to namespace_distribution_stats.csv"
            )

        # Save embedding coverage by type
        if self.stats.get("embedding_coverage_by_type"):
            coverage_df = pd.DataFrame(self.stats["embedding_coverage_by_type"])
            coverage_df.to_csv(
                output_path / "embedding_coverage_by_type_stats.csv", index=False
            )
            logger.info(
                "✓ Saved embedding coverage by type to embedding_coverage_by_type_stats.csv"
            )

        # Save search optimization stats
        if self.stats.get("search_optimization"):
            search_opt_df = pd.DataFrame(self.stats["search_optimization"])
            search_opt_df.to_csv(
                output_path / "search_optimization_stats.csv", index=False
            )
            logger.info(
                "✓ Saved search optimization stats to search_optimization_stats.csv"
            )

        # Save chunking effectiveness stats
        if self.stats.get("chunking_effectiveness"):
            chunking_eff_df = pd.DataFrame(self.stats["chunking_effectiveness"])
            chunking_eff_df.to_csv(
                output_path / "chunking_effectiveness_stats.csv", index=False
            )
            logger.info(
                "✓ Saved chunking effectiveness stats to chunking_effectiveness_stats.csv"
            )

        # Save chunk length analysis stats
        if self.stats.get("chunk_length_analysis"):
            chunk_length_df = pd.DataFrame(self.stats["chunk_length_analysis"])
            chunk_length_df.to_csv(
                output_path / "chunk_length_analysis_stats.csv", index=False
            )
            logger.info(
                "✓ Saved chunk length analysis stats to chunk_length_analysis_stats.csv"
            )

        # Save all stats as JSON for reference
        with open(output_path / "all_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info("✓ Saved complete stats to all_stats.json")

        return output_path

    def run(self):
        """Run all statistics collection and save results."""
        logger.info("🚀 Starting Athena RAG statistics collection...")

        try:
            # Collect all statistics
            self.collect_document_stats()
            self.collect_table_stats()
            self.collect_sql_example_stats()
            self.collect_chunk_stats()
            self.collect_embedding_stats()
            self.collect_metadata_stats()
            self.collect_performance_analysis()
            self.generate_summary_stats()

            # Save to CSV
            output_path = self.save_to_csv()

            logger.info("✅ Statistics collection completed successfully!")
            logger.info(f"📊 Results saved to: {output_path.absolute()}")

            # Print enhanced summary
            print("\n📊 ENHANCED ATHENA RAG STATISTICS SUMMARY")
            print("=" * 70)

            summary = self.stats["summary"]

            # Basic Statistics
            print("📚 DOCUMENT STATISTICS:")
            print(f"  • Total Documents: {summary.get('total_documents', 0)}")
            print(f"  • Total Tables: {summary.get('total_tables', 0)}")
            print(f"  • SQL Examples: {summary.get('sql_examples_count', 0)}")
            print(f"  • Document Types: {summary.get('document_types_count', 0)}")
            print(f"  • Avg SQL Complexity: {summary.get('avg_sql_complexity', 0)}")

            # Chunk Statistics
            print("\n🧩 CHUNK STATISTICS:")
            print(f"  • Total Chunks: {summary.get('total_chunks', 0)}")
            print(
                f"  • Avg Chunks per Document: {summary.get('avg_chunks_per_document', 0)}"
            )
            print(
                f"  • Documents with Chunks: {summary.get('documents_with_chunks', 0)}"
            )
            print(
                f"  • Chunk Types Available: {summary.get('chunk_types_available', 0)}"
            )

            # Embedding Statistics
            print("\n🎯 EMBEDDING STATISTICS:")
            print(f"  • Total Embeddings: {summary.get('total_embeddings', 0)}")
            print(f"  • Embedding Coverage: {summary.get('embedding_coverage', 0)}%")
            print(
                f"  • Embedding Dimension: {summary.get('embedding_dimension', 1024)}"
            )
            print(f"  • Unique Namespaces: {summary.get('unique_namespaces', 0)}")

            # Performance Metrics
            print("\n⚡ PERFORMANCE METRICS:")
            print(
                f"  • Storage Efficiency: {summary.get('storage_efficiency_percentage', 0)}%"
            )

            # Chunking by Type
            chunking_by_type = summary.get("chunking_by_type", {})
            if chunking_by_type:
                print("\n📊 CHUNKING BY DOCUMENT TYPE:")
                for doc_type, stats in chunking_by_type.items():
                    print(f"  • {doc_type.title()}:")
                    print(f"    - Documents: {stats.get('documents', 0)}")
                    print(f"    - Total Chunks: {stats.get('total_chunks', 0)}")
                    print(f"    - Avg Chunks/Doc: {stats.get('avg_chunks', 0)}")

            print("\n🗂️  DETAILED REPORTS SAVED:")
            print("  • chunk_distribution_stats.csv")
            print("  • chunking_efficiency_stats.csv")
            print("  • embedding_coverage_by_type_stats.csv")
            print("  • search_optimization_stats.csv")
            print("  • chunking_effectiveness_stats.csv")
            print("  • chunk_length_analysis_stats.csv")
            print("  • namespace_distribution_stats.csv")

            print("=" * 70)

        except Exception as e:
            logger.error(f"❌ Error collecting statistics: {e}")
            raise
        finally:
            if hasattr(self, "engine"):
                self.engine.dispose()


if __name__ == "__main__":
    stats_generator = AthenaRAGStats()
    stats_generator.run()
