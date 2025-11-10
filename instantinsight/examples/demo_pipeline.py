"""
Demo pipeline for natural language to SQL processing with semantic caching.

This demo showcases the production-ready QueryProcessor for NL2SQL processing.
"""

import json
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger

from src.query_processor import QueryProcessor

# Disable verbose debug logging from various sources to reduce noise
logging.disable(logging.DEBUG)
logger.disable("agno")
logger.remove()
logger.add(sys.stderr, level="INFO")

# Load environment variables from .env file
load_dotenv()


def display_results(result: dict):
    """
    Display query processing results in a formatted way.

    Args:
        result: QueryResult dictionary from QueryProcessor

    """
    print("\n" + "=" * 80)

    # Display cache info if available
    if result.get("cache_hit"):
        confidence = result.get("cache_confidence", 1.0)
        retrieval_ms = result.get("cache_retrieval_ms", 0)
        print(f"🎯 Cache Hit (confidence: {confidence:.3f}):")
        print(f"   Retrieved in {retrieval_ms}ms")
    else:
        print("💔 Cache Miss:")
        sql_gen_ms = result.get("sql_generation_ms", 0)
        print(f"   Generated in {sql_gen_ms}ms")
    print()

    # Display pipeline info if available
    if result.get("pipeline_stages"):
        stages = result["pipeline_stages"]
        total_ms = result.get("total_duration_ms", 0)
        print(f"Pipeline Execution ({total_ms}ms):")
        print("-" * 40)
        print(f"Stages: {', '.join(stages)}")
        print()

    # Display SQL
    if result.get("sql"):
        print("Generated SQL:")
        print("-" * 40)
        print(result["sql"])
        print()

    # Display error or data
    if result.get("error"):
        print(f"Error: {result['error']}")
    elif result.get("data") is not None:
        df = result["data"]
        row_count = result.get("row_count", len(df))
        print(f"Results ({row_count} rows):")
        print("-" * 40)
        print(df.to_string(max_rows=20, max_cols=10))

        if len(df) > 20:
            print(f"\n... showing first 20 of {len(df)} rows")

        # Display export information
        if result.get("export_path"):
            export_type = result.get("export_type", "unknown")
            export_path = result["export_path"]
            print(f"\n📁 Results exported ({export_type}): {export_path}")

    # Display visualization schema if available
    if result.get("visualization"):
        viz = result["visualization"]
        print("\n📊 Visualization Schema:")
        print("-" * 40)

        # Extract metadata
        if viz.get("metadata"):
            metadata = viz["metadata"]
            print(f"Chart Type: {metadata.get('chart_type', 'unknown')}")
            print(f"Confidence: {metadata.get('confidence', 0):.2%}")
            print(f"Reasoning: {metadata.get('reasoning', 'N/A')}")
            if metadata.get("insights"):
                print(f"Insights: {metadata.get('insights')}")
            print()

        # Pretty print the Plotly chart JSON
        if viz.get("chart"):
            print("Plotly Chart JSON:")
            print(json.dumps(viz["chart"], indent=2))
        elif viz.get("error"):
            print(f"Visualization Error: {viz['error']}")

    print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize QueryProcessor with default configuration
    # (pipeline and semantic caching enabled by default)
    processor = QueryProcessor()

    # Example query
    user_query = "How many products are discontinued for each category?"

    print(f"🚀 Processing query: {user_query}")
    result = processor.process_query(
        query=user_query,
        export_results=True,
        generate_visualization=True,
    )
    display_results(result)

    # Clean up
    processor.close()
