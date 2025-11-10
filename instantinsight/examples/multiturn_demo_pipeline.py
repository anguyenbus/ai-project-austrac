"""
Demo pipeline for multi-turn conversations with S3-backed session management.

This demo showcases the multi-turn conversation feature using SessionStore
for server-managed session state, matching the Lambda production architecture.
"""

import json
import logging
import os
import sys
import uuid
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger

from src.query_processor import QueryProcessor
from src.session import SessionStore, query_result_to_turn

# Disable verbose debug logging from various sources to reduce noise
logging.disable(logging.DEBUG)
logger.disable("agno")
logger.remove()
logger.add(sys.stderr, level="INFO")

# Load environment variables from .env file
load_dotenv()


def display_results(result: dict, turn_number: int = 1, session_id: str = None):
    """
    Display query processing results in a formatted way.

    Args:
        result: QueryResult dictionary from QueryProcessor
        turn_number: Current turn number for context
        session_id: Session identifier

    """
    print("\n" + "=" * 80)
    print(f"Turn {turn_number} Results (Session: {session_id[:8]}...)")
    print("=" * 80)

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


def run_demo():
    """
    Run a multi-turn conversation demo using SessionStore.

    This demo uses S3-backed session management, matching the Lambda architecture.
    Sessions are managed server-side with session_id, not client-supplied prior_turns.
    """
    # Initialize QueryProcessor with default configuration
    processor = QueryProcessor()

    # Initialize SessionStore (uses local S3 bucket for demo)
    session_bucket = os.getenv("SESSION_BUCKET", "instantinsight-session")
    session_prefix = os.getenv("SESSION_PREFIX", "sessions/")

    print(
        f"🔧 Initializing SessionStore (bucket: {session_bucket}, prefix: {session_prefix})"
    )
    session_store = SessionStore(bucket=session_bucket, prefix=session_prefix)

    # Generate session ID (in Lambda, this would be auto-generated or provided by client)
    session_id = str(uuid.uuid4())
    print(f"📝 Created new session: {session_id}\n")

    # Example multi-turn conversation
    print("🚀 Starting Multi-Turn Conversation Demo with SessionStore")
    print("This demo shows server-managed sessions using S3 storage.\n")

    # Turn 1: Initial query
    print("-" * 80)
    print("Turn 1: User asks about discontinued products by category")
    print("-" * 80)

    query1 = "How many products are discontinued for each category?"
    print(f"User: {query1}")

    # Load prior turns from session (empty for first turn)
    session = session_store.load(session_id)
    prior_turns = session["turns"]
    print(f"📥 Loaded {len(prior_turns)} prior turns from session")

    result1 = processor.process_query(
        query=query1,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result1, turn_number=1, session_id=session_id)

    # Save turn to session
    turn1 = query_result_to_turn(result1)
    session_store.append(session_id, turn1)
    print("💾 Saved turn to session (total: 1 turn)")

    # Turn 2: Follow-up query referencing previous context
    print("-" * 80)
    print("Turn 2: User asks for active products (referencing previous query)")
    print("-" * 80)

    query2 = "Now show me the active products for each category"
    print(f"User: {query2}")

    # Load prior turns from session
    session = session_store.load(session_id)
    prior_turns = session["turns"]
    print(f"📥 Loaded {len(prior_turns)} prior turns from session")

    result2 = processor.process_query(
        query=query2,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result2, turn_number=2, session_id=session_id)

    # Save turn to session
    turn2 = query_result_to_turn(result2)
    session_store.append(session_id, turn2)
    print("💾 Saved turn to session (total: 2 turns)")

    # Turn 3: Visualization change request
    print("-" * 80)
    print("Turn 3: User requests to change visualization type")
    print("-" * 80)

    query3 = "Change the chart type to pie chart"
    print(f"User: {query3}")

    session = session_store.load(session_id)
    prior_turns = session["turns"]
    print(f"📥 Loaded {len(prior_turns)} prior turns from session")

    result3 = processor.process_query(
        query=query3,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result3, turn_number=3, session_id=session_id)

    turn3 = query_result_to_turn(result3)
    session_store.append(session_id, turn3)
    print("💾 Saved turn to session (total: 3 turns)")

    # Turn 4: Reference to earlier query
    print("-" * 80)
    print("Turn 4: User references the first query with additional requirements")
    print("-" * 80)

    query4 = "Go back to the first query but add a filter for products with more than 5 units"
    print(f"User: {query4}")

    session = session_store.load(session_id)
    prior_turns = session["turns"]
    print(f"📥 Loaded {len(prior_turns)} prior turns from session")

    result4 = processor.process_query(
        query=query4,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result4, turn_number=4, session_id=session_id)

    turn4 = query_result_to_turn(result4)
    session_store.append(session_id, turn4)
    print("💾 Saved turn to session (total: 4 turns)")

    # Turn 5: Show session state
    print("-" * 80)
    print("Turn 5: Display session state from S3")
    print("-" * 80)

    session = session_store.load(session_id)
    print("Session State:")
    print("-" * 40)
    print(f"Session ID: {session['session_id']}")
    print(f"Total Turns: {len(session['turns'])}")
    print(f"Updated At: {session['updated_at']}")
    print("\nTurn History:")
    for i, turn in enumerate(session["turns"], 1):
        print(f"\nTurn {i}: {turn['content']}")
        if turn.get("sql"):
            sql_preview = turn["sql"][:80] + ("..." if len(turn["sql"]) > 80 else "")
            print(f"  SQL: {sql_preview}")
        print(f"  Success: {turn['success']}")
        print(f"  Cache Hit: {turn['cache_hit']}")
        if turn.get("row_count"):
            print(f"  Rows: {turn['row_count']}")
    print()

    # Turn 6: Query with no context (demonstrates context is optional)
    print("-" * 80)
    print("Turn 6: Query with no relevant context")
    print("-" * 80)

    query6 = "How many total orders were placed last month?"
    print(f"User: {query6}")

    session = session_store.load(session_id)
    prior_turns = session["turns"]
    print(f"📥 Loaded {len(prior_turns)} prior turns from session")

    result6 = processor.process_query(
        query=query6,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result6, turn_number=6, session_id=session_id)

    turn6 = query_result_to_turn(result6)
    session_store.append(session_id, turn6)
    print("💾 Saved turn to session (total: 6 turns)")

    # Demonstrate truncation with 7th turn
    print("-" * 80)
    print("Turn 7: Demonstrate automatic truncation to 6 turns")
    print("-" * 80)

    query7 = "Show me product categories"
    print(f"User: {query7}")

    session = session_store.load(session_id)
    print(f"📥 Loaded {len(session['turns'])} prior turns (before truncation)")

    prior_turns = session["turns"]
    result7 = processor.process_query(
        query=query7,
        prior_turns=prior_turns,
        export_results=False,
        generate_visualization=True,
    )
    display_results(result7, turn_number=7, session_id=session_id)

    turn7 = query_result_to_turn(result7)
    session_store.append(session_id, turn7)

    # Verify truncation
    session = session_store.load(session_id)
    print(
        f"💾 Session automatically truncated to {len(session['turns'])} turns (max 6)"
    )
    print(
        f"   Oldest turn removed: Turn 1 ('{session['turns'][0]['content'][:50]}...')"
    )

    # Clean up
    processor.close()

    print("\n" + "=" * 80)
    print("🎉 Multi-Turn Conversation Demo Complete!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("1. Sessions are managed server-side in S3 using session_id")
    print("2. Prior turns are automatically loaded from S3 before each query")
    print("3. New turns are saved to S3 after successful processing")
    print("4. Sessions automatically truncate to 6 most recent turns")
    print("5. This matches the Lambda production architecture exactly")
    print("\nSession cleanup:")
    print(
        f"- Session data stored in S3: s3://{session_bucket}/{session_prefix}session_{session_id}/"
    )
    print("- Use session_store.delete(session_id) to manually clean up")
    print("- Production: automatic cleanup via S3 lifecycle rules (30-day TTL)")


# Example usage
if __name__ == "__main__":
    run_demo()
