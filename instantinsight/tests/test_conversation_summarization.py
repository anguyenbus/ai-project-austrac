"""Test conversation summarization to verify token reduction."""

import json

from flask_app.modification_handler import ModificationHandler


def test_user_message_summarization():
    """Test that user messages are appropriately summarized."""
    handler = ModificationHandler()

    short_user = {"type": "user", "message": "make it blue"}
    assert handler._summarize_conversation_turn(short_user) == "make it blue"

    long_user = {
        "type": "user",
        "message": "Can you please change the visualization to show the data as a "
        "horizontal bar chart with the categories sorted in descending order "
        "and use a blue color scheme?",
    }
    summary = handler._summarize_conversation_turn(long_user)
    assert len(summary) <= 153
    assert "change" in summary.lower()


def test_assistant_message_summarization():
    """Test that assistant messages are compressed to standard responses."""
    handler = ModificationHandler()

    viz_created = {
        "type": "assistant",
        "message": "I've created a visualization for your question. You can now "
        "ask me to modify it (e.g., 'make it a pie chart', 'show as horizontal "
        "bars', etc.)",
    }
    assert handler._summarize_conversation_turn(viz_created) == "Created visualization"

    viz_updated = {
        "type": "assistant",
        "message": "I've updated the chart to use blue colors and sorted the bars.",
    }
    assert handler._summarize_conversation_turn(viz_updated) == "Updated visualization"

    error_msg = {"type": "assistant", "message": "❌ Error: Could not process request"}
    assert handler._summarize_conversation_turn(error_msg) == "Error occurred"


def test_schema_size_validation():
    """Test that schema compaction validates token budget."""
    handler = ModificationHandler()

    large_trace = {
        "data": [
            {
                "type": "scatter",
                "x": list(range(1000)),
                "y": list(range(1000)),
                "marker": {"color": list(range(1000))},
            }
        ],
        "layout": {"title": "Test Chart"},
    }

    compact = handler._build_compact_schema(large_trace)
    compact_json = json.dumps(compact)
    tokens = len(compact_json) // 4

    print(f"Original would be ~{len(json.dumps(large_trace)) // 4} tokens")
    print(f"Compact is ~{tokens} tokens")
    assert tokens < 1000


def test_historical_context_truncation():
    """Test that historical context respects token budget."""
    handler = ModificationHandler()

    long_memory = "x" * 3000
    context = {"original_question": "test"}

    # Populate in-memory storage with long messages
    handler._conversation_memory["test_viz"] = [
        {"role": "user", "content": long_memory},
        {"role": "assistant", "content": long_memory},
        {"role": "user", "content": long_memory},
        {"role": "assistant", "content": long_memory},
        {"role": "user", "content": long_memory},
        {"role": "assistant", "content": long_memory},
    ]

    result = handler._get_historical_context("test_viz", context)
    result_tokens = len(result) // 4

    print(f"Historical context: {len(result)} chars (~{result_tokens} tokens)")
    assert result_tokens < 600


if __name__ == "__main__":
    test_user_message_summarization()
    print("✓ User message summarization works")

    test_assistant_message_summarization()
    print("✓ Assistant message summarization works")

    test_schema_size_validation()
    print("✓ Schema compaction validates size")

    test_historical_context_truncation()
    print("✓ Historical context respects budget")

    print("\nAll summarization tests passed!")
