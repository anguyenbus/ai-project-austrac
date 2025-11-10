"""Tests for semantic cache with conversation context."""

from src.cache.semantic_cache import SemanticCache


class TestCacheContext:
    """Test cache context awareness."""

    def test_cache_key_with_prior_turns(self):
        """Test cache key includes last 3 turns."""
        cache = SemanticCache()

        prior_turns = [
            {"content": "Q1", "sql": "S1"},
            {"content": "Q2", "sql": "S2"},
            {"content": "Q3", "sql": "S3"},
            {"content": "Q4", "sql": "S4"},
        ]

        # Should use last 3 turns (Q2, Q3, Q4)
        key1 = cache._generate_cache_key("user1", "Q5", prior_turns)

        # Same query with different context should produce different key
        different_context = [
            {"content": "Different Q2", "sql": "Different S2"},
            {"content": "Different Q3", "sql": "Different S3"},
            {"content": "Different Q4", "sql": "Different S4"},
        ]
        key2 = cache._generate_cache_key("user1", "Q5", different_context)

        # Keys should be different due to different context
        assert key1 != key2

        # Same query with same last 3 turns should produce same key
        same_context = [
            {"content": "Q2", "sql": "S2"},
            {"content": "Q3", "sql": "S3"},
            {"content": "Q4", "sql": "S4"},
        ]
        key3 = cache._generate_cache_key("user1", "Q5", same_context)
        assert key1 == key3  # Same last 3 turns should match

    def test_cache_key_uses_last_3_turns(self):
        """Test cache key only uses last 3 turns, not all."""
        cache = SemanticCache()

        # Context with 4 turns
        context_with_4 = [
            {"content": "Very old query", "sql": "SELECT old_data"},
            {"content": "Recent query 1", "sql": "SELECT recent_data_1"},
            {"content": "Recent query 2", "sql": "SELECT recent_data_2"},
            {"content": "Recent query 3", "sql": "SELECT recent_data_3"},
        ]

        # Context with only last 3 turns (dropping the first one)
        context_with_3 = [
            {"content": "Recent query 1", "sql": "SELECT recent_data_1"},
            {"content": "Recent query 2", "sql": "SELECT recent_data_2"},
            {"content": "Recent query 3", "sql": "SELECT recent_data_3"},
        ]

        # Generate keys
        key1 = cache._generate_cache_key("user1", "New query", context_with_4)
        key2 = cache._generate_cache_key("user1", "New query", context_with_3)

        # Keys should be the same (only last 3 matter)
        assert key1 == key2

    def test_cache_key_isolation_across_sessions(self):
        """Test different contexts produce different cache keys."""
        cache = SemanticCache()

        # Same query with different contexts
        context1 = [{"content": "Show products", "sql": "SELECT * FROM products"}]

        context2 = [{"content": "Show customers", "sql": "SELECT * FROM customers"}]

        key1 = cache._generate_cache_key("user1", "Show me more", context1)
        key2 = cache._generate_cache_key("user1", "Show me more", context2)

        # Keys should be different
        assert key1 != key2

    def test_cache_key_empty_prior_turns(self):
        """Test cache key with empty prior turns."""
        cache = SemanticCache()

        # Empty list
        key1 = cache._generate_cache_key("user1", "Test query", [])

        # None
        key2 = cache._generate_cache_key("user1", "Test query", None)

        # No prior turns parameter
        key3 = cache._generate_cache_key("user1", "Test query")

        # All should be the same
        assert key1 == key2 == key3

    def test_cache_key_sql_truncation(self):
        """Test SQL is truncated to first 50 characters in cache key."""
        cache = SemanticCache()

        long_sql = "SELECT * FROM very_long_table_name WHERE some_very_long_condition = 'some_very_long_value' AND another_condition"
        short_sql = "SELECT * FROM very_long_table_name WHERE some_very_long_condition = 'some_very_long_value' AND another_condition_that_is_different"
        same_truncated_sql = short_sql[:50]  # Same first 50 chars

        # Context with long SQL
        context1 = [{"content": "Test query", "sql": long_sql}]

        # Context with different SQL but same first 50 chars
        context2 = [{"content": "Test query", "sql": same_truncated_sql}]

        # Context with different first 50 chars
        context3 = [{"content": "Test query", "sql": "Different SQL statement"}]

        # Generate keys
        key1 = cache._generate_cache_key("user1", "New query", context1)
        key2 = cache._generate_cache_key("user1", "New query", context2)
        key3 = cache._generate_cache_key("user1", "New query", context3)

        # Keys 1 and 2 should be the same (same first 50 chars)
        assert key1 == key2

        # Key 3 should be different
        assert key1 != key3
