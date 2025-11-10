"""
Comprehensive tests for SQL formatter utilities.

This module contains rigorous tests for the SQL formatting functions,
including edge cases, error conditions, and various SQL constructs.
"""

import pytest
from src.utils.sql_formatter import (
    enforce_sql_spacing,
    extract_sql_from_text,
    format_sql_for_athena,
    validate_sql_spacing,
)


class TestEnforceSqlSpacing:
    """Test the enforce_sql_spacing function comprehensively."""

    def test_original_problematic_query(self):
        """Test the original query that prompted the fix."""
        sql = (
            'SELECT "transaction sub type", SUM("amount 1") as reconciliation_amount'
            "FROM awsdatacatalog.finanalysers.assetreconciliation"
            'GROUP BY "transaction sub type" ORDER BY "transaction sub type"'
        )

        expected = (
            'SELECT "transaction sub type", SUM("amount 1") as reconciliation_amount '
            "FROM awsdatacatalog.finanalysers.assetreconciliation "
            'GROUP BY "transaction sub type" ORDER BY "transaction sub type"'
        )

        result = enforce_sql_spacing(sql)
        assert result == expected

        # Ensure words containing "on" are preserved
        assert "transaction sub type" in result
        assert "reconciliation" in result
        assert "assetreconciliation" in result

    def test_from_keyword_spacing(self):
        """Test FROM keyword spacing in various contexts."""
        test_cases = [
            ("SELECT * FROM table", "SELECT * FROM table"),  # Already correct
            ("SELECT countFROM table", "SELECT count FROM table"),
            ("SELECT sum(amount)FROM users", "SELECT sum(amount) FROM users"),
            ('SELECT "column name"FROM table', 'SELECT "column name" FROM table'),
            ("SELECT COUNT(*)FROM table", "SELECT COUNT(*) FROM table"),
            ("SELECT id as totalFROM table", "SELECT id AS total FROM table"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_where_keyword_spacing(self):
        """Test WHERE keyword spacing in various contexts."""
        test_cases = [
            ("SELECT * WHERE id > 0", "SELECT * WHERE id > 0"),  # Already correct
            ("SELECT nameWHERE id > 0", "SELECT name WHERE id > 0"),
            (
                'SELECT countWHERE status = "active"',
                'SELECT count WHERE status = "active"',
            ),
            (
                'SELECT "user name"WHERE active = 1',
                'SELECT "user name" WHERE active = 1',
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_group_by_keyword_spacing(self):
        """Test GROUP BY multi-word keyword spacing."""
        test_cases = [
            (
                "SELECT count(*) GROUP BY id",
                "SELECT count(*) GROUP BY id",
            ),  # Already correct
            ("SELECT count(*)GROUP BY id", "SELECT count(*) GROUP BY id"),
            ("SELECT nameGROUP BY department", "SELECT name GROUP BY department"),
            (
                'SELECT "total amount"GROUP BY "user id"',
                'SELECT "total amount" GROUP BY "user id"',
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_order_by_keyword_spacing(self):
        """Test ORDER BY multi-word keyword spacing."""
        test_cases = [
            ("SELECT * ORDER BY name", "SELECT * ORDER BY name"),  # Already correct
            ("SELECT *ORDER BY name", "SELECT * ORDER BY name"),
            ("SELECT countORDER BY date", "SELECT count ORDER BY date"),
            (
                'SELECT "user name"ORDER BY created_at',
                'SELECT "user name" ORDER BY created_at',
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_join_keyword_spacing(self):
        """Test JOIN keyword spacing, including ON conditions."""
        test_cases = [
            # Regular JOINs
            (
                "SELECT * FROM users u JOIN orders o ON u.id = o.user_id",
                "SELECT * FROM users u JOIN orders o ON u.id = o.user_id",
            ),  # Already correct
            ("SELECT * FROM usersJOIN orders", "SELECT * FROM users JOIN orders"),
            (
                "SELECT * FROM users uJOIN orders o",
                "SELECT * FROM users u JOIN orders o",
            ),
            # ON conditions
            (
                "SELECT * FROM users u JOIN ordersON u.id = o.user_id",
                "SELECT * FROM users u JOIN orders ON u.id = o.user_id",
            ),
            (
                "SELECT * FROM users u JOIN orders oON u.id = o.user_id",
                "SELECT * FROM users u JOIN orders o ON u.id = o.user_id",
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_limit_and_other_keywords(self):
        """Test other SQL keywords spacing."""
        test_cases = [
            ("SELECT * LIMIT 10", "SELECT * LIMIT 10"),  # Already correct
            ("SELECT *LIMIT 10", "SELECT * LIMIT 10"),
            ("SELECT countLIMIT 5", "SELECT count LIMIT 5"),
            (
                "SELECT * FROM tableUNION SELECT * FROM other",
                "SELECT * FROM table UNION SELECT * FROM other",
            ),
            ("SELECT nameHAVING count > 0", "SELECT name HAVING count > 0"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_as_keyword_spacing(self):
        """Test AS keyword spacing for aliases."""
        test_cases = [
            (
                "SELECT name AS customer_name",
                "SELECT name AS customer_name",
            ),  # Already correct
            ("SELECT nameAS customer_name", "SELECT name AS customer_name"),
            ('SELECT "user name"AS alias', 'SELECT "user name" AS alias'),
            ("SELECT count(*)AS total", "SELECT count(*) AS total"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_words_containing_keywords_preserved(self):
        """Test that words containing SQL keywords are not broken apart."""
        test_cases = [
            # Words containing "on"
            (
                "SELECT transaction_type FROM table",
                "SELECT transaction_type FROM table",
            ),
            (
                "SELECT reconciliation_amount FROM table",
                "SELECT reconciliation_amount FROM table",
            ),
            (
                "SELECT configuration_id FROM table",
                "SELECT configuration_id FROM table",
            ),
            (
                "SELECT demonstration_data FROM table",
                "SELECT demonstration_data FROM table",
            ),
            # Words containing "or"
            ("SELECT order_id FROM table", "SELECT order_id FROM table"),
            ("SELECT author_name FROM table", "SELECT author_name FROM table"),
            ("SELECT corporate_id FROM table", "SELECT corporate_id FROM table"),
            # Words containing "and"
            ("SELECT standard_rate FROM table", "SELECT standard_rate FROM table"),
            ("SELECT command_type FROM table", "SELECT command_type FROM table"),
            ("SELECT brand_name FROM table", "SELECT brand_name FROM table"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_operator_spacing(self):
        """Test spacing around operators."""
        test_cases = [
            ("SELECT * WHERE id=1", "SELECT * WHERE id = 1"),
            ("SELECT * WHERE count>10", "SELECT * WHERE count > 10"),
            ("SELECT * WHERE amount<=100", "SELECT * WHERE amount <= 100"),
            (
                'SELECT * WHERE status!="inactive"',
                'SELECT * WHERE status != "inactive"',
            ),
            (
                'SELECT * WHERE date<>"2023-01-01"',
                'SELECT * WHERE date <> "2023-01-01"',
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_comma_spacing(self):
        """Test comma spacing."""
        test_cases = [
            ("SELECT id,name,email", "SELECT id, name, email"),
            ("SELECT id ,name , email", "SELECT id, name, email"),
            ("SELECT id  ,  name  ,  email", "SELECT id, name, email"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_parentheses_spacing(self):
        """Test spacing around parentheses."""
        test_cases = [
            ("SELECT COUNT( * )", "SELECT COUNT(*)"),
            ("SELECT SUM( amount )", "SELECT SUM(amount)"),
            (
                "SELECT name FROM table WHERE id IN ( 1, 2, 3 )",
                "SELECT name FROM table WHERE id IN (1, 2, 3)",
            ),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_complex_query_spacing(self):
        """Test complex queries with multiple spacing issues."""
        complex_sql = (
            'SELECT "user name",COUNT(*)as total,SUM("order amount")as revenue'
            "FROM users uJOIN orders oON u.id=o.user_id"
            'WHERE u.status="active"AND o.date>="2023-01-01"'
            'GROUP BY"user name"ORDER BY revenue DESC'
        )

        expected = (
            'SELECT "user name", COUNT(*) AS total, SUM("order amount") AS revenue '
            "FROM users u JOIN orders o ON u.id = o.user_id "
            'WHERE u.status = "active" AND o.date >= "2023-01-01" '
            'GROUP BY "user name" ORDER BY revenue DESC'
        )

        result = enforce_sql_spacing(complex_sql)
        assert result == expected

    def test_case_insensitive_keywords(self):
        """Test that keyword matching is case insensitive."""
        test_cases = [
            ("select countfrom table", "select count from table"),
            ("SELECT countwhere id > 0", "SELECT count where id > 0"),
            ("Select namegroup by dept", "Select name group by dept"),
        ]

        for input_sql, expected in test_cases:
            result = enforce_sql_spacing(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_empty_and_none_input(self):
        """Test edge cases with empty or None input."""
        assert enforce_sql_spacing("") == ""
        assert enforce_sql_spacing(None) is None
        assert enforce_sql_spacing("   ") == ""

    def test_already_properly_spaced(self):
        """Test that already properly spaced SQL is not changed."""
        properly_spaced = (
            'SELECT "user name", COUNT(*) AS total '
            "FROM users u JOIN orders o ON u.id = o.user_id "
            'WHERE u.status = "active" '
            'GROUP BY "user name" ORDER BY total DESC'
        )

        result = enforce_sql_spacing(properly_spaced)
        assert result == properly_spaced


class TestValidateSqlSpacing:
    """Test the validate_sql_spacing function."""

    def test_valid_spacing(self):
        """Test validation of properly spaced SQL."""
        valid_sql = (
            "SELECT name, count FROM users WHERE id > 0 GROUP BY name ORDER BY count"
        )
        is_valid, issues = validate_sql_spacing(valid_sql)
        assert is_valid
        assert len(issues) == 0

    def test_invalid_spacing_detection(self):
        """Test detection of spacing issues."""
        invalid_sql = "SELECT nameFROM usersWHERE id > 0GROUP BY nameORDER BY count"
        is_valid, issues = validate_sql_spacing(invalid_sql)
        assert not is_valid
        assert len(issues) > 0
        assert any("FROM" in issue for issue in issues)

    def test_false_positive_avoidance(self):
        """Test that words containing keywords don't trigger false positives."""
        sql_with_embedded_keywords = "SELECT transaction_type FROM reconciliation_table"
        is_valid, issues = validate_sql_spacing(sql_with_embedded_keywords)
        # This should be valid since the keywords are part of larger words
        assert is_valid or len(issues) == 0


class TestFormatSqlForAthena:
    """Test the format_sql_for_athena function."""

    def test_athena_spacing_enforcement(self):
        """Test that Athena formatting includes spacing enforcement."""
        sql = "SELECT countFROM tableWHERE id > 0"
        result = format_sql_for_athena(sql)
        assert "count FROM" in result
        assert "table WHERE" in result

    def test_backtick_conversion(self):
        """Test conversion of backticks to double quotes."""
        sql = "SELECT `column name`, `other column` FROM `table name`"
        expected = 'SELECT "column name", "other column" FROM "table name"'
        result = format_sql_for_athena(sql)
        assert result == expected

    def test_function_case_conversion(self):
        """Test conversion of function names to lowercase."""
        test_cases = [
            ("SELECT COUNT(*) FROM table", "SELECT count(*) FROM table"),
            (
                "SELECT SUM(amount), AVG(value) FROM table",
                "SELECT sum(amount), avg(value) FROM table",
            ),
            (
                "SELECT MAX(date), MIN(date) FROM table",
                "SELECT max(date), min(date) FROM table",
            ),
            (
                'SELECT COALESCE(name, "unknown") FROM table',
                'SELECT coalesce(name, "unknown") FROM table',
            ),
        ]

        for input_sql, expected in test_cases:
            result = format_sql_for_athena(input_sql)
            assert result == expected, f"Failed for input: {input_sql}"

    def test_combined_athena_formatting(self):
        """Test combined Athena formatting features."""
        sql = "SELECT `user name`,COUNT(*)as totalFROM `user table`WHERE id>0"
        expected = (
            'SELECT "user name", count(*) AS total FROM "user table" WHERE id > 0'
        )
        result = format_sql_for_athena(sql)
        assert result == expected


class TestExtractSqlFromText:
    """Test the extract_sql_from_text function."""

    def test_extract_from_markdown_sql_block(self):
        """Test extraction from markdown SQL code blocks."""
        text = """
        Here's the query you need:
        
        ```sql
        SELECT name, count FROM users WHERE id > 0
        ```
        
        This should work well.
        """

        expected = "SELECT name, count FROM users WHERE id > 0"
        result = extract_sql_from_text(text)
        assert result.strip() == expected

    def test_extract_from_generic_code_block(self):
        """Test extraction from generic code blocks."""
        text = """
        ```
        SELECT name, count FROM users WHERE id > 0
        ```
        """

        expected = "SELECT name, count FROM users WHERE id > 0"
        result = extract_sql_from_text(text)
        assert result.strip() == expected

    def test_extract_from_plain_text(self):
        """Test extraction from plain text starting with SQL keyword."""
        text = """
        The query is:
        SELECT name, count FROM users WHERE id > 0
        And that's it.
        """

        result = extract_sql_from_text(text)
        assert "SELECT name, count FROM users WHERE id > 0" in result

    def test_extract_with_spacing_enforcement(self):
        """Test that extracted SQL has spacing enforced."""
        text = """
        ```sql
        SELECT nameFROM usersWHERE id > 0
        ```
        """

        result = extract_sql_from_text(text)
        assert "name FROM" in result
        assert "users WHERE" in result

    def test_no_sql_found(self):
        """Test behavior when no SQL is found."""
        text = "This is just regular text with no SQL queries."
        result = extract_sql_from_text(text)
        assert result == text.strip()

    def test_multiple_sql_keywords(self):
        """Test extraction starting from different SQL keywords."""
        test_cases = [
            "WITH cte AS (SELECT * FROM table) SELECT * FROM cte",
            "INSERT INTO table VALUES (1, 'name')",
            "UPDATE table SET name = 'new' WHERE id = 1",
            "DELETE FROM table WHERE id = 1",
            "CREATE TABLE test (id INT, name VARCHAR(50))",
        ]

        for sql in test_cases:
            result = extract_sql_from_text(f"Here's a query: {sql}")
            assert sql.split()[0] in result  # First keyword should be preserved


class TestRegressionCases:
    """Test specific regression cases and reported bugs."""

    def test_original_bug_report(self):
        """Test the exact case that was reported as problematic."""
        sql = (
            'SELECT "transaction sub type", SUM("amount 1") as reconciliation_amount'
            "FROM awsdatacatalog.finanalysers.assetreconciliation"
            'GROUP BY "transaction sub type" ORDER BY "transaction sub type"'
        )

        result = enforce_sql_spacing(sql)

        # Should have proper spacing
        assert "reconciliation_amount FROM" in result
        assert "assetreconciliation GROUP BY" in result
        assert 'sub type" ORDER BY' in result

        # Should NOT break words containing keywords
        assert "transacti on" not in result
        assert "reconciliati on" not in result
        assert "assetreconciliati on" not in result

    def test_edge_case_combinations(self):
        """Test combinations of edge cases that might interact poorly."""
        test_cases = [
            # ON in table names with JOIN
            "SELECT * FROM configuration_table c JOIN transaction_log t ON c.id = t.config_id",
            # Multiple keywords in sequence
            "SELECT countFROM tableWHERE statusORDER BY date",
            # Keywords in quotes
            'SELECT "FROM column", "WHERE value" FROM "ORDER table"',
            # Complex nested queries
            (
                "SELECT (SELECT countFROM inner_table) as total"
                "FROM outer_tableWHERE id IN (SELECT idFROM related_table)"
            ),
        ]

        for sql in test_cases:
            # Should not raise exceptions
            result = enforce_sql_spacing(sql)
            assert result is not None
            assert len(result) > 0

            # Should preserve quoted content
            if '"' in sql:
                # Quoted strings should remain intact
                original_quotes = sql.count('"')
                result_quotes = result.count('"')
                assert original_quotes == result_quotes


@pytest.mark.parametrize(
    "function_name",
    [
        "enforce_sql_spacing",
        "validate_sql_spacing",
        "format_sql_for_athena",
        "extract_sql_from_text",
    ],
)
def test_function_exists_and_callable(function_name):
    """Test that all expected functions exist and are callable."""
    from src.utils import sql_formatter

    assert hasattr(sql_formatter, function_name)
    func = getattr(sql_formatter, function_name)
    assert callable(func)


class TestPerformance:
    """Test performance characteristics of the SQL formatter."""

    def test_large_query_performance(self):
        """Test that the formatter handles large queries reasonably."""
        # Generate a large query
        columns = [f'"column_{i}"' for i in range(100)]
        large_sql = f"SELECT {', '.join(columns)}FROM large_tableWHERE id > 0GROUP BY {columns[0]}ORDER BY {columns[1]}"

        # Should complete without hanging
        result = enforce_sql_spacing(large_sql)
        assert "FROM large_table" in result
        assert "table WHERE" in result

    def test_repeated_processing_consistency(self):
        """Test that repeated processing produces consistent results."""
        sql = "SELECT nameFROM tableWHERE id > 0"

        result1 = enforce_sql_spacing(sql)
        result2 = enforce_sql_spacing(result1)
        result3 = enforce_sql_spacing(result2)

        # Results should be identical after first formatting
        assert result1 == result2 == result3
