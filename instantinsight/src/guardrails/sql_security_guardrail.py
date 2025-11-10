"""SQL Security Guardrail - Ensure queries are for legitimate data retrieval only."""

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp


@dataclass
class SecurityViolation:
    """Security violation details."""

    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str


@dataclass
class ValidationResult:
    """Security validation result."""

    is_safe: bool
    violations: list[SecurityViolation]
    sanitized_sql: str | None = None


class SQLSecurityGuardrail:
    """
    Validates SQL queries for security threats.

    Ensures queries are for data retrieval only.
    """

    # Patterns that indicate debugging or testing
    DEBUG_PATTERNS = [
        (r"\b1\s*=\s*1\b", "Always-true condition (1=1)"),
        (r"\b2\s*=\s*2\b", "Always-true condition (2=2)"),
        (r"\bTRUE\s*=\s*TRUE\b", "Always-true condition"),
        (r"WHERE\s+1\b", "WHERE 1 clause"),
        (r"OR\s+1\s*=\s*1", "OR with always-true condition"),
    ]

    # SQL injection patterns
    INJECTION_PATTERNS = [
        (
            r";\s*(DROP|DELETE|TRUNCATE|UPDATE|INSERT|ALTER|CREATE)\s+",
            "SQL command injection",
        ),
        (r"\'\s*;\s*--", "Quote termination with comment"),
        (r"UNION\s+(ALL\s+)?SELECT", "UNION SELECT injection"),
        (r"(EXEC|EXECUTE|xp_cmdshell)", "Command execution attempt"),
        (r"(information_schema|sysobjects|syscolumns)", "System table access"),
    ]

    # Forbidden SQL functions
    FORBIDDEN_FUNCTIONS = {
        "sleep",
        "pg_sleep",
        "waitfor",
        "delay",
        "exec",
        "execute",
        "xp_cmdshell",
        "load_file",
        "into_outfile",
        "user",
        "database",
        "version",
    }

    # System tables that should not be accessed
    SYSTEM_TABLES = {
        "information_schema",
        "pg_catalog",
        "mysql",
        "sys",
        "sysobjects",
        "syscolumns",
        "sysusers",
    }

    def __init__(self, max_joins: int = 5, max_subqueries: int = 3):
        """
        Initialize guardrail.

        Args:
            max_joins: Maximum allowed JOINs
            max_subqueries: Maximum allowed subqueries

        """
        self.max_joins = max_joins
        self.max_subqueries = max_subqueries
        self.debug_regex = [
            (re.compile(p, re.IGNORECASE), d) for p, d in self.DEBUG_PATTERNS
        ]
        self.injection_regex = [
            (re.compile(p, re.IGNORECASE), d) for p, d in self.INJECTION_PATTERNS
        ]

    def validate(self, sql: str) -> ValidationResult:
        """
        Validate SQL query for security threats.

        Args:
            sql: SQL query to validate

        Returns:
            ValidationResult with violations if any

        """
        if not sql or not sql.strip():
            return ValidationResult(
                is_safe=False,
                violations=[SecurityViolation("EMPTY_QUERY", "LOW", "Empty query")],
            )

        violations = []

        # Check for debugging patterns
        violations.extend(
            self._check_patterns(sql, self.debug_regex, "DEBUG_PATTERN", "HIGH")
        )

        # Check for injection patterns
        violations.extend(
            self._check_patterns(sql, self.injection_regex, "INJECTION", "CRITICAL")
        )

        # Parse and validate SQL structure
        try:
            parsed = sqlglot.parse_one(sql)

            # Only allow SELECT statements
            if not isinstance(parsed, exp.Select):
                violations.append(
                    SecurityViolation(
                        "FORBIDDEN_OPERATION",
                        "CRITICAL",
                        f"Only SELECT allowed, found {type(parsed).__name__}",
                    )
                )

            # Check for forbidden operations in the query tree
            violations.extend(self._check_forbidden_operations(parsed))

            # Check for forbidden functions
            violations.extend(self._check_forbidden_functions(parsed))

            # Check for system tables
            violations.extend(self._check_system_tables(parsed))

            # Check query complexity
            violations.extend(self._check_complexity(parsed))

        except Exception as e:
            violations.append(
                SecurityViolation(
                    "PARSE_ERROR", "HIGH", f"Failed to parse SQL: {str(e)}"
                )
            )

        # Determine if safe
        is_safe = not any(v.severity in ["HIGH", "CRITICAL"] for v in violations)

        # Generate sanitized SQL if safe enough
        sanitized_sql = self._sanitize(sql) if is_safe else None

        return ValidationResult(
            is_safe=is_safe, violations=violations, sanitized_sql=sanitized_sql
        )

    def _check_patterns(
        self, sql: str, patterns: list, violation_type: str, severity: str
    ) -> list[SecurityViolation]:
        """Check SQL against regex patterns."""
        violations = []
        for pattern, description in patterns:
            if pattern.search(sql):
                violations.append(
                    SecurityViolation(violation_type, severity, description)
                )
        return violations

    def _check_forbidden_operations(
        self, parsed: exp.Expression
    ) -> list[SecurityViolation]:
        """Check for forbidden SQL operations."""
        violations = []

        forbidden_types = (
            exp.Update,
            exp.Delete,
            exp.Insert,
            exp.Drop,
            exp.Create,
            exp.Alter,
            exp.Truncate,
        )

        for node in parsed.walk():
            if isinstance(node, forbidden_types):
                violations.append(
                    SecurityViolation(
                        "FORBIDDEN_OPERATION",
                        "CRITICAL",
                        f"{type(node).__name__} operation not allowed",
                    )
                )

        return violations

    def _check_forbidden_functions(
        self, parsed: exp.Expression
    ) -> list[SecurityViolation]:
        """Check for forbidden SQL functions."""
        violations = []

        for func in parsed.find_all(exp.Func):
            func_name = str(func.this).lower() if hasattr(func, "this") else ""

            if func_name in self.FORBIDDEN_FUNCTIONS:
                violations.append(
                    SecurityViolation(
                        "FORBIDDEN_FUNCTION",
                        "HIGH",
                        f"Function {func_name} not allowed",
                    )
                )

        return violations

    def _check_system_tables(self, parsed: exp.Expression) -> list[SecurityViolation]:
        """Check for system table access."""
        violations = []

        for table in parsed.find_all(exp.Table):
            table_name = str(table.name).lower()

            for system_table in self.SYSTEM_TABLES:
                if system_table in table_name:
                    violations.append(
                        SecurityViolation(
                            "SYSTEM_TABLE",
                            "HIGH",
                            f"System table {table_name} access not allowed",
                        )
                    )
                    break

        return violations

    def _check_complexity(self, parsed: exp.Expression) -> list[SecurityViolation]:
        """Check query complexity."""
        violations = []

        # Count JOINs
        joins = list(parsed.find_all(exp.Join))
        if len(joins) > self.max_joins:
            violations.append(
                SecurityViolation(
                    "EXCESSIVE_JOINS",
                    "MEDIUM",
                    f"Too many JOINs: {len(joins)} > {self.max_joins}",
                )
            )

        # Check for cartesian products
        for join in joins:
            if not join.on and not join.using:
                violations.append(
                    SecurityViolation(
                        "CARTESIAN_PRODUCT",
                        "HIGH",
                        "JOIN without condition creates cartesian product",
                    )
                )

        # Count subqueries
        subqueries = list(parsed.find_all(exp.Subquery))
        if len(subqueries) > self.max_subqueries:
            violations.append(
                SecurityViolation(
                    "EXCESSIVE_SUBQUERIES",
                    "MEDIUM",
                    f"Too many subqueries: {len(subqueries)} > {self.max_subqueries}",
                )
            )

        return violations

    def _sanitize(self, sql: str) -> str:
        """Add safety limits to SQL."""
        sanitized = sql.strip()

        # Remove SQL comments
        sanitized = re.sub(r"--.*$", "", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        # Add LIMIT if not present
        if "LIMIT" not in sanitized.upper():
            sanitized = f"{sanitized} LIMIT 1000"

        return sanitized
