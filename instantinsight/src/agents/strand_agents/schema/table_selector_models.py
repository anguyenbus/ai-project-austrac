"""
Data models and utilities for TableAgent selection.

Contains Pydantic models and helper functions used by the Agno-based TableAgent.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# Export additional models for compatibility
__all__ = [
    "clean_schema",
    "TableAnalysis",
    "JoinRelationship",
    "DuplicateTableAnalysis",
    "UnionAnalysis",
    "LLMTableSelectionResult",
    "TableSelectionResult",
    "TableMetadata",
]


def clean_schema(schema_string):
    """Clean and deduplicate schema string."""
    # Replace \n with actual line breaks
    cleaned = schema_string.replace("\\n", "\n")

    # Split into lines and remove duplicates while preserving order
    lines = cleaned.split("\n")
    seen_tables = set()
    cleaned_lines = []
    skip_until_next_table = False

    for line in lines:
        if line.startswith("Table Name:"):
            table_name = line
            if table_name in seen_tables:
                skip_until_next_table = True
                continue
            else:
                seen_tables.add(table_name)
                skip_until_next_table = False
                cleaned_lines.append(line)
        elif not skip_until_next_table:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


class TableAnalysis(BaseModel):
    """Analysis of a single table's relevance to the query."""

    table_name: str = Field(description="Full table name (schema.table or just table)")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Relevance confidence (0.0 to 1.0)"
    )
    relevance_reasoning: str = Field(
        description="Why this table is relevant to the query"
    )
    key_columns: list[str] = Field(description="Most relevant columns for this query")
    table_purpose: str = Field(description="What this table is used for")


class JoinRelationship(BaseModel):
    """Potential JOIN relationship between Analyser tables - USE WITH EXTREME CAUTION."""

    table1: str = Field(description="First Analyser table name")
    table2: str = Field(description="Second Analyser table name")
    join_type: str = Field(
        description="Type of relational join: INNER, LEFT, RIGHT, FULL OUTER (NEVER UNION)",
        pattern=r"^(INNER|LEFT|RIGHT|FULL\s+OUTER|CROSS)(\s+JOIN)?$",
    )
    join_condition: str = Field(
        description="EXACT join condition with verified column names (e.g., analyser1.customer_code = analyser2.customer_code)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this join relationship (should be high for Analyser joins)",
    )
    is_necessary: bool = Field(
        default=False,
        description="True ONLY if query absolutely cannot be answered with single Analyser AND join relationship is verified safe",
    )

    # New fields for safety
    relationship_verified: bool = Field(
        default=False,
        description="True only if the linking relationship between these Analysers is predefined and documented",
    )
    data_integrity_risk: str = Field(
        description="Assessment of data integrity risk: LOW (verified relationship), MEDIUM (likely safe), HIGH (dangerous guess)"
    )
    alternative_single_table: str | None = Field(
        default=None,
        description="If available, suggest a single Analyser table that could answer most of this query instead",
    )


class DuplicateTableAnalysis(BaseModel):
    """Analysis of potentially duplicate or similar tables."""

    similar_tables: list[str] = Field(
        description="Tables that appear to be duplicates or very similar"
    )
    recommended_table: str = Field(
        description="Most authoritative/complete table to use"
    )
    similarity_reason: str = Field(
        description="Why these tables are considered similar/duplicates"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence that these are duplicates"
    )


class UnionAnalysis(BaseModel):
    """Analysis for UNION operations - only when truly necessary."""

    tables_to_union: list[str] = Field(
        description="Tables with similar schemas that could be combined"
    )
    union_type: str = Field(
        description="UNION or UNION ALL", pattern=r"^UNION(\s+ALL)?$"
    )
    justification: str = Field(
        description="Why UNION is necessary (e.g., combining similar data from multiple time periods)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence that UNION is the right approach"
    )

    # Validation fields to prevent inappropriate UNION suggestions
    union_purpose: str = Field(
        description="Specific purpose: 'frequency_analysis', 'historical_comparison', 'data_consolidation', or 'comprehensive_search'"
    )
    expected_benefit: str = Field(
        description="What benefit does UNION provide that JOIN cannot? (e.g., 'combines rows from similar tables', 'aggregates across time periods')"
    )
    alternative_considered: str = Field(
        description="Why JOIN or single table query wouldn't work for this specific case"
    )
    duplicate_check_performed: bool = Field(
        description="True if you verified these tables are NOT duplicates of each other"
    )


class LLMTableSelectionResult(BaseModel):
    """Structured result from LLM table selection analysis."""

    # Core selection results
    selected_tables: list[TableAnalysis] = Field(
        description="Top 3-5 most relevant tables for the query, ranked by relevance"
    )

    # Additional context - only when truly needed
    related_tables: list[str] = Field(
        default=[],
        description="Additional tables that might be useful ONLY if primary tables are insufficient",
    )

    # JOIN analysis - only necessary joins
    suggested_joins: list[JoinRelationship] = Field(
        default=[],
        description="ONLY necessary JOIN relationships between selected tables (excludes UNION)",
    )

    # UNION analysis - separate and only when truly necessary
    union_analysis: UnionAnalysis | None = Field(
        default=None,
        description="UNION analysis - only suggest if query requires combining similar data from multiple tables",
    )

    # Duplicate table detection
    duplicate_analysis: DuplicateTableAnalysis | None = Field(
        default=None,
        description="Analysis of potentially duplicate tables - recommend single authoritative table",
    )

    # Query strategy
    requires_multiple_tables: bool = Field(
        description="True only if query cannot be answered with a single table"
    )

    # Analysis metadata
    query_analysis: str = Field(description="Analysis of what the user is asking for")

    selection_reasoning: str = Field(
        description="Overall reasoning for table selection strategy"
    )

    complexity_assessment: str = Field(
        description="Query complexity: simple, moderate, complex"
    )


@dataclass
class TableSelectionResult:
    """Result from TableAgent table selection process."""

    # Primary tables most relevant to the query
    selected_tables: list[str] = field(default_factory=list)

    # Additional related tables that might be useful
    related_tables: list[str] = field(default_factory=list)

    # Suggested join paths between tables (only necessary joins)
    join_paths: list[dict[str, Any]] = field(default_factory=list)

    # Union analysis (only when truly necessary)
    union_analysis: dict[str, Any] | None = None

    # Duplicate analysis (detecting similar tables)
    duplicate_analysis: dict[str, Any] | None = None

    # Whether query requires multiple tables
    requires_multiple_tables: bool = False

    # Confidence scores for each table (0.0 to 1.0)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    # Explanation of selection reasoning
    selection_reasoning: str = ""

    # Additional metadata and context
    metadata: dict[str, Any] = field(default_factory=dict)

    # Search strategy used
    strategy_used: str = "vector_search"

    # Total tables considered
    total_tables_analyzed: int = 0

    # Schema context
    schema_context: str = ""


@dataclass
class TableMetadata:
    """Metadata for a database table."""

    name: str
    full_name: str  # Fully qualified name (e.g., awsdatacatalog.database.table)
    columns: list[str] = field(default_factory=list)
    description: str = ""
    table_type: str = "table"  # table, view, materialized_view
    relationships: list[dict[str, str]] = field(default_factory=list)
    usage_frequency: float = 0.0  # Historical usage score
    schema_ddl: str = ""  # DDL definition
