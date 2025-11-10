"""Central registry for cached system prompts (agno agents)."""

from __future__ import annotations

from . import (
    column_mapper,
    output_visualizer,
    query_clarifier,
    query_modification_decider,
    schema_filter,
    schema_table_selector,
    sql_generator,
)

PROMPT_REGISTRY = {
    "QUERY_MODIFICATION_DECIDER": query_modification_decider.PROMPT,
    "SCHEMA_TABLE_SELECTOR": schema_table_selector.PROMPT,
    "COLUMN_MAPPER_MAPPING": column_mapper.MAPPING_SYSTEM_PROMPT,
    "COLUMN_MAPPER_CANDIDATE": column_mapper.CANDIDATE_SYSTEM_PROMPT,
    "OUTPUT_VISUALIZER": output_visualizer.PROMPT,
    "SCHEMA_FILTER_EXTRACTION": schema_filter.FILTER_EXTRACTION_SYSTEM_PROMPT,
    "SQL_GENERATOR": sql_generator.PROMPT,
    "QUERY_CLARIFIER": query_clarifier.CLARIFICATION_SYSTEM_PROMPT,
}


class Prompts:
    """Attribute-style access to cached system prompts for agno agents."""

    QUERY_MODIFICATION_DECIDER = PROMPT_REGISTRY["QUERY_MODIFICATION_DECIDER"]
    SCHEMA_TABLE_SELECTOR = PROMPT_REGISTRY["SCHEMA_TABLE_SELECTOR"]
    COLUMN_MAPPER_MAPPING = PROMPT_REGISTRY["COLUMN_MAPPER_MAPPING"]
    COLUMN_MAPPER_CANDIDATE = PROMPT_REGISTRY["COLUMN_MAPPER_CANDIDATE"]
    OUTPUT_VISUALIZER = PROMPT_REGISTRY["OUTPUT_VISUALIZER"]
    SCHEMA_FILTER_EXTRACTION = PROMPT_REGISTRY["SCHEMA_FILTER_EXTRACTION"]
    SQL_GENERATOR = PROMPT_REGISTRY["SQL_GENERATOR"]
    QUERY_CLARIFIER = PROMPT_REGISTRY["QUERY_CLARIFIER"]


__all__ = ["PROMPT_REGISTRY", "Prompts"]
