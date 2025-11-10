"""
Flexible visualization utilities for handling various JSON structures from LLMs.

Provides robust extraction, validation, and fallback generation for Plotly charts.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger


def extract_plotly_figure(
    viz_data: Any,
) -> tuple[go.Figure | None, dict[str, Any] | None]:
    """
    Extract Plotly figure from various JSON structures with fallback strategies.

    Args:
        viz_data: Visualization data in various possible formats

    Returns:
        Tuple of (figure, metadata) where figure is a Plotly Figure or None

    """
    if not viz_data:
        logger.debug("No visualization data provided")
        return None, None

    figure = None
    metadata = {}

    # Extract metadata first (if available)
    metadata = extract_metadata(viz_data)

    # Try extraction strategies in priority order
    extraction_strategies = [
        extract_from_chart_key,
        extract_from_figure_key,
        extract_direct_plotly,
        extract_from_nested_visualization,
        extract_from_success_format,
    ]

    for strategy in extraction_strategies:
        try:
            chart_data = strategy(viz_data)
            if chart_data and validate_plotly_schema(chart_data):
                figure = create_figure_from_data(chart_data)
                if figure:
                    logger.info(
                        f"Successfully extracted figure using {strategy.__name__}"
                    )
                    break
        except Exception as e:
            logger.debug(f"Strategy {strategy.__name__} failed: {e}")
            continue

    if not figure:
        logger.warning("Could not extract valid Plotly figure from any strategy")

    return figure, metadata


def extract_from_chart_key(viz_data: Any) -> dict | None:
    """Extract from viz_data['chart'] structure (VisualizationAgent format)."""
    if isinstance(viz_data, dict) and "chart" in viz_data:
        return viz_data["chart"]
    return None


def extract_from_figure_key(viz_data: Any) -> dict | None:
    """Extract from viz_data['figure'] structure."""
    if isinstance(viz_data, dict) and "figure" in viz_data:
        return viz_data["figure"]
    return None


def extract_direct_plotly(viz_data: Any) -> dict | None:
    """Check if viz_data is already a Plotly structure."""
    if isinstance(viz_data, dict) and "data" in viz_data:
        return viz_data
    return None


def extract_from_nested_visualization(viz_data: Any) -> dict | None:
    """Extract from nested visualization key."""
    if isinstance(viz_data, dict):
        if "visualization" in viz_data:
            viz = viz_data["visualization"]
            if isinstance(viz, dict):
                if "chart" in viz:
                    return viz["chart"]
                elif "data" in viz:
                    return viz
    return None


def extract_from_success_format(viz_data: Any) -> dict | None:
    """Extract from success/chart format (VisualizationAgent response)."""
    if isinstance(viz_data, dict) and viz_data.get("success"):
        if "chart" in viz_data:
            return viz_data["chart"]
    return None


def validate_plotly_schema(chart_data: dict) -> bool:
    """
    Validate if the data structure is compatible with Plotly.

    Args:
        chart_data: Dictionary that should contain Plotly chart data

    Returns:
        True if valid Plotly structure, False otherwise

    """
    if not isinstance(chart_data, dict):
        return False

    # Must have 'data' key at minimum
    if "data" not in chart_data:
        return False

    # Data should be a list
    if not isinstance(chart_data["data"], list):
        # Try to wrap single trace in list
        if isinstance(chart_data["data"], dict):
            chart_data["data"] = [chart_data["data"]]
        else:
            return False

    # Basic validation of traces
    for trace in chart_data["data"]:
        if not isinstance(trace, dict):
            return False
        # Should have some data fields (x, y, values, etc.)
        data_fields = {"x", "y", "values", "z", "lat", "lon", "locations"}
        if not any(field in trace for field in data_fields):
            # Allow empty traces for certain types
            if "type" not in trace:
                return False

    return True


def create_figure_from_data(chart_data: dict) -> go.Figure | None:
    """
    Create a Plotly Figure from validated chart data.

    Args:
        chart_data: Validated Plotly chart data

    Returns:
        Plotly Figure or None if creation fails

    """
    try:
        # Ensure data is properly formatted
        data = chart_data.get("data", [])

        # Fix common issues with trace types
        if isinstance(data, list):
            for trace in data:
                if isinstance(trace, dict) and "type" in trace:
                    # Map common variations to valid Plotly types
                    trace_type = trace["type"].lower()
                    type_mapping = {
                        "line": "scatter",  # Line charts are 'scatter' with mode='lines'
                        "point": "scatter",
                        "points": "scatter",
                    }

                    if trace_type in type_mapping:
                        trace["type"] = type_mapping[trace_type]
                        # For line charts, set the mode if not specified
                        if trace_type == "line" and "mode" not in trace:
                            trace["mode"] = "lines"

        figure = go.Figure(data=data, layout=chart_data.get("layout", {}))

        # Add frames if present (for animations)
        if "frames" in chart_data:
            figure.frames = chart_data["frames"]

        return figure
    except Exception as e:
        logger.error(f"Failed to create Plotly figure: {e}")
        return None


def extract_metadata(viz_data: Any) -> dict[str, Any]:
    """
    Extract metadata from visualization data.

    Args:
        viz_data: Visualization data that may contain metadata

    Returns:
        Dictionary of metadata

    """
    metadata = {}

    if not isinstance(viz_data, dict):
        return metadata

    # Direct metadata key
    if "metadata" in viz_data:
        metadata.update(viz_data["metadata"])

    # Common metadata fields that might be at root level
    metadata_fields = [
        "chart_type",
        "confidence",
        "reasoning",
        "insights",
        "title",
        "description",
        "data_rows",
        "data_columns",
        "filters_applied",
        "aggregations",
    ]

    for field in metadata_fields:
        if field in viz_data and field not in metadata:
            metadata[field] = viz_data[field]

    return metadata


def auto_generate_fallback(df: pd.DataFrame, query: str = "") -> go.Figure:
    """
    Auto-generate a basic visualization based on DataFrame characteristics.

    Args:
        df: DataFrame to visualize
        query: Original query for context

    Returns:
        Basic Plotly figure

    """
    if df.empty:
        return create_empty_figure("No data to visualize")

    # Analyze DataFrame to determine best chart type
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # If we have time-like column, prefer line chart
    time_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if time_cols and numeric_cols:
        fig = px.line(
            df, x=time_cols[0], y=numeric_cols[0], title="Time Series Visualization"
        )
        return fig

    # If we have categorical and numeric, use bar chart
    if categorical_cols and numeric_cols:
        # Limit categories for readability
        cat_col = categorical_cols[0]
        if df[cat_col].nunique() <= 20:
            fig = px.bar(
                df, x=cat_col, y=numeric_cols[0], title="Categorical Distribution"
            )
            return fig

    # If only numeric columns, use scatter plot
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
        return fig

    # Single numeric column - histogram
    if len(numeric_cols) == 1:
        fig = px.histogram(df, x=numeric_cols[0], title="Distribution")
        return fig

    # Last resort - show first few columns as table-like bar chart
    if len(df) <= 20:
        fig = create_table_figure(df)
        return fig

    return create_empty_figure("Could not determine visualization type")


def create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white"
    )
    return fig


def create_table_figure(df: pd.DataFrame) -> go.Figure:
    """Create a simple table visualization for small datasets."""
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title="Data Table")
    return fig


def enhance_figure_with_metadata(
    figure: go.Figure, metadata: dict[str, Any]
) -> go.Figure:
    """
    Enhance a Plotly figure with metadata information.

    Args:
        figure: Plotly figure to enhance
        metadata: Metadata to add to the figure

    Returns:
        Enhanced figure

    """
    if not metadata:
        return figure

    # Add title if provided
    if "title" in metadata and not figure.layout.title.text:
        figure.update_layout(title=metadata["title"])

    # Add insights as annotations if available
    if "insights" in metadata and isinstance(metadata["insights"], list):
        # Add first insight as subtitle
        if metadata["insights"]:
            figure.update_layout(
                annotations=[
                    dict(
                        text=metadata["insights"][0],
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=1.05,
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=10, color="gray"),
                    )
                ]
            )

    # Store metadata in figure for later reference
    figure.update_layout(meta=metadata)

    return figure
