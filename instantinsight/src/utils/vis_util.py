"""Visualization utilities for rendering Plotly charts from schema."""

import json
from typing import Any


def render_plotly_chart(viz_schema: dict[str, Any], auto_open: bool = True) -> bool:
    """
    Render Plotly chart from schema to display interactive visualization.

    Args:
        viz_schema: Visualization schema containing chart data and layout
        auto_open: Whether to automatically open chart in browser

    Returns:
        True if chart was rendered successfully, False otherwise

    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo

        if not viz_schema or not viz_schema.get("chart"):
            print("No visualization data available")
            return False

        chart_data = viz_schema["chart"]

        # Create figure from schema
        fig = go.Figure(
            data=chart_data.get("data", []), layout=chart_data.get("layout", {})
        )

        # Display the chart
        pyo.plot(fig, auto_open=auto_open, filename="temp_chart.html")
        print("📊 Interactive chart opened in browser")
        return True

    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return False
    except Exception as e:
        print(f"Error rendering chart: {e}")
        return False


def display_chart_schema(viz_schema: dict[str, Any]) -> None:
    """
    Display visualization schema information and metadata.

    Args:
        viz_schema: Visualization schema containing metadata and chart data

    """
    if not viz_schema:
        return

    print("\n📊 Visualization Schema:")
    print("-" * 40)

    # Extract metadata
    if viz_schema.get("metadata"):
        metadata = viz_schema["metadata"]
        print(f"Chart Type: {metadata.get('chart_type', 'unknown')}")
        print(f"Confidence: {metadata.get('confidence', 0):.2%}")
        print(f"Reasoning: {metadata.get('reasoning', 'N/A')}")
        if metadata.get("insights"):
            print(f"Insights: {metadata.get('insights')}")
        print()

    # Pretty print the Plotly chart JSON
    if viz_schema.get("chart"):
        print("Plotly Chart JSON:")
        print(json.dumps(viz_schema["chart"], indent=2))
    elif viz_schema.get("error"):
        print(f"Visualization Error: {viz_schema['error']}")


def render_chart_from_schema(schema: dict[str, Any]) -> str | None:
    """
    Create HTML file from Plotly schema without opening browser.

    Args:
        schema: Plotly chart schema with data and layout

    Returns:
        Path to generated HTML file or None if failed

    """
    try:
        from pathlib import Path

        import plotly.graph_objects as go
        import plotly.offline as pyo

        if not schema or not schema.get("data"):
            return None

        # Create figure from schema
        fig = go.Figure(data=schema.get("data", []), layout=schema.get("layout", {}))

        # Generate HTML file
        html_path = Path("temp_chart.html")
        pyo.plot(fig, auto_open=False, filename=str(html_path))

        return str(html_path)

    except ImportError:
        print("Plotly not installed")
        return None
    except Exception as e:
        print(f"Error creating chart file: {e}")
        return None
