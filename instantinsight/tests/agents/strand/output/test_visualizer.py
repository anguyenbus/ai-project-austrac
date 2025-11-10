"""
Comprehensive tests for VisualizationAgent (Strands implementation).

Tests cover:
- Chart recommendation with LLM
- Plotly JSON generation
- Process method with retry logic
- Different chart types
- Error handling
- Wrapper compatibility
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.agents.strand_agents.output.visualizer import (
    VisualizationAgent,
    VisualizationConfig,
    VisualizationCore,
)
from src.utils.vis_types import AxisConfig, ChartRecommendation, ChartType

# ============================================================================
# UNIT TESTS - VisualizationCore
# ============================================================================


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_visualization_core_initialization(mock_agent_class):
    """Test VisualizationCore initializes correctly with Strands Agent."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    core = VisualizationCore()

    assert core.agent is not None
    assert core.model_id is not None
    assert core.aws_region is not None
    mock_agent_class.assert_called_once()


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_recommend_chart_with_llm_success(mock_agent_class):
    """Test successful chart recommendation."""
    # Setup mock
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Test Chart",
        confidence=0.95,
        reasoning="Bar chart is best for categorical data",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    # Create test data
    df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, 20, 30]})

    # Test
    core = VisualizationCore()
    result = core.recommend_chart_with_llm(df, "Show me sales by category")

    assert result is not None
    assert result.chart_type == ChartType.BAR
    assert result.confidence == 0.95
    assert result.title == "Test Chart"
    mock_agent_instance.structured_output.assert_called_once()


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_recommend_chart_with_llm_with_visual_hint(mock_agent_class):
    """Test chart recommendation with visual hint."""
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.LINE,
        x_axis=AxisConfig(column="date"),
        y_axis=AxisConfig(column="sales"),
        title="Sales Trend",
        confidence=0.9,
        reasoning="Line chart shows trends over time",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame(
        {"date": ["2024-01", "2024-02", "2024-03"], "sales": [100, 150, 200]}
    )

    core = VisualizationCore()
    result = core.recommend_chart_with_llm(
        df, "Show sales trend", visual_hint="line chart"
    )

    assert result.chart_type == ChartType.LINE


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_recommend_chart_with_llm_error_handling(mock_agent_class):
    """Test error handling in chart recommendation."""
    mock_agent_instance = Mock()
    mock_agent_instance.structured_output.side_effect = Exception("LLM error")
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})

    core = VisualizationCore()

    with pytest.raises(Exception, match="LLM error"):
        core.recommend_chart_with_llm(df, "Show data")


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_generate_plotly_json_bar_chart(mock_agent_class):
    """Test Plotly JSON generation for bar chart."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, 20, 30]})

    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category", label="Category"),
        y_axis=AxisConfig(column="value", label="Value"),
        title="Bar Chart Test",
        confidence=0.95,
        reasoning="Test",
    )

    core = VisualizationCore()
    result = core.generate_plotly_json(df, recommendation)

    assert result is not None
    assert "data" in result
    assert "layout" in result
    assert result["layout"]["title"] == "Bar Chart Test"


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_generate_plotly_json_with_filters(mock_agent_class):
    """Test Plotly JSON generation with data filters."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B", "C", "D"], "value": [10, 20, 30, 40]})

    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Filtered Chart",
        filters=["value > 15"],
        confidence=0.9,
        reasoning="Test",
    )

    core = VisualizationCore()
    # This will attempt to filter - ChartGenerators handles the actual filtering
    result = core.generate_plotly_json(df, recommendation)

    assert result is not None


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_generate_plotly_json_with_sorting(mock_agent_class):
    """Test Plotly JSON generation with sorting."""
    from src.utils.vis_types import SortOrder

    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["C", "A", "B"], "value": [30, 10, 20]})

    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Sorted Chart",
        sort_order=SortOrder.ASC,
        confidence=0.9,
        reasoning="Test",
    )

    core = VisualizationCore()
    result = core.generate_plotly_json(df, recommendation)

    assert result is not None


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_generate_plotly_json_with_limit(mock_agent_class):
    """Test Plotly JSON generation with row limit."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame(
        {"category": ["A", "B", "C", "D", "E"], "value": [10, 20, 30, 40, 50]}
    )

    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Limited Chart",
        limit=3,
        confidence=0.9,
        reasoning="Test",
    )

    core = VisualizationCore()
    result = core.generate_plotly_json(df, recommendation)

    assert result is not None


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_process_empty_dataframe(mock_agent_class):
    """Test process method with empty dataframe."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame()

    core = VisualizationCore()
    result = core.process(df, "Show data")

    assert result["success"] is False
    assert "error" in result
    assert "No data available" in result["error"]


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_process_success(mock_agent_class):
    """Test successful process execution."""
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Success Chart",
        confidence=0.95,
        reasoning="Test reasoning",
        insights="Test insights",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})

    core = VisualizationCore()
    result = core.process(df, "Show data")

    assert result["success"] is True
    assert "chart" in result
    assert "metadata" in result
    assert result["metadata"]["chart_type"] == "bar"
    assert result["metadata"]["confidence"] == 0.95
    assert result["metadata"]["insights"] == "Test insights"


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_process_with_retry(mock_agent_class):
    """Test process method with retry logic."""
    mock_agent_instance = Mock()

    # First attempt fails, second succeeds
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Retry Chart",
        confidence=0.9,
        reasoning="Test",
    )
    mock_agent_instance.structured_output.side_effect = [
        Exception("First attempt failed"),
        recommendation,
    ]
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})

    config = VisualizationConfig(max_retries=2)
    core = VisualizationCore(config=config)
    result = core.process(df, "Show data")

    # Should succeed on retry
    assert result["success"] is True
    assert mock_agent_instance.structured_output.call_count == 2


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_process_max_retries_exceeded(mock_agent_class):
    """Test process method when max retries exceeded."""
    mock_agent_instance = Mock()
    mock_agent_instance.structured_output.side_effect = Exception("Persistent error")
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})

    config = VisualizationConfig(max_retries=2)
    core = VisualizationCore(config=config)
    result = core.process(df, "Show data")

    assert result["success"] is False
    assert "error" in result
    assert mock_agent_instance.structured_output.call_count == 2


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_process_with_normalized_query_hint(mock_agent_class):
    """Test process method with normalized query visual hint."""
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.PIE,
        x_axis=AxisConfig(column="category"),
        y_axis=AxisConfig(column="value"),
        title="Pie Chart",
        confidence=0.9,
        reasoning="Test",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"category": ["A", "B", "C"], "value": [30, 40, 30]})

    normalized_query = Mock()
    normalized_query.required_visuals = "pie chart"

    core = VisualizationCore()
    result = core.process(df, "Show distribution", normalized_query=normalized_query)

    assert result["success"] is True


# ============================================================================
# WRAPPER COMPATIBILITY TESTS
# ============================================================================


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_visualization_agent_wrapper_initialization(mock_agent_class):
    """Test VisualizationAgent wrapper initializes correctly."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    agent = VisualizationAgent()

    assert agent.core is not None
    assert agent.config is not None


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_visualization_agent_recommend_delegates(mock_agent_class):
    """Test wrapper delegates recommend_chart_with_llm to core."""
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="x"),
        y_axis=AxisConfig(column="y"),
        title="Test",
        confidence=0.9,
        reasoning="Test",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})

    agent = VisualizationAgent()
    result = agent.recommend_chart_with_llm(df, "Show data")

    assert result == recommendation


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_visualization_agent_generate_plotly_delegates(mock_agent_class):
    """Test wrapper delegates generate_plotly_json to core."""
    mock_agent_instance = Mock()
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="x"),
        y_axis=AxisConfig(column="y"),
        title="Test",
        confidence=0.9,
        reasoning="Test",
    )

    agent = VisualizationAgent()
    result = agent.generate_plotly_json(df, recommendation)

    assert result is not None
    assert "data" in result
    assert "layout" in result


@patch("src.agents.strand_agents.output.visualizer.Agent")
def test_visualization_agent_process_delegates(mock_agent_class):
    """Test wrapper delegates process to core."""
    mock_agent_instance = Mock()
    recommendation = ChartRecommendation(
        chart_type=ChartType.BAR,
        x_axis=AxisConfig(column="x"),
        y_axis=AxisConfig(column="y"),
        title="Test",
        confidence=0.9,
        reasoning="Test",
    )
    mock_agent_instance.structured_output.return_value = recommendation
    mock_agent_class.return_value = mock_agent_instance

    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})

    agent = VisualizationAgent()
    result = agent.process(df, "Show data")

    assert result["success"] is True


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_visualization_config_defaults():
    """Test VisualizationConfig uses default values correctly."""
    config = VisualizationConfig()

    assert config.aws_region is not None
    assert config.model_id is not None
    assert config.enable_llm_recommendation is True
    assert config.max_rows_for_analysis == 100
    assert config.max_retries == 2


def test_visualization_config_custom_values():
    """Test VisualizationConfig accepts custom values."""
    config = VisualizationConfig(
        aws_region="us-west-2",
        model_id="custom-model",
        enable_llm_recommendation=False,
        max_rows_for_analysis=50,
        max_retries=3,
    )

    assert config.aws_region == "us-west-2"
    assert config.model_id == "custom-model"
    assert config.enable_llm_recommendation is False
    assert config.max_rows_for_analysis == 50
    assert config.max_retries == 3
