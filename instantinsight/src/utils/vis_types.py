"""Visualization type enums and models."""

from enum import Enum

from pydantic import BaseModel, Field, validator


class ChartType(str, Enum):
    """Supported chart types."""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SCALAR = "scalar"


class AggregationType(str, Enum):
    """Aggregation types for data."""

    SUM = "sum"
    AVG = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNTROWS = "countrows"
    DISTINCT_COUNT = "nunique"


class SortOrder(str, Enum):
    """Sort order options."""

    ASC = "asc"
    DESC = "desc"


class TimeUnit(str, Enum):
    """Time units for temporal data."""

    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    QUARTER = "quarter"
    DAY = "day"


class AxisConfig(BaseModel):
    """Configuration for chart axis."""

    column: str = Field(description="Column name for axis data")
    aggregation: AggregationType | None = Field(
        default=None, description="Aggregation to apply"
    )
    bin_size: int | None = Field(
        default=None, description="Number of bins for numeric discretization"
    )
    time_unit: TimeUnit | None = Field(
        default=None, description="Time unit for temporal discretization"
    )
    label: str | None = Field(default=None, description="Axis label")


class ChartRecommendation(BaseModel):
    """Enhanced LLM structured response for chart recommendation."""

    chart_type: ChartType = Field(description="Recommended chart type")

    # Data filtering
    filters: list[str] = Field(
        default_factory=list,
        description="List of filter conditions (e.g., 'column > 0')",
    )

    # Axes configuration
    x_axis: AxisConfig | None = Field(default=None, description="X-axis configuration")
    y_axis: AxisConfig | None = Field(default=None, description="Y-axis configuration")

    # Additional columns for multi-series
    y_columns: list[str] = Field(
        default_factory=list, description="Additional Y columns for multi-series charts"
    )

    # Pie chart specific
    labels_column: str | None = Field(
        default=None, description="Column for pie chart labels"
    )
    values_column: str | None = Field(
        default=None, description="Column for pie chart values"
    )

    # Heatmap specific
    z_column: str | None = Field(default=None, description="Column for heatmap values")

    # Styling
    title: str = Field(description="Chart title")
    color_column: str | None = Field(
        default=None, description="Column for color grouping"
    )
    stacked: bool = Field(default=False, description="Whether to stack bars/areas")
    orientation: str = Field(default="v", description="Chart orientation (v/h)")

    # Sorting
    sort_order: SortOrder | None = Field(
        default=None, description="Sort order for data"
    )
    limit: int | None = Field(default=None, description="Limit to top N items")

    # Metadata
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level")
    reasoning: str = Field(description="Explanation of chart selection")
    insights: str | None = Field(default=None, description="Key insights")

    @validator("chart_type", pre=True)
    def normalize_chart_type(cls, v):
        """Normalize chart type to enum value."""
        if isinstance(v, str):
            # Handle common aliases
            aliases = {
                "histogram": ChartType.HISTOGRAM,
                "hist": ChartType.HISTOGRAM,
                "bars": ChartType.BAR,
                "lines": ChartType.LINE,
                "dots": ChartType.SCATTER,
                "points": ChartType.SCATTER,
                "tree": ChartType.TREEMAP,
                "tree_map": ChartType.TREEMAP,
            }
            return aliases.get(v.lower(), v)
        return v
