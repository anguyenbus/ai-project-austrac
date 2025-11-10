"""
VisualizationAgent - Chart generation from query results using Strands framework.

Enhanced with patterns from chat2plot:
- Better schema validation with enums
- Data filtering and aggregation support
- Error recovery and retry logic
- Data transformation capabilities
- Migrated to use Strands framework for enhanced LLM-powered chart recommendations
"""

from typing import Any

import pandas as pd
from loguru import logger
from strands.agent import Agent
from strands.models.bedrock import BedrockModel

from src.agents.prompt_builders.output.visualizer import VisPromptBuilders
from src.agents.prompt_builders.prompts import Prompts
from src.utils.chart_generators import ChartGenerators
from src.utils.data_processors import DataProcessors
from src.utils.langfuse_client import langfuse_context, observe
from src.utils.strand_callback_helper import (
    create_usage_callback,
    update_langfuse_with_usage,
)
from src.utils.vis_types import ChartRecommendation, ChartType, SortOrder

from ..model_config import get_agent_config


class VisualizationConfig:
    """Configuration for VisualizationAgent."""

    def __init__(
        self,
        aws_region: str = None,
        model_id: str = None,
        enable_llm_recommendation: bool = True,
        max_rows_for_analysis: int = 100,
        max_retries: int = 2,
    ):
        """
        Initialize visualization configuration.

        Args:
            aws_region: AWS region for Bedrock service
            model_id: Model identifier for LLM
            enable_llm_recommendation: Enable LLM-based chart recommendations
            max_rows_for_analysis: Maximum data rows to analyze
            max_retries: Maximum retry attempts for chart generation

        """
        # Get configuration from centralized config
        config = get_agent_config("OutputVisualizer", aws_region)

        # Use provided values or fall back to config
        self.aws_region = aws_region or config["aws_region"]
        self.model_id = model_id or config["model_id"]
        self.enable_llm_recommendation = enable_llm_recommendation
        self.max_rows_for_analysis = max_rows_for_analysis
        self.max_retries = max_retries


class VisualizationAgent:
    """Strands-based visualization agent for chart generation."""

    def __init__(
        self,
        config: VisualizationConfig | None = None,
        model_id: str | None = None,
        aws_region: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        """
        Initialize VisualizationAgent with Strands Agent.

        Args:
            config: Visualization configuration object
            model_id: Model identifier for LLM
            aws_region: AWS region for Bedrock service
            debug_mode: Enable debug logging

        """
        self.config = config or VisualizationConfig()

        # Get configuration from centralized config
        agent_config = get_agent_config("OutputVisualizer", aws_region)

        # Use provided values, then config values, then defaults
        self.aws_region = (
            aws_region or self.config.aws_region or agent_config["aws_region"]
        )
        self.model_id = model_id or self.config.model_id or agent_config["model_id"]
        self.debug_mode = debug_mode

        # Build agent
        self.agent = self._build_agent(agent_config)

        logger.info("✓ VisualizationAgent initialized")

    def _build_agent(self, config: dict) -> Agent:
        """Build Strands Agent with usage tracking."""
        model = BedrockModel(
            model_id=self.model_id,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            streaming=False,
        )

        # Create callback for usage tracking
        callback, self._usage_container = create_usage_callback()

        # Use the cached system prompt directly
        base_instructions = Prompts.OUTPUT_VISUALIZER

        return Agent(
            model=model,
            system_prompt=base_instructions,
            callback_handler=callback,
        )

    def recommend_chart_with_llm(
        self,
        df: pd.DataFrame,
        query: str,
        retry_context: str | None = None,
        visual_hint: str | None = None,
    ) -> ChartRecommendation:
        """
        Use Strands Agent to recommend appropriate chart type and configuration.

        Args:
            df: Result dataframe
            query: Original user query
            retry_context: Error context for retry attempts
            visual_hint: Optional hint about desired visualization type

        Returns:
            LLM-generated chart recommendation

        """
        try:
            # Analyze dataframe
            analysis = DataProcessors.analyze_dataframe(df)

            # Build prompt using existing prompt builder
            prompt = VisPromptBuilders.build_visualization_prompt(
                df, query, analysis, retry_context, visual_hint
            )

            # Reset usage tracking
            self._usage_container["last_usage"] = None

            # Call Strands Agent with structured output
            llm_result = self.agent.structured_output(ChartRecommendation, prompt)

            # Update Langfuse with usage and costs
            update_langfuse_with_usage(
                self._usage_container,
                self.model_id,
                "VisualizationAgent",
                langfuse_context,
            )

            logger.info(
                f"Strands recommended {llm_result.chart_type} chart with confidence {llm_result.confidence}"
            )
            return llm_result

        except Exception as e:
            logger.error(f"Strands recommendation failed: {e}")
            raise

    def generate_plotly_json(
        self, df: pd.DataFrame, recommendation: ChartRecommendation
    ) -> dict[str, Any]:
        """
        Generate clean Plotly JSON description based on recommendation.

        Args:
            df: Result dataframe
            recommendation: Chart recommendation

        Returns:
            JSON chart description for Plotly/Dash

        """
        # Apply filters first
        if recommendation.filters:
            df = DataProcessors.filter_data(df, recommendation.filters)

        # Apply sorting if specified
        if recommendation.sort_order and recommendation.x_axis:
            df = df.sort_values(
                by=recommendation.x_axis.column,
                ascending=(recommendation.sort_order == SortOrder.ASC),
            )

        # Apply limit if specified
        if recommendation.limit:
            df = df.head(recommendation.limit)

        chart_json = {"data": [], "layout": {"title": recommendation.title}}

        # Add axis titles if provided
        if recommendation.x_axis and recommendation.x_axis.label:
            chart_json["layout"]["xaxis"] = {"title": recommendation.x_axis.label}
        if recommendation.y_axis and recommendation.y_axis.label:
            chart_json["layout"]["yaxis"] = {"title": recommendation.y_axis.label}

        # Generate data based on chart type
        if recommendation.chart_type == ChartType.BAR:
            ChartGenerators.generate_bar_chart(df, recommendation, chart_json)

            # Post-process horizontal bars to ensure correct x/y data orientation
            if recommendation.orientation == "h":
                for trace in chart_json["data"]:
                    if trace.get("type") == "bar":
                        # For horizontal bars, x should be numeric values, y should be categories
                        # Check if x contains strings (categories) - if so, swap with y
                        x_data = trace.get("x", [])
                        y_data = trace.get("y", [])

                        if x_data and isinstance(x_data[0], str):
                            # x contains categories, need to swap
                            trace["x"], trace["y"] = y_data, x_data

                        trace["orientation"] = "h"
        elif recommendation.chart_type == ChartType.LINE:
            ChartGenerators.generate_line_chart(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.SCATTER:
            ChartGenerators.generate_scatter_chart(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.PIE:
            ChartGenerators.generate_pie_chart(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.AREA:
            ChartGenerators.generate_area_chart(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.HISTOGRAM:
            ChartGenerators.generate_histogram(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.BOX:
            ChartGenerators.generate_box_plot(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.HEATMAP:
            ChartGenerators.generate_heatmap(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.TREEMAP:
            ChartGenerators.generate_treemap(df, recommendation, chart_json)
        elif recommendation.chart_type == ChartType.SCALAR:
            ChartGenerators.generate_scalar(df, recommendation, chart_json)

        return chart_json

    @observe(as_type="generation")
    def process(
        self,
        df: pd.DataFrame,
        query: str,
        normalized_query: Any | None = None,
    ) -> dict[str, Any]:
        """
        Process visualization generation with retry logic.

        Args:
            df: Result dataframe from SQL query
            query: Original user query
            normalized_query: Optional normalized query with filter hints

        Returns:
            Visualization result with JSON chart description

        """
        if df.empty:
            logger.warning("Empty dataframe provided for visualization")
            return {
                "success": False,
                "error": "No data available for visualization",
            }

        last_error = None

        visual_hint = None
        if normalized_query:
            if hasattr(normalized_query, "required_visuals"):
                visual_hint = normalized_query.required_visuals
            elif isinstance(normalized_query, dict):
                visual_hint = normalized_query.get("required_visuals")

        if isinstance(visual_hint, str):
            visual_hint = visual_hint.strip() or None

        for attempt in range(self.config.max_retries):
            try:
                # Get chart recommendation using Strands
                retry_context = str(last_error) if last_error else None
                recommendation = self.recommend_chart_with_llm(
                    df, query, retry_context, visual_hint
                )

                logger.info(
                    f"Attempt {attempt + 1}: {recommendation.chart_type} chart "
                    f"with confidence {recommendation.confidence}"
                )

                # Generate Plotly JSON
                chart_json = self.generate_plotly_json(df, recommendation)

                result = {
                    "success": True,
                    "chart": chart_json,
                    "metadata": {
                        "chart_type": recommendation.chart_type.value,
                        "confidence": recommendation.confidence,
                        "reasoning": recommendation.reasoning,
                        "data_rows": len(df),
                        "data_columns": list(df.columns),
                        "filters_applied": recommendation.filters,
                    },
                }

                if recommendation.insights:
                    result["metadata"]["insights"] = recommendation.insights

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if attempt == self.config.max_retries - 1:
                    logger.error(f"All attempts failed. Last error: {str(e)}")
                    return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}
