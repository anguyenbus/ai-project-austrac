"""
Handler for visualization modification requests.

Separates modification logic from UI components.
"""

import copy
import json
from typing import Any

import plotly.graph_objects as go

# NOTE: Agno imports removed after migration to Strand
# from agno.db.in_memory.in_memory_db import InMemoryDb
# from agno.memory import MemoryManager
# from agno.models.message import Message
from loguru import logger

from src.agents.strand_agents.output.visualizer import VisualizationAgent
from src.agents.strand_agents.query.modification_decider import (
    ModificationDecisionAgent,
)
from src.rag.pipeline import Pipeline

from .visualization_utils import extract_plotly_figure


class ModificationHandler:
    """Handles all visualization modification requests."""

    def __init__(self, pipeline: Pipeline | None = None):
        """Initialize the modification handler."""
        self.pipeline = pipeline
        self.decision_agent = ModificationDecisionAgent()
        self.visualization_agent = VisualizationAgent()

        # NOTE: Agno memory management removed after migration to Strand
        # self.memory_db = InMemoryDb()
        # self.memory_manager = MemoryManager(
        #     model=self.decision_agent.core.model, db=self.memory_db
        # )

        # self._attach_memory(self.decision_agent.core.agent)
        # self._attach_memory(self.visualization_agent.core.agent)

        self._memory_offsets: dict[str, int] = {}
        self.max_trace_points = 40

    # NOTE: Agno memory management removed after migration to Strand
    # def _attach_memory(self, agent) -> None:
    #     """Attach shared memory capabilities to an agno agent."""
    #     agent.db = self.memory_db
    #     agent.memory_manager = self.memory_manager
    #     agent.enable_user_memories = True
    #     agent.add_memories_to_context = True
    #     agent.add_history_to_context = True

    def _build_compact_schema(
        self,
        schema: Any,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}

        chart_dict = self._extract_chart_dict(schema)
        compact_chart: dict[str, Any] = {}
        compact_chart["data"] = [
            self._summarize_trace(trace)
            for trace in chart_dict.get("data", [])
            if isinstance(trace, dict)
        ]

        layout_summary = self._summarize_layout(chart_dict.get("layout"))
        if layout_summary:
            compact_chart["layout"] = layout_summary

        compact: dict[str, Any] = {"chart": compact_chart}
        collected_metadata = self._collect_metadata(metadata, context)
        if collected_metadata:
            compact["metadata"] = collected_metadata

        if isinstance(schema.get("success"), bool):
            compact["success"] = schema["success"]

        return compact

    def _extract_chart_dict(self, schema: dict[str, Any]) -> dict[str, Any]:
        chart = schema.get("chart")
        if isinstance(chart, dict):
            return chart
        return schema

    def _summarize_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        for key in ("type", "name", "mode", "orientation", "stackgroup", "groupnorm"):
            value = trace.get(key)
            if value is not None:
                summary[key] = value

        for key in ("x", "y", "labels", "values"):
            if key in trace:
                summary[key] = self._truncate_sequence(trace[key])

        if "z" in trace:
            summary["z_preview"] = self._truncate_matrix(trace["z"])

        marker_summary = self._summarize_marker(trace.get("marker"))
        if marker_summary:
            summary["marker"] = marker_summary

        return summary

    def _summarize_marker(self, marker: Any) -> dict[str, Any]:
        if isinstance(marker, dict):
            summary: dict[str, Any] = {}
            color = marker.get("color")
            if isinstance(color, str):
                summary["color"] = color
            elif isinstance(color, list | tuple):
                summary["color_preview"] = self._truncate_sequence(color)

            size = marker.get("size")
            if isinstance(size, int | float | str):
                summary["size"] = size

            return summary

        if isinstance(marker, str):
            return {"color": marker}

        return {}

    def _summarize_layout(self, layout: Any) -> dict[str, Any]:
        if not isinstance(layout, dict):
            return {}

        summary: dict[str, Any] = {}
        title_text = self._extract_text(layout.get("title"))
        if title_text:
            summary["title"] = title_text

        xaxis_summary = self._summarize_axis(layout.get("xaxis"))
        if xaxis_summary:
            summary["xaxis"] = xaxis_summary

        yaxis_summary = self._summarize_axis(layout.get("yaxis"))
        if yaxis_summary:
            summary["yaxis"] = yaxis_summary

        for key in ("barmode", "barnorm", "hovermode"):
            value = layout.get(key)
            if value:
                summary[key] = value

        legend_summary = self._summarize_legend(layout.get("legend"))
        if legend_summary:
            summary["legend"] = legend_summary

        return summary

    def _summarize_axis(self, axis: Any) -> dict[str, Any]:
        if not isinstance(axis, dict):
            return {}

        summary: dict[str, Any] = {}
        title_text = self._extract_text(axis.get("title"))
        if title_text:
            summary["title"] = title_text

        for key in ("type", "categoryorder", "tickformat", "tickmode"):
            value = axis.get(key)
            if value:
                summary[key] = value

        return summary

    def _summarize_legend(self, legend: Any) -> dict[str, Any]:
        if not isinstance(legend, dict):
            return {}

        summary: dict[str, Any] = {}
        orientation = legend.get("orientation")
        if orientation:
            summary["orientation"] = orientation

        title_text = self._extract_text(legend.get("title"))
        if title_text:
            summary["title"] = title_text

        return summary

    def _collect_metadata(
        self,
        metadata: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        collected: dict[str, Any] = {}

        for source in (context, metadata):
            if not isinstance(source, dict):
                continue

            for key in (
                "chart_type",
                "confidence",
                "data_columns",
                "data_rows",
                "filters_applied",
                "insights",
                "reasoning",
            ):
                value = source.get(key)
                if value not in (None, ""):
                    collected[key] = value

        return collected

    def _extract_text(self, value: Any) -> str | None:
        if isinstance(value, dict):
            for candidate in ("text", "title"):
                text = value.get(candidate)
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped:
                        return stripped
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        return None

    def _truncate_sequence(self, values: Any) -> Any:
        if not isinstance(values, list | tuple):
            return values

        total = len(values)
        truncated = list(values[: self.max_trace_points])
        if total > self.max_trace_points:
            truncated.append(f"... ({total - self.max_trace_points} more)")
        return truncated

    def _truncate_matrix(self, matrix: Any) -> Any:
        if not isinstance(matrix, list | tuple):
            return matrix

        row_limit = min(len(matrix), self.max_trace_points)
        rows = [self._truncate_sequence(row) for row in matrix[:row_limit]]
        if len(matrix) > self.max_trace_points:
            rows.append(f"... ({len(matrix) - self.max_trace_points} more rows)")
        return rows

    def process_modification_request(
        self,
        user_message: str,
        current_viz_data: dict,
        viz_id: str,
        conversation_history: list,
    ) -> dict:
        """
        Process user modification request and return result.

        Returns:
            Dict with keys:
                - success: bool
                - message: str (user-facing message)
                - updated_figure: Dict (if visualization was updated)
                - updated_context: Dict (if context was updated)
                - updated_data: DataFrame (if data was updated)

        """
        try:
            context = current_viz_data.get("context", {})
            historical_context = self._get_historical_context(viz_id, context)

            # Get current SQL and extract Plotly schema using flexible extractor
            current_sql = context.get("sql_query", "")
            raw_plotly_schema = context.get("plotly_schema", {})

            # Extract the actual Plotly chart schema using our flexible extractor
            figure, metadata = extract_plotly_figure(raw_plotly_schema)

            if figure and hasattr(figure, "to_dict"):
                current_plotly_schema = figure.to_dict()
            else:
                current_plotly_schema = copy.deepcopy(raw_plotly_schema)

            if not isinstance(current_plotly_schema, dict):
                current_plotly_schema = {}

            context["plotly_schema"] = current_plotly_schema

            compact_plotly_schema = self._build_compact_schema(
                current_plotly_schema,
                metadata=current_viz_data.get("metadata"),
                context=context,
            )

            schema_json = json.dumps(compact_plotly_schema)
            schema_size = len(schema_json)
            estimated_tokens = schema_size // 4

            if estimated_tokens > 2000:
                logger.warning(
                    f"Compact schema still large: {schema_size} chars (~{estimated_tokens} tokens)"
                )
                self.max_trace_points = max(10, self.max_trace_points // 2)
                compact_plotly_schema = self._build_compact_schema(
                    current_plotly_schema,
                    metadata=current_viz_data.get("metadata"),
                    context=context,
                )
                logger.info(
                    f"Re-compacted with max_trace_points={self.max_trace_points}"
                )

            logger.info(f"Analyzing modification request: {user_message}")

            # Use ModificationDecisionAgent to decide processing approach
            decision_result = self.decision_agent.decide_processing_approach(
                historical_context=historical_context,
                user_message=user_message,
                current_sql=current_sql,
                current_plotly_schema=compact_plotly_schema,
                user_id=viz_id,
            )

            if not decision_result.get("success"):
                return {
                    "success": False,
                    "message": f"❌ Decision analysis failed: {decision_result.get('error', 'Unknown error')}",
                }

            processing_method = decision_result.get("processing_method")
            logger.info(
                f"Decision: {processing_method} (confidence: {decision_result.get('confidence', 0)})"
            )

            # Route to appropriate handler
            if processing_method == "simple_visual_modification":
                return self._handle_simple_visual(
                    decision_result, current_viz_data, user_message
                )
            elif processing_method == "visualization_agent_only":
                return self._handle_schema_only(
                    decision_result, current_viz_data, user_message
                )
            elif processing_method == "simple_sql_modification":
                return self._handle_simple_sql(
                    decision_result, current_viz_data, user_message
                )
            else:
                return self._handle_full_pipeline(
                    decision_result,
                    current_viz_data,
                    user_message,
                    historical_context,
                    current_sql,
                    compact_plotly_schema,
                )

        except Exception as e:
            logger.error(f"Error processing modification request: {e}")
            return {"success": False, "message": f"❌ Error: {str(e)}"}

    def _summarize_conversation_turn(self, entry: dict) -> str:
        """Summarize a conversation entry to reduce token usage in memory."""
        content = entry.get("message", "")
        role = entry.get("type", "user")

        if not content:
            return ""

        if role == "user":
            if len(content) <= 100:
                return content

            if len(content) <= 150:
                return content

            return content[:150] + "..."

        else:
            if "visualization" in content.lower() or "created" in content.lower():
                return "Created visualization"
            elif "updated" in content.lower() or "modified" in content.lower():
                return "Updated visualization"
            elif "error" in content.lower() or "❌" in content:
                return "Error occurred"
            elif len(content) <= 80:
                return content
            else:
                return content[:80] + "..."

    def sync_history(self, viz_id: str, conversation_history: list[dict]) -> None:
        """Persist conversation turns to shared memory for later retrieval."""
        if not viz_id:
            return

        last_index = self._memory_offsets.get(viz_id, 0)
        new_entries = conversation_history[last_index:]

        if not new_entries:
            return

        # NOTE: Agno Message class removed after migration to Strand
        # messages: list[Message] = []
        for entry in new_entries:
            summary = self._summarize_conversation_turn(entry)
            if not summary:
                continue

            # NOTE: Agno Message class removed after migration to Strand
            # role = entry.get("type", "user")
            # role_value = "assistant" if role == "assistant" else "user"
            # messages.append(Message(role=role_value, content=summary))
            pass

        # NOTE: Agno memory management removed after migration to Strand
        # if messages:
        #     try:
        #         self.memory_manager.create_user_memories(
        #             messages=messages,
        #             user_id=viz_id,
        #             agent_id=self.decision_agent.core.agent.id,
        #         )
        #     except Exception as exc:
        #         logger.warning(f"Failed to sync conversation history: {exc}")

        self._memory_offsets[viz_id] = len(conversation_history)

    def _get_historical_context(self, viz_id: str, context: dict) -> str:
        """Gather prior messages from memory to inform modification prompts."""
        # NOTE: Agno memory management removed after migration to Strand
        # Memory-based context retrieval is disabled
        # if viz_id:
        #     try:
        #         memories = self.memory_manager.search_user_memories(
        #             user_id=viz_id, retrieval_method="last_n", limit=6
        #         )
        #         snippets = [
        #             memory.memory.strip()
        #             for memory in memories
        #             if getattr(memory, "memory", "").strip()
        #         ]
        #         if snippets:
        #             historical = "\n".join(snippets)
        #             context_tokens = len(historical) // 4
        #
        #             if context_tokens > 500:
        #                 logger.warning(
        #                     f"Historical context large: {len(historical)} chars (~{context_tokens} tokens), truncating"
        #                 )
        #                 while snippets and len("\n".join(snippets)) // 4 > 500:
        #                     snippets.pop(0)
        #                 historical = "\n".join(snippets) if snippets else ""
        #
        #             return historical
        #     except Exception as exc:
        #         logger.warning(f"Memory lookup failed for {viz_id}: {exc}")

        original_question = context.get("original_question", "").strip()
        if original_question:
            return f"Original question: {original_question}"
        return ""

    def _filter_trace_properties(self, trace_update: dict) -> dict:
        """
        Filter trace properties to remove invalid ones based on trace type.

        Args:
            trace_update: Dictionary of trace properties to update

        Returns:
            Filtered dictionary with only valid properties

        """
        # Get the trace type
        trace_type = trace_update.get("type", "").lower()

        # Define valid properties for each trace type
        trace_type_properties = {
            "pie": {
                "labels",
                "values",
                "type",
                "name",
                "textinfo",
                "textposition",
                "hole",
                "rotation",
                "direction",
                "sort",
                "marker",
                "pull",
            },
            "bar": {
                "x",
                "y",
                "type",
                "name",
                "orientation",
                "marker",
                "text",
                "textposition",
                "width",
                "offset",
            },
            "scatter": {"x", "y", "type", "mode", "name", "marker", "line", "text"},
            "line": {"x", "y", "type", "mode", "name", "line", "marker"},
            "treemap": {
                "labels",
                "parents",
                "values",
                "type",
                "name",
                "textinfo",
                "textposition",
                "textfont",
                "marker",
                "branchvalues",
                "maxdepth",
            },
            "heatmap": {
                "x",
                "y",
                "z",
                "type",
                "name",
                "colorscale",
                "colorbar",
                "text",
                "texttemplate",
                "textfont",
                "showscale",
                "zmin",
                "zmax",
            },
        }

        # If we don't know the trace type or it's not in our list, allow all properties
        if trace_type not in trace_type_properties:
            return trace_update

        # Filter properties
        valid_properties = trace_type_properties[trace_type]
        filtered_update = {
            key: value for key, value in trace_update.items() if key in valid_properties
        }

        # Log if we filtered out any properties
        removed_properties = set(trace_update.keys()) - set(filtered_update.keys())
        if removed_properties:
            logger.debug(
                f"Filtered out invalid properties for {trace_type}: {removed_properties}"
            )

        return filtered_update

    def _is_chart_type_change(self, modified_config: dict) -> bool:
        """
        Check if the modification involves changing chart type.

        Args:
            modified_config: Plotly configuration changes

        Returns:
            True if this is a chart type change

        """
        # Check if data array contains type changes
        if "data" in modified_config and isinstance(modified_config["data"], list):
            for trace_update in modified_config["data"]:
                if isinstance(trace_update, dict) and "type" in trace_update:
                    return True
        return False

    def _handle_simple_visual(
        self, decision_result: dict, viz_data: dict, user_message: str
    ) -> dict:
        """Handle simple visual modifications by directly updating Plotly config."""
        try:
            modified_config = decision_result.get("modified_plotly_config")
            if not modified_config:
                return {
                    "success": False,
                    "message": "❌ No visual configuration provided for modification.",
                }

            # Check if this is actually a chart type change (should be schema-only)
            if self._is_chart_type_change(modified_config):
                logger.info(
                    "Detected chart type change, routing to schema-only handler"
                )
                # Create a new decision result for schema-only handling
                schema_decision = {
                    "modify_schema": user_message,
                    "explanation": "Chart type change detected, using VisualizationAgent",
                }
                return self._handle_schema_only(schema_decision, viz_data, user_message)

            # Get current figure and ensure it's in dict format
            current_figure_obj = viz_data.get("figure")
            if isinstance(current_figure_obj, go.Figure):
                current_figure = current_figure_obj.to_dict()
            else:
                current_figure = copy.deepcopy(current_figure_obj or {})

            # Apply the modifications directly
            for key, value in modified_config.items():
                if key == "layout":
                    if "layout" not in current_figure:
                        current_figure["layout"] = {}
                    current_figure["layout"].update(value)
                elif key == "data" and isinstance(value, list):
                    # Update trace properties with validation
                    for i, trace_update in enumerate(value):
                        if i < len(current_figure.get("data", [])) and trace_update:
                            # Filter out invalid properties based on trace type
                            filtered_update = self._filter_trace_properties(
                                trace_update
                            )
                            current_figure["data"][i].update(filtered_update)
                else:
                    current_figure[key] = value

            # Convert back to Plotly Figure object
            updated_figure = go.Figure(current_figure)

            # Update context
            context = viz_data.get("context", {})
            context["user_modification"] = user_message
            context["plotly_schema"] = current_figure

            return {
                "success": True,
                "message": f"✅ Visual update applied. {decision_result.get('explanation', '')}",
                "updated_figure": updated_figure,
                "updated_context": context,
            }

        except Exception as e:
            logger.error(f"Simple visual modification failed: {e}")
            return {"success": False, "message": f"❌ Visual update error: {str(e)}"}

    def _handle_schema_only(
        self, decision_result: dict, viz_data: dict, user_message: str
    ) -> dict:
        """Handle schema-only modifications using VisualizationAgent."""
        try:
            if not self.visualization_agent:
                return {
                    "success": False,
                    "message": "❌ VisualizationAgent not available.",
                }

            current_data = viz_data.get("data")
            if current_data is None or current_data.empty:
                return {
                    "success": False,
                    "message": "❌ No data available for visualization modification.",
                }

            # Create modification query for VisualizationAgent
            modification_instruction = decision_result.get(
                "modify_schema", user_message
            )
            viz_query = f"Create visualization based on user request: {modification_instruction}"

            # Process with VisualizationAgent
            viz_result = self.visualization_agent.process(current_data, viz_query)

            if viz_result.get("success"):
                new_chart = viz_result.get("chart")
                if new_chart:
                    # Use flexible extractor to create proper Plotly figure
                    updated_figure, viz_metadata = extract_plotly_figure(viz_result)

                    if not updated_figure:
                        # Fallback: try to create figure from chart directly
                        try:
                            updated_figure = go.Figure(
                                data=new_chart.get("data", []),
                                layout=new_chart.get("layout", {}),
                            )
                        except Exception as e:
                            logger.error(f"Failed to create figure from chart: {e}")
                            return {
                                "success": False,
                                "message": f"❌ Could not create visualization from generated schema: {str(e)}",
                            }

                    context = viz_data.get("context", {})
                    new_context = {
                        "sql_query": context.get("sql_query", ""),
                        "original_question": context.get("original_question", ""),
                        "user_modification": user_message,
                        "plotly_schema": viz_result,  # Store the full result
                        "data_shape": context.get("data_shape", ""),
                    }

                    # Add metadata if available
                    if viz_metadata:
                        new_context.update(
                            {
                                "chart_type": viz_metadata.get("chart_type"),
                                "confidence": viz_metadata.get("confidence"),
                                "insights": viz_metadata.get("insights", []),
                            }
                        )

                    chart_type = viz_result.get("metadata", {}).get(
                        "chart_type", "visualization"
                    )
                    return {
                        "success": True,
                        "message": f"✅ Updated to {chart_type} (schema-only). {decision_result.get('explanation', '')}",
                        "updated_figure": updated_figure,
                        "updated_context": new_context,
                    }
                else:
                    return {
                        "success": False,
                        "message": f"⚠️ Could not generate new visualization: {viz_result.get('error', 'Unknown error')}",
                    }
            else:
                return {
                    "success": False,
                    "message": f"❌ Visualization modification failed: {viz_result.get('error', 'Unknown error')}",
                }

        except Exception as e:
            logger.error(f"Schema-only modification failed: {e}")
            return {
                "success": False,
                "message": f"❌ Schema modification error: {str(e)}",
            }

    def _handle_simple_sql(
        self, decision_result: dict, viz_data: dict, user_message: str
    ) -> dict:
        """Handle simple SQL modifications directly without full pipeline."""
        try:
            modified_sql = decision_result.get("modified_sql")
            if not modified_sql:
                # Fallback to full pipeline if no modified SQL provided
                return {
                    "success": False,
                    "message": "❌ No modified SQL provided for simple modification.",
                }

            if not self.pipeline or not self.pipeline.rag_engine:
                return {
                    "success": False,
                    "message": "❌ Database connection not available.",
                }

            # Execute the modified SQL directly
            logger.info(f"Executing modified SQL: {modified_sql[:100]}...")
            new_data = self.pipeline.rag_engine.execute_sql(modified_sql)

            if new_data is None or new_data.empty:
                return {
                    "success": False,
                    "message": "❌ Modified query returned no data.",
                }

            # Generate new visualization with the new data
            if self.visualization_agent:
                viz_query = f"Create visualization for: {user_message}"
                viz_result = self.visualization_agent.process(new_data, viz_query)

                if viz_result.get("success"):
                    new_chart = viz_result.get("chart")
                    if new_chart:
                        context = viz_data.get("context", {})
                        new_context = {
                            "sql_query": modified_sql,
                            "original_question": context.get("original_question", ""),
                            "user_modification": user_message,
                            "plotly_schema": new_chart,
                            "data_shape": f"{len(new_data)} rows × {len(new_data.columns)} columns",
                        }

                        return {
                            "success": True,
                            "message": f"✅ Quick update completed. {decision_result.get('explanation', '')}",
                            "updated_figure": new_chart,
                            "updated_context": new_context,
                            "updated_data": new_data,
                        }

            return {
                "success": False,
                "message": "⚠️ Could not generate visualization for modified data.",
            }

        except Exception as e:
            logger.error(f"Simple SQL modification failed: {e}")
            return {"success": False, "message": f"❌ Error: {str(e)}"}

    def _handle_full_pipeline(
        self,
        decision_result: dict,
        viz_data: dict,
        user_message: str,
        historical_context: str,
        current_sql: str,
        current_plotly_schema: dict,
    ) -> dict:
        """Handle modifications requiring full Pipeline processing."""
        try:
            if not self.pipeline:
                return {
                    "success": False,
                    "message": "❌ Pipeline not available for processing SQL modifications.",
                }

            # Build the complete context for Pipeline
            context_parts = []
            context_parts.append(f"Historical context:\n{historical_context}")

            if current_sql:
                context_parts.append(f"Current SQL query:\n{current_sql}")

            if current_plotly_schema:
                schema_str = json.dumps(current_plotly_schema, indent=2)
                context_parts.append(f"Current Plotly schema:\n{schema_str}")

            context_parts.append(f"User modification request: {user_message}")
            full_query = "\n\n".join(context_parts)

            logger.info(f"Processing with full pipeline: {user_message}")

            # Process through Pipeline
            pipeline_result = self.pipeline.process(
                query=full_query, generate_visualization=True
            )

            if pipeline_result.success:
                if pipeline_result.visualization:
                    viz_result = pipeline_result.visualization
                    new_figure = viz_result.get("chart") or viz_result.get("figure")

                    if new_figure:
                        execution_stage = pipeline_result.stages.get("QUERY_EXECUTION")
                        new_data = (
                            execution_stage.data
                            if execution_stage and execution_stage.success
                            else viz_data.get("data")
                        )

                        context = viz_data.get("context", {})
                        new_context = {
                            "sql_query": pipeline_result.sql,
                            "original_question": context.get("original_question", ""),
                            "user_modification": user_message,
                            "plotly_schema": viz_result.get("chart", {}),
                            "data_shape": f"{len(new_data)} rows × {len(new_data.columns)} columns"
                            if new_data is not None and hasattr(new_data, "shape")
                            else "N/A",
                        }

                        chart_type = viz_result.get("metadata", {}).get(
                            "chart_type", "visualization"
                        )
                        return {
                            "success": True,
                            "message": f"✅ Updated to {chart_type} (full pipeline). {decision_result.get('explanation', '')}",
                            "updated_figure": new_figure,
                            "updated_context": new_context,
                            "updated_data": new_data,
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"⚠️ Generated SQL but could not create visualization: {pipeline_result.sql[:100] if pipeline_result.sql else 'No SQL'}...",
                        }
                else:
                    if pipeline_result.sql:
                        return {
                            "success": False,
                            "message": f"✅ Generated new SQL but no visualization was created: {pipeline_result.sql[:100]}...",
                        }
                    else:
                        return {
                            "success": False,
                            "message": "⚠️ Pipeline succeeded but no SQL or visualization was generated.",
                        }
            else:
                error_msg = pipeline_result.error or "Unknown error"
                return {
                    "success": False,
                    "message": f"❌ Could not process modification: {error_msg}",
                }

        except Exception as e:
            logger.error(f"Full pipeline modification failed: {e}")
            return {"success": False, "message": f"❌ Pipeline error: {str(e)}"}
