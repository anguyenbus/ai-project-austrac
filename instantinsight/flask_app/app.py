"""Flask application for Natural Language to SQL/Visualization."""

import datetime
import json
import sys
from pathlib import Path

import plotly.utils
from flask import Flask, jsonify, render_template, request, session
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from src.rag import RAGEngine
from src.rag.pipeline import Pipeline, Stage

from .modification_handler import ModificationHandler
from .visualization_utils import (
    auto_generate_fallback,
    enhance_figure_with_metadata,
    extract_plotly_figure,
)

app = Flask(__name__)
app.secret_key = "instantinsight-secret-key-2024"

# Initialize RAG components
rag_engine, _ = RAGEngine.create_instance()
pipeline = Pipeline(rag_instance=rag_engine)
modification_handler = ModificationHandler(pipeline=pipeline)

# Store visualizations and conversation history
visualizations = {}
conversation_histories = {}
chart_counter = 0


@app.route("/")
def index():
    """Render main application page."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Process natural language queries and generate SQL with visualizations."""
    try:
        data = request.json
        message = data.get("message", "")

        if not message:
            return jsonify({"success": False, "error": "Empty message"})

        logger.info(f"Processing message: {message}")

        # Process with pipeline
        pipeline_result = pipeline.process(message)

        response = {"success": False}

        # Check if we have a clarification message in the SQL field (starts with SQL comments)
        if pipeline_result.sql and pipeline_result.sql.strip().startswith("--"):
            # This is a clarification message formatted as SQL comments
            clarification_text = pipeline_result.sql

            # Remove SQL comment markers to get the actual message
            lines = clarification_text.strip().split("\n")
            clean_lines = []
            for line in lines:
                if line.strip().startswith("--"):
                    clean_lines.append(line.strip()[2:].strip())
                else:
                    clean_lines.append(line.strip())

            main_message = " ".join(clean_lines)

            # Parse the message for context
            response["clarification"] = main_message
            response["context_type"] = "vague_query"
            response["reasoning"] = "Query needs more specificity"

            # Extract suggestions from the message - use domain-specific examples
            suggestions = []
            if (
                "Could you specify:" in main_message
                or "For example:" in main_message
                or "examples:" in main_message.lower()
            ):
                suggestions.append(
                    "What is the total quantity of products sold by category?"
                )
                suggestions.append(
                    "What is the total value of products in stock for each category?"
                )
            response["suggestions"] = suggestions

            # Add query issues
            query_issues = []
            if (
                "too general" in main_message.lower()
                or "too broad" in main_message.lower()
            ):
                query_issues.append("Query too vague to generate meaningful SQL")
            if "need to know" in main_message.lower():
                query_issues.append(
                    "Missing required information about tables or columns"
                )
            response["query_issues"] = query_issues

            return jsonify(response)

        if not pipeline_result.success or not pipeline_result.sql:
            # Check if validation stage has clarification data
            validation_stage = pipeline_result.stages.get(Stage.QUERY_VALIDATION)

            if validation_stage and validation_stage.data:
                # Parse clarification message if it contains CANNOT FIND TABLES
                clarification_text = validation_stage.data

                if "CANNOT FIND TABLES:" in clarification_text:
                    # Extract the clarification message
                    clarification_parts = clarification_text.split(
                        "CANNOT FIND TABLES:"
                    )
                    main_message = (
                        clarification_parts[1].strip()
                        if len(clarification_parts) > 1
                        else clarification_text
                    )

                    response["clarification"] = main_message
                    response["context_type"] = "vague_query"
                    response["reasoning"] = "Query needs more specificity"

                    # Extract suggestions if present
                    suggestions = []
                    if "Try:" in main_message or "Please specify:" in main_message:
                        suggestions.append(
                            "Be more specific about what data you want to see"
                        )
                        suggestions.append("Include table or column names if known")
                        suggestions.append("Provide filters or conditions for the data")
                    response["suggestions"] = suggestions

                    # Add query issues
                    query_issues = []
                    if (
                        "too vague" in main_message.lower()
                        or "too broad" in main_message.lower()
                    ):
                        query_issues.append(
                            "Query too vague to generate meaningful SQL"
                        )
                    if "unclear" in main_message.lower():
                        query_issues.append("Unclear what specific data is requested")
                    response["query_issues"] = query_issues

                else:
                    # Still a clarification but different format
                    response["clarification"] = clarification_text
                    response["context_type"] = "validation_failed"
                    response["reasoning"] = (
                        validation_stage.error or "Query validation failed"
                    )
            else:
                response["error"] = "Could not generate SQL query"

            return jsonify(response)

        sql = pipeline_result.sql

        # Legacy check for old format - keeping for backwards compatibility
        if sql.strip().startswith("CANNOT FIND TABLES:"):
            clarification = sql.replace("CANNOT FIND TABLES:", "").strip()
            response["clarification"] = clarification
            return jsonify(response)

        # Execute SQL
        data_df = None
        if hasattr(pipeline_result, "data") and pipeline_result.data is not None:
            data_df = pipeline_result.data
        elif "QUERY_EXECUTION" in pipeline_result.stages:
            execution_stage = pipeline_result.stages["QUERY_EXECUTION"]
            if execution_stage.success and execution_stage.data is not None:
                data_df = execution_stage.data
        else:
            data_df = rag_engine.execute_sql(sql)

        if data_df is None or data_df.empty:
            response["sql"] = sql
            response["no_data"] = True
            return jsonify(response)

        # Create visualization with flexible extraction
        figure = None
        viz_metadata = None

        if hasattr(pipeline_result, "visualization") and pipeline_result.visualization:
            viz_schema = pipeline_result.visualization

            # Use flexible extractor
            figure, viz_metadata = extract_plotly_figure(viz_schema)

            # If extraction failed but we have data, try auto-generation
            if not figure and data_df is not None and not data_df.empty:
                logger.info("Falling back to auto-generated visualization")
                figure = auto_generate_fallback(data_df, message)
                viz_metadata = {"auto_generated": True, "original_query": message}

            # Enhance figure with metadata if available
            if figure and viz_metadata:
                figure = enhance_figure_with_metadata(figure, viz_metadata)

        # Prepare response
        response["success"] = True
        response["sql"] = sql
        response["rows"] = len(data_df)
        response["columns"] = len(data_df.columns)
        response["sample_data"] = data_df.head(3).to_html(classes="table table-sm")

        if figure:
            response["figure"] = json.loads(
                plotly.utils.PlotlyJSONEncoder().encode(figure)
            )
            response["has_visualization"] = True

            # Store visualization data with enhanced metadata
            viz_id = f"viz_{datetime.datetime.now().strftime('%H%M%S')}"
            context = {
                "original_question": message,
                "sql_query": sql,
                "data_shape": f"{len(data_df)} rows, {len(data_df.columns)} columns",
                "data_columns": list(data_df.columns),
                "plotly_schema": viz_schema,
            }

            # Merge visualization metadata into context
            if viz_metadata:
                context.update(
                    {
                        "chart_type": viz_metadata.get("chart_type"),
                        "confidence": viz_metadata.get("confidence"),
                        "reasoning": viz_metadata.get("reasoning"),
                        "insights": viz_metadata.get("insights"),
                        "auto_generated": viz_metadata.get("auto_generated", False),
                    }
                )

            visualizations[viz_id] = {
                "figure": figure,
                "data": data_df,
                "context": context,
                "metadata": viz_metadata,
                "viz_id": viz_id,
            }
            response["viz_id"] = viz_id

            # Initialize conversation history for this visualization
            conversation_histories[viz_id] = [
                {"type": "user", "message": message},
                {
                    "type": "assistant",
                    "message": "Visualization created. You can now ask me to modify it.",
                },
            ]

            modification_handler.sync_history(viz_id, conversation_histories[viz_id])

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/transfer_visualization", methods=["POST"])
def transfer_visualization():
    """Transfer visualization to side pane for persistent display."""
    global chart_counter
    try:
        data = request.json
        viz_id = data.get("viz_id")

        if viz_id not in visualizations:
            return jsonify({"success": False, "error": "Visualization not found"})

        _ = visualizations[viz_id]
        chart_counter += 1
        tab_name = f"Chart {chart_counter}"

        # Store in session for side pane
        if "side_pane_tabs" not in session:
            session["side_pane_tabs"] = {}

        session["side_pane_tabs"][tab_name] = viz_id
        session.modified = True

        return jsonify({"success": True, "tab_name": tab_name, "viz_id": viz_id})

    except Exception as e:
        logger.error(f"Error transferring visualization: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_visualization/<viz_id>")
def get_visualization(viz_id):
    """Retrieve stored visualization data by ID."""
    try:
        if viz_id not in visualizations:
            return jsonify({"success": False, "error": "Visualization not found"})

        viz_data = visualizations[viz_id]
        context = viz_data["context"]

        # Prepare data for different tabs
        response = {
            "success": True,
            "chart": json.loads(
                plotly.utils.PlotlyJSONEncoder().encode(viz_data["figure"])
            ),
            "sql": context.get("sql_query", ""),
            "original_question": context.get("original_question", ""),
            "data_shape": context.get("data_shape", ""),
            "columns": context.get(
                "data_columns",
                list(viz_data["data"].columns) if "data" in viz_data else [],
            ),
            "vis_config": context.get("plotly_schema", {}),
            "sample_data": viz_data["data"].head(10).to_html(classes="table table-sm")
            if "data" in viz_data
            else "",
            # Include metadata for enhanced UI display
            "metadata": viz_data.get("metadata", {}),
            "chart_type": context.get("chart_type"),
            "confidence": context.get("confidence"),
            "insights": context.get("insights", []),
            "auto_generated": context.get("auto_generated", False),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting visualization: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/modify_visualization", methods=["POST"])
def modify_visualization():
    """Process natural language requests to modify existing visualizations."""
    try:
        data = request.json
        viz_id = data.get("viz_id")
        modification = data.get("modification")

        if viz_id not in visualizations:
            return jsonify({"success": False, "error": "Visualization not found"})

        viz_data = visualizations[viz_id]

        # Get or create conversation history
        if viz_id not in conversation_histories:
            conversation_histories[viz_id] = []

        # Add user message to conversation history
        conversation_histories[viz_id].append({"type": "user", "message": modification})

        # Process modification using ModificationHandler
        result = modification_handler.process_modification_request(
            user_message=modification,
            current_viz_data=viz_data,
            viz_id=viz_id,
            conversation_history=conversation_histories[viz_id],
        )

        # Add assistant response to conversation history
        conversation_histories[viz_id].append(
            {
                "type": "assistant",
                "message": result.get("message", "Modification processed"),
            }
        )

        modification_handler.sync_history(viz_id, conversation_histories[viz_id])

        if result.get("success"):
            # Update visualization if modified
            if "updated_figure" in result:
                visualizations[viz_id]["figure"] = result["updated_figure"]

            if "updated_context" in result:
                visualizations[viz_id]["context"] = result["updated_context"]

            if "updated_data" in result:
                visualizations[viz_id]["data"] = result["updated_data"]

            return jsonify(
                {
                    "success": True,
                    "message": result.get("message", "Visualization updated"),
                    "chart": json.loads(
                        plotly.utils.PlotlyJSONEncoder().encode(
                            result["updated_figure"]
                        )
                    )
                    if "updated_figure" in result
                    else None,
                }
            )

        return jsonify(
            {
                "success": False,
                "message": result.get("message", "Could not modify visualization"),
            }
        )

    except Exception as e:
        logger.error(f"Error modifying visualization: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_conversation_history/<viz_id>")
def get_conversation_history(viz_id):
    """Retrieve conversation history for a specific visualization."""
    try:
        if viz_id not in conversation_histories:
            return jsonify({"success": True, "history": []})

        return jsonify({"success": True, "history": conversation_histories[viz_id]})

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    logger.info("Starting Flask instantinsight application")
    app.run(debug=True, port=5007)
