"""Chart generation utilities for visualization agent."""

import pandas as pd

from .vis_types import AggregationType


class ChartGenerators:
    """Utility class for generating different chart types."""

    @staticmethod
    def apply_aggregation(
        df: pd.DataFrame,
        group_cols: list[str],
        value_col: str,
        agg_type: AggregationType,
    ) -> pd.DataFrame:
        """Apply aggregation to dataframe."""
        if not group_cols:
            if agg_type == AggregationType.COUNTROWS:
                return pd.DataFrame({value_col: [len(df)]})
            else:
                agg_value = df[value_col].agg(agg_type.value)
                return pd.DataFrame({value_col: [agg_value]})

        grouped = df.groupby(group_cols, dropna=False)

        if agg_type == AggregationType.COUNTROWS:
            # Use a different column name if value_col is already in group_cols
            count_col = (
                value_col if value_col not in group_cols else f"{value_col}_count"
            )
            result = grouped.size().reset_index(name=count_col)
        else:
            result = grouped[value_col].agg(agg_type.value).reset_index()

        return result

    @staticmethod
    def generate_bar_chart(df, rec, chart_json):
        """Generate bar chart traces."""
        x_col = rec.x_axis.column if rec.x_axis else None
        y_cols = [rec.y_axis.column] if rec.y_axis else rec.y_columns

        for y_col in y_cols:
            if rec.y_axis and rec.y_axis.aggregation and x_col:
                agg_df = ChartGenerators.apply_aggregation(
                    df, [x_col], y_col, rec.y_axis.aggregation
                )
                x_data = agg_df[x_col].tolist()
                # Handle potential column name change for COUNTROWS
                actual_y_col = (
                    f"{y_col}_count"
                    if y_col == x_col
                    and rec.y_axis.aggregation == AggregationType.COUNTROWS
                    else y_col
                )
                y_data = agg_df[actual_y_col].tolist()
            else:
                x_data = df[x_col].tolist() if x_col else list(range(len(df)))
                y_data = df[y_col].tolist()

            trace = {
                "x": x_data,
                "y": y_data,
                "type": "bar",
                "name": y_col,
            }

            if rec.orientation == "h":
                trace["x"], trace["y"] = trace["y"], trace["x"]
                trace["orientation"] = "h"

            chart_json["data"].append(trace)

        if rec.stacked:
            chart_json["layout"]["barmode"] = "stack"

    @staticmethod
    def generate_line_chart(df, rec, chart_json):
        """Generate line chart traces."""
        x_col = rec.x_axis.column if rec.x_axis else None
        y_cols = [rec.y_axis.column] if rec.y_axis else rec.y_columns

        for y_col in y_cols:
            trace = {
                "x": df[x_col].tolist() if x_col else list(range(len(df))),
                "y": df[y_col].tolist(),
                "type": "scatter",
                "mode": "lines+markers",
                "name": y_col,
            }
            chart_json["data"].append(trace)

    @staticmethod
    def generate_scatter_chart(df, rec, chart_json):
        """Generate scatter chart traces."""
        x_col = rec.x_axis.column if rec.x_axis else None
        y_cols = [rec.y_axis.column] if rec.y_axis else rec.y_columns

        for y_col in y_cols:
            trace = {
                "x": df[x_col].tolist() if x_col else list(range(len(df))),
                "y": df[y_col].tolist(),
                "type": "scatter",
                "mode": "markers",
                "name": y_col,
            }
            chart_json["data"].append(trace)

    @staticmethod
    def generate_pie_chart(df, rec, chart_json):
        """Generate pie chart trace."""
        if rec.labels_column and rec.values_column:
            if rec.y_axis and rec.y_axis.aggregation:
                agg_df = ChartGenerators.apply_aggregation(
                    df, [rec.labels_column], rec.values_column, rec.y_axis.aggregation
                )
                labels = agg_df[rec.labels_column].tolist()
                # Handle potential column name change for COUNTROWS
                actual_values_col = (
                    f"{rec.values_column}_count"
                    if rec.values_column == rec.labels_column
                    and rec.y_axis.aggregation == AggregationType.COUNTROWS
                    else rec.values_column
                )
                values = agg_df[actual_values_col].tolist()
            else:
                labels = df[rec.labels_column].tolist()
                values = df[rec.values_column].tolist()
        else:
            categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
            if categorical_cols:
                value_counts = df[categorical_cols[0]].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
            else:
                labels = []
                values = []

        trace = {
            "labels": labels,
            "values": values,
            "type": "pie",
        }
        chart_json["data"].append(trace)

    @staticmethod
    def generate_area_chart(df, rec, chart_json):
        """Generate area chart traces."""
        x_col = rec.x_axis.column if rec.x_axis else None
        y_cols = [rec.y_axis.column] if rec.y_axis else rec.y_columns

        for i, y_col in enumerate(y_cols):
            trace = {
                "x": df[x_col].tolist() if x_col else list(range(len(df))),
                "y": df[y_col].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": y_col,
                "stackgroup": "one" if rec.stacked else None,
                "fill": "tonexty" if i > 0 and not rec.stacked else "tozeroy",
            }
            chart_json["data"].append(trace)

    @staticmethod
    def generate_histogram(df, rec, chart_json):
        """Generate histogram trace."""
        x_col = rec.x_axis.column if rec.x_axis else df.columns[0]
        trace = {
            "x": df[x_col].tolist(),
            "type": "histogram",
            "name": x_col,
        }
        if rec.x_axis and rec.x_axis.bin_size:
            trace["nbinsx"] = rec.x_axis.bin_size
        chart_json["data"].append(trace)

    @staticmethod
    def generate_box_plot(df, rec, chart_json):
        """Generate box plot traces."""
        y_cols = [rec.y_axis.column] if rec.y_axis else rec.y_columns
        x_col = rec.x_axis.column if rec.x_axis else None

        for y_col in y_cols:
            trace = {"y": df[y_col].tolist(), "type": "box", "name": y_col}
            if x_col:
                trace["x"] = df[x_col].tolist()
            chart_json["data"].append(trace)

    @staticmethod
    def generate_heatmap(df, rec, chart_json):
        """Generate heatmap trace."""
        if rec.z_column:
            y_col = rec.y_columns[0] if rec.y_columns else df.columns[0]
            x_col = rec.x_axis.column if rec.x_axis else df.columns[1]

            pivot_df = df.pivot_table(
                values=rec.z_column, index=y_col, columns=x_col, aggfunc="mean"
            )
            trace = {
                "z": pivot_df.values.tolist(),
                "x": pivot_df.columns.tolist(),
                "y": pivot_df.index.tolist(),
                "type": "heatmap",
            }
            chart_json["data"].append(trace)

    @staticmethod
    def generate_treemap(df, rec, chart_json):
        """Generate treemap traces."""
        if rec.labels_column and rec.values_column:
            labels = df[rec.labels_column].tolist()
            values = df[rec.values_column].tolist()
        elif rec.x_axis and rec.y_axis:
            labels = df[rec.x_axis.column].tolist()
            values = df[rec.y_axis.column].tolist()
        else:
            string_cols = df.select_dtypes(include=["object", "category"]).columns
            numeric_cols = df.select_dtypes(include=["number"]).columns

            if len(string_cols) > 0 and len(numeric_cols) > 0:
                labels = df[string_cols[0]].tolist()
                values = df[numeric_cols[0]].tolist()
            else:
                labels = df.index.astype(str).tolist()
                values = df.iloc[:, 0].tolist()

        trace = {
            "type": "treemap",
            "labels": labels,
            "values": values,
            "parents": [""] * len(labels),
        }

        chart_json["data"].append(trace)

    @staticmethod
    def generate_scalar(df, rec, chart_json):
        """Generate scalar (single value) display."""
        if rec.y_axis and rec.y_axis.aggregation:
            value = df[rec.y_axis.column].agg(rec.y_axis.aggregation.value)
        else:
            value = df.iloc[0, 0] if not df.empty else 0

        trace = {
            "type": "indicator",
            "mode": "number",
            "value": float(value),
            "title": {"text": rec.title},
        }
        chart_json["data"].append(trace)
