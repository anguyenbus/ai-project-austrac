"""Data processing utilities for visualization agent."""

from typing import Any

import pandas as pd
from loguru import logger


class DataProcessors:
    """Utility class for data processing operations."""

    @staticmethod
    def filter_data(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
        """
        Apply filters to dataframe.

        Args:
            df: Input dataframe
            filters: List of filter conditions

        Returns:
            Filtered dataframe

        """
        if not filters:
            return df

        try:
            query_str = " and ".join(filters)
            return df.query(query_str)
        except Exception as e:
            logger.warning(f"Filter failed: {e}, trying with escaped columns")
            escaped_filters = []
            for f in filters:
                parts = f.split()
                if len(parts) >= 3:
                    col = parts[0]
                    if " " in col or "-" in col:
                        col = f"`{col}`"
                    escaped_filters.append(f"{col} {' '.join(parts[1:])}")
                else:
                    escaped_filters.append(f)

            query_str = " and ".join(escaped_filters)
            return df.query(query_str)

    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze dataframe structure for visualization.

        Args:
            df: Result dataframe from SQL query

        Returns:
            Analysis of dataframe characteristics

        """
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
        }

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "unique_count": df[col].nunique(),
                "null_count": int(df[col].isnull().sum()),
            }

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["type"] = "temporal"
                col_info["min"] = str(df[col].min())
                col_info["max"] = str(df[col].max())
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_info["type"] = "numerical"
                col_info["min"] = (
                    float(df[col].min()) if not df[col].isna().all() else None
                )
                col_info["max"] = (
                    float(df[col].max()) if not df[col].isna().all() else None
                )
                col_info["mean"] = (
                    float(df[col].mean()) if not df[col].isna().all() else None
                )
            else:
                col_info["type"] = "categorical"
                top_values = df[col].value_counts().head(5)
                col_info["top_values"] = top_values.index.tolist()
                col_info["top_frequencies"] = [int(x) for x in top_values.values]

            analysis["columns"][col] = col_info

        return analysis
