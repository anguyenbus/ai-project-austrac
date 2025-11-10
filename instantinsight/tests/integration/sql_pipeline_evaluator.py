"""SQL pipeline evaluator for testing natural language to SQL conversion."""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import RAGEvaluator
from src.rag import RAGEngine
from src.rag.pipeline import Pipeline

# Resolve test file path relative to project root
PROJECT_ROOT = (
    Path(__file__).parent.parent.parent
)  # Go up 3 levels: sql_pipeline_evaluator.py -> integration -> tests -> nl2vis
TEST_FILE = PROJECT_ROOT / "tests/data/validation/custom_analysers_tests.yaml"


class SQLPipelineEvaluator:
    """
    Evaluate SQL generation pipeline performance and accuracy.

    This class tests the complete Natural Language to SQL pipeline by:
    - Loading test queries from YAML files
    - Generating SQL from natural language questions
    - Executing and validating the generated SQL
    - Calculating RAG metrics and performance statistics
    - Saving detailed evaluation results
    """

    def __init__(self, use_pipeline: bool = True, enable_rag_metrics: bool = True):
        """
        Initialize the SQL Pipeline Evaluator.

        Args:
            use_pipeline: Whether to use Pipeline with recovery refinement
                (default: True)
            enable_rag_metrics: Whether to enable RAG evaluation metrics
                (default: True)

        """
        logger.info("Initializing SQL Pipeline Evaluator")
        # Create results directory if it doesn't exist
        self.results_dir = Path(__file__).parent / "pipeline_evaluation_results"
        self.results_dir.mkdir(exist_ok=True)

        # Load SQL examples from YAML
        self.sql_examples = self._load_sql_examples()

        # Set RAG metrics flag first (needed by Pipeline initialization)
        self.enable_rag_metrics = enable_rag_metrics
        self.rag_evaluator = None

        # Initialize Pipeline for SQL generation with recovery refinement
        self.use_pipeline = use_pipeline
        self.pipeline = None
        self.rag_engine = None
        if use_pipeline:
            logger.info(
                f"Attempting to initialize Pipeline (use_pipeline={use_pipeline})"
            )
            self._init_pipeline()
            logger.info(
                f"After initialization: use_pipeline={self.use_pipeline}, "
                f"pipeline exists={self.pipeline is not None}"
            )

    def _init_pipeline(self):
        """Initialize Pipeline with RAGEngine and recovery refinement."""
        try:
            # Get RAGEngine instance for Pipeline
            vn, error = RAGEngine.create_instance()

            if vn:
                # Store RAGEngine instance for RAG evaluator
                self.rag_engine = vn
                logger.info("✓ RAGEngine obtained successfully")

                # Initialize Pipeline with RAGEngine (includes recovery refinement)
                self.pipeline = Pipeline(rag_instance=vn)
                logger.info("✓ Pipeline object created successfully")

                # Disable cache for validation to ensure fresh generation
                if hasattr(self.pipeline, "semantic_cache"):
                    self.pipeline.semantic_cache = None
                    logger.info("✓ Pipeline cache disabled for validation")

                # Verify Pipeline is working
                if hasattr(self.pipeline, "process") and callable(
                    self.pipeline.process
                ):
                    logger.info("✓ Pipeline process method available")
                    # Keep use_pipeline as True - don't change it
                    logger.info(
                        "✓ Pipeline initialized with recovery refinement for validation"
                    )
                else:
                    logger.error("Pipeline process method not available")
                    self.use_pipeline = False
                    return

                # Initialize RAG evaluator now that we have rag_engine
                if self.enable_rag_metrics:
                    self._init_rag_evaluator()

            else:
                logger.error(f"RAGEngine initialization failed: {error}")
                self.use_pipeline = False
                return

        except Exception as e:
            logger.error(f"Failed to initialize Pipeline: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            self.use_pipeline = False

    def _init_rag_evaluator(self):
        """Initialize RAG evaluator for metrics calculation."""
        try:
            self.rag_evaluator = RAGEvaluator(
                rag_engine=self.rag_engine,
            )
            logger.info("✓ RAG evaluator initialized for metrics calculation")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG evaluator: {e}")
            self.enable_rag_metrics = False

    def _load_sql_examples(self):
        """Load SQL examples from sql_examples.yaml file."""
        yaml_file = TEST_FILE

        if not yaml_file.exists():
            logger.error(f"YAML file not found: {yaml_file}")
            return []

        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Handle both list format and nested format
            if isinstance(data, list):
                queries = data
            else:
                queries = data.get("examples", {}).get("queries", [])

            logger.info(f"Loaded {len(queries)} queries from {yaml_file.name}")
            return queries

        except Exception as e:
            logger.error(f"Error loading YAML file: {e}")
            return []

    def _execute_sql_with_retry(self, sql: str, query_type: str) -> pd.DataFrame:
        """
        Execute SQL query with error handling and delay to prevent Athena throttling.

        Args:
            sql: SQL query to execute
            query_type: Type of query (for logging purposes)

        Returns:
            DataFrame with query results

        """
        try:
            logger.info(f"Executing {query_type} SQL...")

            # Add 1.5s delay before Athena query to prevent throttling/skipping
            logger.debug("Adding 1.5s delay before Athena query execution...")
            time.sleep(1.5)  # 1.5 second delay to prevent Athena from skipping tests

            # Use RAGEngine directly for SQL execution
            if hasattr(self, "rag_engine") and self.rag_engine:
                df = self.rag_engine.execute_sql(sql)
            else:
                # Fallback: create instance if needed
                vn, _ = RAGEngine.create_instance()
                if vn:
                    df = vn.execute_sql(sql)
                else:
                    df = None

            if df is None:
                logger.warning(f"{query_type} SQL execution returned None")
                return pd.DataFrame()
            elif df.empty:
                logger.warning(f"{query_type} SQL returned no data")
                return df
            else:
                logger.info(
                    f"{query_type} SQL executed successfully: {len(df)} rows returned"
                )
                return df

        except Exception as e:
            logger.error(f"{query_type} SQL execution failed: {e}")
            return pd.DataFrame()

    def _save_query_results(
        self, df: pd.DataFrame, query_num: int, sql_type: str
    ) -> str:
        """Save query results to CSV file."""
        if df.empty:
            logger.warning(f"No data to save for query_{query_num}_{sql_type}.csv")
            return ""

        filename = f"query_{query_num}_{sql_type}.csv"
        filepath = self.results_dir / filename

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Results saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            return ""

    def _generate_sql_from_question(self, question: str):
        """
        Generate SQL from natural language question.

        Returns:
            Tuple of (sql, execution_success, pipeline_result) where
            execution_success indicates if the SQL was successfully executed
            (with or without refinement)

        """
        try:
            logger.info("Generating SQL from natural language question...")

            # Debug Pipeline availability
            logger.debug(
                f"use_pipeline: {self.use_pipeline}, "
                f"pipeline: {self.pipeline is not None}"
            )

            # Use Pipeline if available (includes recovery refinement)
            if self.use_pipeline and self.pipeline:
                logger.debug(
                    "Using Pipeline for SQL generation with recovery refinement"
                )
                result = self.pipeline.process(question, return_context=True)

                if result.success and result.sql:
                    logger.info("SQL generated and executed successfully")
                    # Check if refinement was applied
                    if "QUERY_EXECUTION_RETRY" in result.stages:
                        logger.info("✅ Recovery refinement was applied successfully")
                    return result.sql, True, result
                elif result.sql:
                    # SQL was generated but execution failed
                    logger.warning("SQL generated but execution failed")
                    return result.sql, False, result
                else:
                    logger.warning("SQL generation failed")
                    return "", False, result
            else:
                # Try using RAGEngine directly as fallback
                if self.rag_engine:
                    logger.warning("Pipeline not available, using RAGEngine directly")
                    sql = self.rag_engine.generate_sql(question)
                    if sql and not sql.startswith("CANNOT FIND"):
                        return (
                            sql,
                            False,
                            None,
                        )  # Not executed through Pipeline, so execution_success
                        # = False, no pipeline result
                    else:
                        return "", False, None
                else:
                    logger.warning("Neither Pipeline nor RAGEngine available")
                    return "", False, None

        except Exception as e:
            logger.error(f"SQL generation/execution failed: {e}")
            return "", False, None

    def process_all_queries(self) -> pd.DataFrame:
        """Process all queries and create validation summary."""
        if not self.sql_examples:
            logger.error("No SQL examples loaded. Cannot proceed.")
            return pd.DataFrame()

        logger.info(f"Processing {len(self.sql_examples)} queries...")

        validation_data = []

        for i, query_data in enumerate(self.sql_examples, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing Query {i}/{len(self.sql_examples)}")
            logger.info(f"{'=' * 60}")

            # Add delay between queries to prevent Athena throttling
            if i > 1:
                logger.debug("Adding 2s delay between test queries...")
                time.sleep(2.0)  # 2 second delay between each test query

            question = query_data.get("question", f"Query {i}")
            expected_sql = query_data.get("sql", "").strip()

            logger.info(f"Question: {question}")

            # Initialize validation row
            validation_row = {
                "query_number": i,
                "query": question,
                "expected_sql": expected_sql,
                "generated_sql": "",
                "table_generated": False,
                "sql_method": (
                    "Pipeline+Refinement"
                    if self.use_pipeline and self.pipeline
                    else "Direct"
                ),
                "execution_success": False,
                "refinement_applied": False,
                # Row count metrics
                "expected_rows": 0,
                "generated_rows": 0,
                "rows_match": False,
                # RAG metrics
                "schema_adherence": 0.0,
                "context_recall": 0.0,
                "has_hallucination": False,
                "consistency": 0.0,
            }

            # Process expected SQL
            expected_df = pd.DataFrame()
            if expected_sql:
                logger.info("Processing expected SQL...")
                expected_df = self._execute_sql_with_retry(expected_sql, "expected")
                if not expected_df.empty:
                    self._save_query_results(expected_df, i, "expected")
                    validation_row["expected_rows"] = len(expected_df)
            else:
                logger.warning(f"No expected SQL found for query {i}")

            # Generate and process SQL from question
            # (with automatic refinement if needed)
            generated_sql, execution_success, pipeline_result = (
                self._generate_sql_from_question(question)
            )
            validation_row["generated_sql"] = generated_sql
            validation_row["execution_success"] = execution_success

            generated_df = pd.DataFrame()
            if generated_sql and execution_success:
                # Pipeline already executed the query successfully
                logger.info("SQL was generated and executed successfully via Pipeline")
                validation_row["table_generated"] = True
                # Note: We could extract the dataframe from Pipeline result if needed
                # For now, re-execute to get the dataframe for saving
                generated_df = self._execute_sql_with_retry(generated_sql, "generated")
                if not generated_df.empty:
                    self._save_query_results(generated_df, i, "generated")
                    validation_row["generated_rows"] = len(generated_df)
            elif generated_sql:
                logger.warning(f"SQL generated but execution failed for query {i}")
            else:
                logger.warning(f"Could not generate SQL for query {i}")

            # Simple row count comparison
            if expected_sql and generated_sql:
                expected_rows = len(expected_df)
                generated_rows = len(generated_df)
                validation_row["expected_rows"] = expected_rows
                validation_row["generated_rows"] = generated_rows

                # Check if row counts match
                validation_row["rows_match"] = expected_rows == generated_rows

                logger.info(
                    f"Row count comparison - Expected: {expected_rows}, "
                    f"Generated: {generated_rows}, Match: {validation_row['rows_match']}"
                )

            # Calculate RAG metrics if evaluator is available
            if self.enable_rag_metrics and self.rag_evaluator and generated_sql:
                try:
                    # Get retrieved context from Pipeline result if available
                    retrieved_context = ""
                    if (
                        pipeline_result
                        and hasattr(pipeline_result, "retrieved_context")
                        and pipeline_result.retrieved_context
                    ):
                        retrieved_context = pipeline_result.retrieved_context
                        logger.debug(
                            f"Using retrieved context: {len(retrieved_context)} "
                            f"characters"
                        )

                    rag_result = self.rag_evaluator.evaluate_single(
                        question=question,
                        generated_sql=generated_sql,
                        retrieved_context=retrieved_context,
                        successful_execution=execution_success,
                    )

                    # Update validation row with RAG metrics
                    validation_row["schema_adherence"] = rag_result.get(
                        "schema_adherence", 0.0
                    )
                    validation_row["context_recall"] = rag_result.get(
                        "context_recall", 0.0
                    )
                    validation_row["has_hallucination"] = rag_result.get(
                        "has_hallucination", False
                    )

                    logger.info(
                        f"RAG metrics calculated - Schema adherence: "
                        f"{validation_row['schema_adherence']:.3f}, "
                        f"Context: {len(retrieved_context)} chars"
                    )

                except Exception as e:
                    logger.warning(f"RAG metrics calculation failed: {e}")
                    import traceback

                    logger.debug(f"RAG metrics error: {traceback.format_exc()}")

            # Optional consistency check (every 10th query to avoid overhead)
            if self.enable_rag_metrics and self.rag_evaluator and i % 10 == 0:
                try:
                    consistency_result = self.rag_evaluator.evaluate_consistency(
                        question, num_generations=2
                    )
                    validation_row["consistency"] = consistency_result.get(
                        "consistency", 0.0
                    )
                    logger.info(
                        f"Consistency score: {validation_row['consistency']:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"Consistency evaluation failed: {e}")

            # Add to validation data
            validation_data.append(validation_row)

            # Show progress summary
            logger.info(f"Query {i} Summary:")
            logger.info(f"  SQL Method: {validation_row['sql_method']}")
            logger.info(f"  Expected SQL provided: {'Yes' if expected_sql else 'No'}")
            logger.info(f"  Generated SQL: {'Yes' if generated_sql else 'No'}")
            logger.info(f"  Expected results: {len(expected_df)} rows")
            logger.info(f"  Generated results: {len(generated_df)} rows")
            logger.info(
                f"  Table generated: "
                f"{'Yes' if validation_row['table_generated'] else 'No'}"
            )

        # Create validation summary DataFrame
        validation_df = pd.DataFrame(validation_data)

        # Save validation summary
        summary_filepath = self.results_dir / "validation_summary.csv"
        validation_df.to_csv(summary_filepath, index=False)
        logger.info(f"\nValidation summary saved to: {summary_filepath}")

        # Display summary statistics
        self._display_summary_stats(validation_df)

        return validation_df

    def _display_summary_stats(self, validation_df: pd.DataFrame):
        """Display summary statistics."""
        total_queries = len(validation_df)
        expected_provided = len(validation_df[validation_df["expected_sql"] != ""])
        generated_success = len(validation_df[validation_df["generated_sql"] != ""])
        tables_generated = len(validation_df[validation_df["table_generated"]])

        # Count by SQL method
        pipeline_queries = len(
            validation_df[validation_df["sql_method"] == "Pipeline+Refinement"]
        )
        direct_queries = len(validation_df[validation_df["sql_method"] == "Direct"])

        # Count execution successes
        execution_successes = len(validation_df[validation_df["execution_success"]])

        logger.info(f"\n{'=' * 60}")
        logger.info("VALIDATION SUMMARY STATISTICS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total queries processed: {total_queries}")
        logger.info("SQL Generation Method:")
        logger.info(
            f"  - Pipeline+Refinement: {pipeline_queries}/{total_queries} "
            f"({pipeline_queries / total_queries * 100:.1f}%)"
        )
        logger.info(
            f"  - Direct: {direct_queries}/{total_queries} "
            f"({direct_queries / total_queries * 100:.1f}%)"
        )
        logger.info(
            f"Execution successes (with refinement): {execution_successes}/{total_queries} "
            f"({execution_successes / total_queries * 100:.1f}%)"
        )
        logger.info(
            f"Expected SQL provided: {expected_provided}/{total_queries} "
            f"({expected_provided / total_queries * 100:.1f}%)"
        )
        logger.info(
            f"SQL generated successfully: {generated_success}/{total_queries} "
            f"({generated_success / total_queries * 100:.1f}%)"
        )
        logger.info(
            f"Tables generated successfully: {tables_generated}/{total_queries} "
            f"({tables_generated / total_queries * 100:.1f}%)"
        )

        # RAG metrics summary
        if self.enable_rag_metrics and "schema_adherence" in validation_df.columns:
            avg_schema_adherence = validation_df["schema_adherence"].mean()
            avg_context_recall = validation_df["context_recall"].mean()
            hallucination_rate = validation_df["has_hallucination"].mean()

            logger.info("\nRAG EVALUATION METRICS:")
            logger.info(f"  - Average Schema Adherence: {avg_schema_adherence:.3f}")
            logger.info(f"  - Average Context Recall: {avg_context_recall:.3f}")
            logger.info(
                f"  - Hallucination Rate: {hallucination_rate:.3f} "
                f"({hallucination_rate * 100:.1f}%)"
            )

            if "consistency" in validation_df.columns:
                avg_consistency = validation_df[validation_df["consistency"] > 0][
                    "consistency"
                ].mean()
                if not pd.isna(avg_consistency):
                    logger.info(f"  - Average Consistency: {avg_consistency:.3f}")

        logger.info(f"\nFiles created in {self.results_dir}:")
        logger.info("  - validation_summary.csv (main results)")
        logger.info("  - query_{i}_expected.csv (expected results)")
        logger.info("  - query_{i}_generated.csv (generated results)")

        # Save RAG evaluator results if available
        if self.enable_rag_metrics and self.rag_evaluator:
            rag_results_file = str(self.results_dir / "rag_evaluation_results.csv")
            self.rag_evaluator.save_results(rag_results_file)
            logger.info("  - rag_evaluation_results.csv (detailed RAG metrics)")

        # Generate metrics summary CSV
        self._generate_metrics_summary(validation_df)

    def _generate_metrics_summary(self, validation_df: pd.DataFrame):
        """Generate a simple metrics summary JSON file."""
        import json

        try:
            # Calculate metrics
            total = len(validation_df)
            sql_generated = len(validation_df[validation_df["generated_sql"] != ""])
            execution_success = len(validation_df[validation_df["execution_success"]])
            tables_generated = len(validation_df[validation_df["table_generated"]])

            # Calculate accuracy (queries where row counts match)
            if "rows_match" in validation_df.columns:
                rows_match = len(validation_df[validation_df["rows_match"]])
                accuracy = (rows_match / total * 100) if total > 0 else 0
            else:
                accuracy = 0

            # Calculate percentages
            generation_rate = (sql_generated / total * 100) if total > 0 else 0
            success_rate = (execution_success / total * 100) if total > 0 else 0
            table_rate = (tables_generated / total * 100) if total > 0 else 0

            # RAG metrics if available
            if "schema_adherence" in validation_df.columns:
                avg_schema = validation_df["schema_adherence"].mean() * 100
                avg_recall = validation_df["context_recall"].mean() * 100
                hallucination_rate = validation_df["has_hallucination"].mean() * 100
            else:
                avg_schema = avg_recall = hallucination_rate = 0

            # Create metrics dictionary
            metrics_summary = {
                "total_tests": total,
                "sql_generated": sql_generated,
                "execution_success": execution_success,
                "tables_generated": tables_generated,
                "accuracy": round(accuracy, 1),
                "generation_rate_percent": round(generation_rate, 1),
                "success_rate_percent": round(success_rate, 1),
                "table_generation_rate_percent": round(table_rate, 1),
                "avg_schema_adherence_percent": round(avg_schema, 1),
                "avg_context_recall_percent": round(avg_recall, 1),
                "hallucination_rate_percent": round(hallucination_rate, 1),
                "timestamp": datetime.now().isoformat(),
            }

            # Save to JSON
            summary_file = self.results_dir / "metrics_summary.json"
            with open(summary_file, "w") as f:
                json.dump(metrics_summary, f, indent=2)

            logger.info("  - metrics_summary.json (aggregated metrics)")

        except Exception as e:
            logger.warning(f"Failed to generate metrics summary: {e}")


def main():
    """Run the SQL pipeline evaluation."""
    logger.info("Starting SQL Pipeline Evaluation")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize evaluator with Pipeline and RAG metrics
    # (includes recovery refinement, no caching)
    evaluator = SQLPipelineEvaluator(use_pipeline=True, enable_rag_metrics=True)

    # Process all queries
    validation_df = evaluator.process_all_queries()

    if not validation_df.empty:
        logger.info("\nSQL Pipeline evaluation completed successfully!")
        logger.info(f"Check the results directory: {evaluator.results_dir}")
    else:
        logger.error("SQL Pipeline evaluation failed - no results generated")


if __name__ == "__main__":
    main()
