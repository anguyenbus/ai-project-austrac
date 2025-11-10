"""
Minimal RAG evaluator for text2SQL.

Integrates with RAGEngine and provides production-ready metrics.
"""

import pandas as pd
from loguru import logger

from .metrics.rag_metrics import RAGMetrics


class RAGEvaluator:
    """Minimal evaluator for RAG-based text2SQL systems."""

    def __init__(
        self,
        rag_engine=None,
    ):
        """
        Initialize RAG evaluator.

        Args:
            rag_engine: RAGEngine instance

        """
        self.rag_engine = rag_engine
        self.metrics = RAGMetrics()

        # Results storage
        self.evaluation_results = []

    def evaluate_single(
        self,
        question: str,
        generated_sql: str,
        retrieved_context: str = "",
        successful_execution: bool = False,
        expected_sql: str | None = None,
    ) -> dict:
        """
        Evaluate a single question-SQL pair.

        Args:
            question: Natural language question
            generated_sql: Generated SQL query
            retrieved_context: Retrieved RAG context (DDL, examples)
            successful_execution: Whether SQL executed successfully
            expected_sql: Expected SQL (optional, for comparison)

        Returns:
            Dictionary with evaluation metrics

        """
        result = {
            "question": question,
            "generated_sql": generated_sql,
            "successful_execution": successful_execution,
        }

        # Core metrics
        if retrieved_context:
            result["schema_adherence"] = self.metrics.schema_adherence_score(
                generated_sql, retrieved_context
            )
            result["context_recall"] = self.metrics.context_recall_score(
                generated_sql, retrieved_context, expected_sql, successful_execution
            )

        # Store result
        self.evaluation_results.append(result)

        return result

    def evaluate_consistency(
        self,
        question: str,
        num_generations: int = 3,
    ) -> dict:
        """
        Evaluate consistency by generating SQL multiple times.

        Args:
            question: Natural language question
            num_generations: Number of times to generate SQL

        Returns:
            Dictionary with consistency metrics

        """
        if not self.rag_engine:
            return {"error": "No RAGEngine provided"}

        generated_sqls = []

        for i in range(num_generations):
            try:
                sql = self.rag_engine.generate_sql(question)
                if sql and not sql.startswith("CANNOT FIND TABLES"):
                    generated_sqls.append(sql)
            except Exception as e:
                logger.warning(f"Generation {i + 1} failed: {e}")

        if len(generated_sqls) < 2:
            return {
                "question": question,
                "consistency": 1.0,  # Single or no generation = perfect consistency
                "num_generations": len(generated_sqls),
                "error": "Insufficient generations for consistency check",
            }

        consistency_score = self.metrics.consistency_score(generated_sqls)

        return {
            "question": question,
            "consistency": consistency_score,
            "num_generations": len(generated_sqls),
            "unique_queries": len(set(sql.lower().strip() for sql in generated_sqls)),
        }

    def evaluate_batch(
        self,
        test_questions: list[dict],
        include_consistency: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate a batch of test questions.

        Args:
            test_questions: List of dicts with 'question' and optionally 'expected_sql'
            include_consistency: Whether to include consistency evaluation

        Returns:
            DataFrame with evaluation results

        """
        results = []

        for i, test_case in enumerate(test_questions, 1):
            question = test_case.get("question", "")
            expected_sql = test_case.get("sql", "")

            logger.info(f"Evaluating {i}/{len(test_questions)}: {question[:50]}...")

            # Generate SQL and get context if possible
            try:
                # For now, we'll use the standard generate_sql
                # In future, modify to return context as well
                generated_sql = self.rag_engine.generate_sql(question)

                # Try to execute to check validity
                successful_execution = False
                if generated_sql and not generated_sql.startswith("CANNOT FIND"):
                    try:
                        df = self.rag_engine.execute_sql(generated_sql)
                        successful_execution = not df.empty
                    except Exception:
                        successful_execution = False

                # Evaluate single query
                eval_result = self.evaluate_single(
                    question=question,
                    generated_sql=generated_sql,
                    retrieved_context="",  # Would need modification to get this
                    successful_execution=successful_execution,
                    expected_sql=expected_sql,
                )

                # Add consistency check if requested
                if include_consistency:
                    consistency_result = self.evaluate_consistency(question)
                    eval_result["consistency"] = consistency_result.get(
                        "consistency", 0.0
                    )

                results.append(eval_result)

            except Exception as e:
                logger.error(f"Failed to evaluate question {i}: {e}")
                results.append(
                    {
                        "question": question,
                        "error": str(e),
                        "generated_sql": "",
                        "successful_execution": False,
                    }
                )

        return pd.DataFrame(results)

    def get_summary_statistics(self) -> dict:
        """Get summary statistics from all evaluations."""
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}

        df = pd.DataFrame(self.evaluation_results)

        stats = {
            "total_evaluations": len(df),
            "successful_executions": df["successful_execution"].sum()
            if "successful_execution" in df
            else 0,
            "execution_rate": df["successful_execution"].mean()
            if "successful_execution" in df
            else 0.0,
        }

        # Add metric averages if available
        for metric in ["schema_adherence", "context_recall", "consistency"]:
            if metric in df.columns:
                stats[f"avg_{metric}"] = df[metric].mean()
                stats[f"min_{metric}"] = df[metric].min()
                stats[f"max_{metric}"] = df[metric].max()

        # Hallucination statistics
        if "has_hallucination" in df.columns:
            stats["hallucination_rate"] = df["has_hallucination"].mean()
            stats["queries_with_hallucination"] = df["has_hallucination"].sum()

        return stats

    def save_results(self, filepath: str = "rag_evaluation_results.csv"):
        """Save evaluation results to CSV."""
        if not self.evaluation_results:
            logger.warning("No results to save")
            return

        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} evaluation results to {filepath}")

    def clear_results(self):
        """Clear stored evaluation results."""
        self.evaluation_results = []
        logger.info("Cleared evaluation results")
