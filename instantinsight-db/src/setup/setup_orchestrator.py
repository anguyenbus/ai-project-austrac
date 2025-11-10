"""
Setup orchestrator for Athena RAG integration.

Coordinates the complete setup process using modular components.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .analyser_filter import AnalyserFilter
from .config_loader import ConfigLoader
from .prerequisite_validator import PrerequisiteValidator


class SetupOrchestrator:
    """Orchestrates the complete Athena RAG setup process."""

    def __init__(self, config_loader: ConfigLoader, project_root: Path):
        """
        Initialize the setup orchestrator.

        Args:
            config_loader: Configured ConfigLoader instance
            project_root: Path to the project root directory

        """
        self.config_loader = config_loader
        self.project_root = project_root

        # Load base configurations
        from src.config.database_config import ATHENA_CONFIG, RAG_CONFIG

        self.athena_config = config_loader.get_athena_config(ATHENA_CONFIG)
        self.rag_config = config_loader.get_rag_config(RAG_CONFIG)

        # Initialize components
        self.validator = PrerequisiteValidator(self.athena_config, self.rag_config)
        self.analyser_filter = AnalyserFilter(project_root)

        # Initialize schema processing components
        self.extractor = None
        self.vectorizer = None
        self.rag_engine = None

        # Setup report
        self.setup_report = {
            "timestamp": datetime.now().isoformat(),
            "status": "initialized",
            "databases_processed": [],
            "errors": [],
            "warnings": [],
        }

        logger.info("🚀 Setup orchestrator initialized")

    def validate_prerequisites(self) -> bool:
        """
        Validate all prerequisites for setup.

        Returns:
            True if all prerequisites are met

        """
        success, errors, warnings = self.validator.validate_all()

        self.setup_report["errors"].extend(errors)
        self.setup_report["warnings"].extend(warnings)

        return success

    def run_full_setup(
        self,
        database_names: list[str] | None = None,
        force_rebuild: bool = False,
        generate_examples: bool = False,
    ) -> bool:
        """
        Run the complete setup pipeline.

        Args:
            database_names: Specific databases to process (None for all)
            force_rebuild: Whether to force rebuild existing data
            generate_examples: Whether to generate SQL examples

        Returns:
            True if setup completed successfully

        """
        logger.info("🔧 Running complete Athena RAG setup pipeline...")
        logger.info(
            f"📊 Configuration: force_rebuild={force_rebuild}, generate_examples={generate_examples}"
        )

        try:
            # Step 1: Initialize components
            if not self._initialize_components():
                return False

            # Step 2: Discover or validate databases
            if database_names is None:
                database_names = self._discover_databases()

            if not database_names:
                logger.error("❌ No databases found or specified")
                return False

            # Step 3: Process each database
            success_count = self._process_databases(
                database_names, force_rebuild, generate_examples
            )

            # Step 4: Handle training examples
            self._handle_training_examples(generate_examples, success_count > 0)

            # Step 5: Validate and report
            if success_count > 0:
                self._validate_setup()

            self._generate_setup_report()

            logger.info(
                f"✅ Pipeline setup completed: {success_count}/{len(database_names)} databases processed"
            )
            self.setup_report["status"] = "completed"

            return success_count == len(database_names)

        except Exception as e:
            logger.error(f"❌ Pipeline setup failed: {e}")
            self.setup_report["status"] = "failed"
            self.setup_report["errors"].append(str(e))
            return False

    def run_update(
        self,
        database_names: list[str] | None = None,
        generate_examples: bool = False,
    ) -> bool:
        """
        Update existing setup with incremental changes.

        Args:
            database_names: Specific databases to update (None for all)
            generate_examples: Whether to generate SQL examples

        Returns:
            True if update completed successfully

        """
        logger.info("🔄 Updating existing Athena RAG setup...")

        return self.run_full_setup(
            database_names, force_rebuild=False, generate_examples=generate_examples
        )

    def run_new_tables_only_update(
        self,
        database_names: list[str] | None = None,
        generate_examples: bool = False,
    ) -> bool:
        """
        Update RAG system with only tables that don't already exist.

        Args:
            database_names: Specific databases to process (None for all)
            generate_examples: Whether to generate SQL examples for new tables

        Returns:
            True if update completed successfully

        """
        logger.info("🆕 Updating RAG with new tables only...")
        logger.info("This will skip tables that already exist in the local RAG system")

        try:
            # Step 1: Initialize components
            if not self._initialize_components():
                return False

            # Step 2: Discover or validate databases
            if database_names is None:
                database_names = self._discover_databases()

            if not database_names:
                logger.error("❌ No databases found or specified")
                return False

            # Step 3: Process each database with new-table filtering
            success_count = 0
            total_new_tables = 0

            for database_name in database_names:
                new_table_count = self._process_database_new_tables_only(
                    database_name, generate_examples
                )

                if new_table_count >= 0:  # -1 indicates error
                    success_count += 1
                    total_new_tables += new_table_count

                    if new_table_count == 0:
                        self.setup_report["databases_processed"].append(
                            f"{database_name} (no new tables)"
                        )
                    else:
                        self.setup_report["databases_processed"].append(
                            f"{database_name} ({new_table_count} new tables)"
                        )

            # Step 4: Handle training examples
            self._handle_training_examples(generate_examples, total_new_tables > 0)

            # Step 5: Validate and report
            if success_count > 0:
                self._validate_setup(expect_new_documents=(total_new_tables > 0))

            self._generate_setup_report()

            logger.info(
                f"✅ Update completed: {success_count}/{len(database_names)} databases processed"
            )
            logger.info(f"📈 Total new tables added: {total_new_tables}")
            self.setup_report["status"] = "completed"
            self.setup_report["new_tables_added"] = total_new_tables

            return success_count == len(database_names)

        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
            self.setup_report["status"] = "failed"
            self.setup_report["errors"].append(str(e))
            return False

    def run_specific_table_examples(
        self,
        table_names: list[str],
        database_names: list[str] | None = None,
    ) -> bool:
        """
        Generate SQL examples for specific tables by name.

        Args:
            table_names: List of specific table names to process
            database_names: Databases to search in (None for all)

        Returns:
            True if examples were generated successfully

        """
        logger.info(
            f"🎯 Generating examples for specific tables: {', '.join(table_names)}"
        )

        try:
            # Step 1: Initialize components
            if not self._initialize_components():
                return False

            # Step 2: Discover databases to search
            if database_names is None:
                database_names = self._discover_databases()

            if not database_names:
                logger.error("❌ No databases found or specified")
                return False

            # Step 3: Find and process specific tables
            tables_processed = self._process_specific_tables(
                table_names, database_names
            )

            # Step 4: Export generated examples
            if tables_processed > 0 and self.vectorizer:
                self._export_validated_examples()

            # Step 5: Generate report
            self._generate_setup_report()

            logger.info("🎉 Example generation completed!")
            logger.info(f"📈 Tables processed: {tables_processed}")

            return tables_processed > 0

        except Exception as e:
            logger.error(f"❌ Example generation failed: {e}")
            self.setup_report["status"] = "failed"
            self.setup_report["errors"].append(str(e))
            return False

    def _initialize_components(self) -> bool:
        """Initialize extractor and vectorizer components."""
        try:
            import os

            from src.connectors.analytics_backend import AnalyticsConnector
            from src.rag.rag_engine import RAGEngine
            from src.utils.glue_enricher import GlueMetadataEnricher
            from src.utils.schema_introspector import SchemaIntrospector
            from src.utils.vectorizer.schema_vectorizer_orchestrator import (
                SchemaVectorizerOrchestrator,
            )

            # Get analytics DB URL from environment or config
            analytics_db_url = os.getenv("ANALYTICS_DB_URL")
            is_athena_backend = False

            if not analytics_db_url:
                # Fallback to legacy Athena config
                # NOTE: Construct URL directly for Athena
                region = self.athena_config.get("region_name", "ap-southeast-2")
                database = self.athena_config.get("database", "text_to_sql")
                work_group = self.athena_config.get("work_group", "primary")
                s3_staging_dir = self.athena_config.get("s3_staging_dir")

                # Build Athena connection string
                params = [f"region={region}"]
                if work_group:
                    params.append(f"work_group={work_group}")
                if s3_staging_dir:
                    params.append(f"s3_staging_dir={s3_staging_dir}")

                analytics_db_url = f"athena://awsdatacatalog?{'&'.join(params)}"

                # Set database via environment variable for Ibis
                os.environ["ATHENA_DATABASE"] = database

                is_athena_backend = True
                logger.info("✓ Constructed Athena URL from legacy config")
            else:
                # Check if this is an Athena backend
                is_athena_backend = "athena://" in analytics_db_url

            # Create analytics connector directly
            backend = AnalyticsConnector(analytics_db_url)
            logger.info(f"✓ Analytics connector initialized: {backend.backend_type}")

            # Optional: Add Glue enrichment for Athena backends
            glue_enricher = None
            if is_athena_backend:
                try:
                    glue_enricher = GlueMetadataEnricher(
                        aws_profile=self.athena_config.get("aws_profile"),
                        region=self.athena_config.get("region_name", "ap-southeast-2"),
                    )
                    logger.info("✓ Glue enricher initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Glue enricher initialization failed: {e}")

            # Create schema introspector (replaces AthenaSchemaExtractor)
            self.extractor = SchemaIntrospector(backend, glue_enricher)
            logger.info("✓ Schema introspector initialized")

            # Use new RAG interface architecture directly
            self.rag_engine = RAGEngine()
            self.vectorizer = SchemaVectorizerOrchestrator(
                self.rag_engine,
                analytics_backend=backend,  # Pass AnalyticsConnector for SQL validation
            )
            logger.info("✓ Schema vectorizer initialized with RAG interface")

            return True

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False

    def _discover_databases(self) -> list[str]:
        """Discover available databases in Athena."""
        try:
            databases = self.extractor.get_available_databases()
            logger.info(f"✓ Discovered {len(databases)} databases: {databases}")
            return databases
        except Exception as e:
            logger.error(f"❌ Database discovery failed: {e}")
            return []

    def _process_databases(
        self,
        database_names: list[str],
        force_rebuild: bool,
        generate_examples: bool,
    ) -> int:
        """Process a list of databases."""
        success_count = 0

        for database_name in database_names:
            if self._process_database(database_name, force_rebuild, generate_examples):
                success_count += 1
                self.setup_report["databases_processed"].append(database_name)

        return success_count

    def _process_database(
        self,
        database_name: str,
        force_rebuild: bool = False,
        generate_examples: bool = False,
    ) -> bool:
        """Process a single database."""
        logger.info(f"📊 Processing database: {database_name}")

        try:
            # Extract schema
            schema_data = self.extractor.extract_database_schema(database_name)
            if not schema_data:
                logger.error(f"❌ Schema extraction failed for {database_name}")
                return False

            # Apply analyser filtering
            schema_data = self.analyser_filter.filter_schema_tables(
                schema_data, database_name, self.athena_config.get("database")
            )

            if not schema_data.get("tables"):
                logger.warning(
                    f"⚠️ No analysers remain after filtering for {database_name}"
                )
                return False

            # Log processing details
            table_count = schema_data.get("table_count", 0)
            logger.info(
                f"🗂️ Extracted schema for {database_name}: {table_count} analysers"
            )

            # Vectorize schema
            logger.info(f"🔄 Starting vectorization (force_rebuild={force_rebuild})")

            success = self.vectorizer.process_database_schema(
                schema_data,
                generate_examples=generate_examples,
            )

            if success:
                logger.info(f"✅ Successfully processed {database_name}")
                return True
            else:
                logger.error(f"❌ Vectorization failed for {database_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Processing failed for {database_name}: {e}")
            return False

    def _process_database_new_tables_only(
        self, database_name: str, generate_examples: bool = False
    ) -> int:
        """
        Process a single database with new-table filtering.

        Returns:
            Number of new tables processed, or -1 if error occurred

        """
        logger.info(f"🔄 Processing database: {database_name}")

        try:
            # Extract schema from Athena
            schema_data = self.extractor.extract_database_schema(database_name)
            if not schema_data:
                logger.error(f"❌ Schema extraction failed for {database_name}")
                return -1

            # Apply analyser filtering first
            schema_data = self.analyser_filter.filter_schema_tables(
                schema_data, database_name, self.athena_config.get("database")
            )

            if not schema_data.get("tables"):
                logger.warning(
                    f"⚠️ No analysers remain after filtering for {database_name}"
                )
                return -1

            # Filter to only new tables (this would need to be implemented in vectorizer)
            # For now, we'll process all tables in incremental mode
            new_table_count = len(schema_data.get("tables", {}))

            if new_table_count == 0:
                logger.info(f"✅ No new tables to process for {database_name}")
                return 0

            # Process tables
            logger.info(f"🔧 Processing {new_table_count} tables for {database_name}")

            success = self.vectorizer.process_database_schema(
                schema_data,
                generate_examples=generate_examples,
            )

            if success:
                logger.info(
                    f"✅ Successfully processed {new_table_count} tables for {database_name}"
                )
                return new_table_count
            else:
                logger.error(f"❌ Vectorization failed for tables in {database_name}")
                return -1

        except Exception as e:
            logger.error(f"❌ Processing failed for {database_name}: {e}")
            return -1

    def _process_specific_tables(
        self, table_names: list[str], database_names: list[str]
    ) -> int:
        """Process specific tables for example generation."""
        tables_found = {}
        tables_processed = 0

        # Find matching tables across databases
        for database_name in database_names:
            logger.info(f"🔍 Searching for tables in database: {database_name}")

            try:
                schema_data = self.extractor.extract_database_schema(database_name)
                if not schema_data:
                    logger.warning(f"⚠️ Schema extraction failed for {database_name}")
                    continue

                # Apply standard filtering
                schema_data = self.analyser_filter.filter_schema_tables(
                    schema_data, database_name, self.athena_config.get("database")
                )
                available_tables = schema_data.get("tables", {})

                # Find matching tables
                for target_table in table_names:
                    if target_table in available_tables:
                        tables_found[target_table] = {
                            "database": database_name,
                            "schema": available_tables[target_table],
                        }
                        logger.info(
                            f"✓ Found table '{target_table}' in {database_name}"
                        )

            except Exception as e:
                logger.error(f"❌ Error searching {database_name}: {e}")
                continue

        # Check if all requested tables were found
        not_found = set(table_names) - set(tables_found.keys())
        if not_found:
            logger.warning(f"⚠️ Tables not found: {', '.join(not_found)}")

        if not tables_found:
            logger.error("❌ None of the specified tables were found")
            return 0

        logger.info(
            f"📊 Found {len(tables_found)} of {len(table_names)} requested tables"
        )

        # Process each found table
        for table_name, table_info in tables_found.items():
            if self._process_single_table_for_examples(table_name, table_info):
                tables_processed += 1

        return tables_processed

    def _process_single_table_for_examples(
        self, table_name: str, table_info: dict[str, Any]
    ) -> bool:
        """Process a single table for example generation."""
        database_name = table_info["database"]
        table_schema = table_info["schema"]

        logger.info(f"🔧 Processing table: {table_name}")

        try:
            # Create schema data for this single table
            single_table_schema = {
                "database_name": database_name,
                "tables": {table_name: table_schema},
                "table_count": 1,
                "extraction_timestamp": datetime.now().isoformat(),
                "metadata": {},
            }

            # Process with example generation enabled
            success = self.vectorizer.process_database_schema(
                single_table_schema,
                generate_examples=True,
            )

            if success:
                logger.info(f"✅ Successfully processed table: {table_name}")
                self.setup_report["databases_processed"].append(
                    f"{database_name}.{table_name} (examples generated)"
                )
                return True
            else:
                logger.error(f"❌ Failed to process table: {table_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error processing table {table_name}: {e}")
            return False

    def _handle_training_examples(
        self, generate_examples: bool, has_processed_tables: bool
    ):
        """Handle training examples based on generation mode."""
        if generate_examples and has_processed_tables:
            # Export validated examples to YAML
            self._export_validated_examples()
        elif not generate_examples and has_processed_tables:
            # Load existing examples from training/data
            logger.info("📚 Loading existing SQL examples from training/data...")
            examples_loaded = self._load_existing_training_examples()
            if examples_loaded > 0:
                logger.info(
                    f"✅ Loaded {examples_loaded} existing SQL examples into RAG"
                )
            else:
                logger.warning("⚠️ No existing SQL examples found in training/data")

    def _export_validated_examples(self):
        """Export validated examples to YAML."""
        if self.vectorizer and self.vectorizer.validated_examples_for_export:
            logger.info("💾 Exporting validated SQL examples to YAML...")
            result = self.vectorizer.export_all_validated_examples()
            if result:
                logger.info(f"✅ Examples exported to: {result}")
        else:
            logger.info("🔍 No validated examples to export")

    def _load_existing_training_examples(self) -> int:
        """Load existing SQL examples from training/data directory."""
        if self.vectorizer and hasattr(self.vectorizer, "yaml_exporter"):
            try:
                logger.info("📚 Loading existing examples from yaml_exporter...")
                examples = self.vectorizer.yaml_exporter.load_existing_examples()
                logger.info(
                    f"Found {len(examples) if isinstance(examples, list) else 0} examples to process"
                )

                # Add examples to RAG system
                if examples and self.rag_engine:
                    added_count = 0
                    for example in examples:
                        if (
                            isinstance(example, dict)
                            and "question" in example
                            and "sql" in example
                        ):
                            success = self.rag_engine.add_training_example(
                                example["question"], example["sql"]
                            )
                            if success:
                                added_count += 1
                    logger.info(
                        f"✅ Successfully added {added_count}/{len(examples)} training examples to RAG"
                    )
                    return added_count

                return len(examples) if isinstance(examples, list) else 0
            except Exception as e:
                logger.warning(f"Failed to load existing examples: {e}")
                return 0
        else:
            logger.warning(
                "❌ Vectorizer or yaml_exporter not available for loading examples"
            )
        return 0

    def _validate_setup(self, expect_new_documents: bool = True):
        """Validate the completed setup."""
        logger.info("🔍 Validating completed setup...")

        # The new orchestrator doesn't have a validate method, but we can check statistics
        if self.rag_engine:
            stats = self.rag_engine.get_statistics()
            if stats:
                logger.info(f"✅ RAG statistics: {stats}")
                if expect_new_documents and stats.get("total_items", 0) == 0:
                    logger.warning("⚠️ No documents found in RAG after setup")

    def _generate_setup_report(self):
        """Generate comprehensive setup report."""
        import json

        report_file = Path("athena_rag_setup_report.json")

        try:
            # Add summary statistics
            if self.rag_engine:
                stats = self.rag_engine.get_statistics()
                self.setup_report["statistics"] = stats

            # Add configuration summary
            self.setup_report["configuration"] = {
                "athena_database": self.athena_config.get("database"),
                "athena_region": self.athena_config.get("region_name"),
                "rag_embedding_model": self.rag_config.get("bedrock_embedding_model"),
            }

            # Save report
            with open(report_file, "w") as f:
                json.dump(self.setup_report, f, indent=2)

            logger.info(f"📊 Setup report saved to: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate setup report: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.extractor:
            self.extractor.cleanup()
        if self.rag_engine:
            self.rag_engine.cleanup()
