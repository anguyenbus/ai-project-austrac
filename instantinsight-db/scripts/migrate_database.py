#!/usr/bin/env python3
"""
Database Migration Management Script for pgvector RAG.

This script provides an easy interface to manage Alembic migrations
for the pgvector RAG system.
"""

import subprocess
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.config.database_config import (
        DATABASE_URL,
        test_connections,
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)


def run_alembic_command(args: list) -> bool:
    """Run an alembic command using poetry."""
    try:
        cmd = ["poetry", "run", "alembic"] + args
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=project_root, check=True, capture_output=True, text=True
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Alembic command failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def check_database_connection() -> bool:
    """Check if database is accessible."""
    logger.info("Checking database connection...")
    try:
        results = test_connections()
        if results.get("postgresql"):
            logger.info("✓ PostgreSQL connection successful")
            return True
        else:
            logger.error("✗ PostgreSQL connection failed")
            return False
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def show_current_revision():
    """Show current database revision."""
    logger.info("Checking current database revision...")
    return run_alembic_command(["current"])


def show_migration_history():
    """Show migration history."""
    logger.info("Migration history:")
    return run_alembic_command(["history", "--verbose"])


def upgrade_to_head():
    """Upgrade database to latest migration."""
    logger.info("Upgrading database to latest migration...")
    return run_alembic_command(["upgrade", "head"])


def upgrade_to_revision(revision: str):
    """Upgrade database to specific revision."""
    logger.info(f"Upgrading database to revision: {revision}")
    return run_alembic_command(["upgrade", revision])


def downgrade_to_revision(revision: str):
    """Downgrade database to specific revision."""
    logger.info(f"Downgrading database to revision: {revision}")
    return run_alembic_command(["downgrade", revision])


def create_migration(message: str, autogenerate: bool = False):
    """Create a new migration."""
    args = ["revision"]
    if autogenerate:
        args.append("--autogenerate")
    args.extend(["-m", message])

    logger.info(f"Creating new migration: {message}")
    return run_alembic_command(args)


def show_sql_for_upgrade(revision: str = "head"):
    """Show SQL that would be executed for upgrade."""
    logger.info(f"Showing SQL for upgrade to {revision}:")
    return run_alembic_command(["upgrade", "--sql", revision])


def validate_database_schema():
    """Validate that database schema matches migrations."""
    logger.info("Validating database schema...")

    # Check if we can connect
    if not check_database_connection():
        return False

    # Show current revision
    if not show_current_revision():
        return False

    # Try a dry-run upgrade to see if there are pending migrations
    logger.info("Checking for pending migrations...")
    return run_alembic_command(["upgrade", "--sql", "head"])


def init_fresh_database():
    """Initialize a fresh database with all migrations."""
    logger.info("Initializing fresh database...")

    if not check_database_connection():
        logger.error(
            "Cannot connect to database. Please ensure PostgreSQL is running and accessible."
        )
        return False

    # Upgrade to latest
    if not upgrade_to_head():
        logger.error("Failed to apply migrations")
        return False

    logger.info("✓ Database initialization complete!")
    return True


def main():
    """Provide migration management interface."""
    if len(sys.argv) < 2:
        print(
            """
Database Migration Management

Usage: python scripts/migrate_database.py <command> [args]

Commands:
  init              - Initialize fresh database with all migrations
  current           - Show current database revision
  history           - Show migration history
  upgrade [rev]     - Upgrade to head (or specific revision)
  downgrade <rev>   - Downgrade to specific revision
  create <message>  - Create new manual migration
  autogenerate <msg>- Create new migration with autogenerate
  show-sql [rev]    - Show SQL for upgrade (default: head)
  validate          - Validate database schema
  check             - Check database connection

Examples:
  python scripts/migrate_database.py init
  python scripts/migrate_database.py upgrade
  python scripts/migrate_database.py create "Add new index"
  python scripts/migrate_database.py autogenerate "Update user model"
  python scripts/migrate_database.py downgrade -1
        """
        )
        return

    command = sys.argv[1].lower()

    # Show database info
    logger.info(f"Using database: {DATABASE_URL}")

    if command == "init":
        init_fresh_database()
    elif command == "current":
        show_current_revision()
    elif command == "history":
        show_migration_history()
    elif command == "upgrade":
        if len(sys.argv) > 2:
            upgrade_to_revision(sys.argv[2])
        else:
            upgrade_to_head()
    elif command == "downgrade":
        if len(sys.argv) < 3:
            logger.error("Please specify revision to downgrade to")
            return
        downgrade_to_revision(sys.argv[2])
    elif command == "create":
        if len(sys.argv) < 3:
            logger.error("Please specify migration message")
            return
        create_migration(sys.argv[2], autogenerate=False)
    elif command == "autogenerate":
        if len(sys.argv) < 3:
            logger.error("Please specify migration message")
            return
        create_migration(sys.argv[2], autogenerate=True)
    elif command == "show-sql":
        if len(sys.argv) > 2:
            show_sql_for_upgrade(sys.argv[2])
        else:
            show_sql_for_upgrade()
    elif command == "validate":
        validate_database_schema()
    elif command == "check":
        check_database_connection()
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
