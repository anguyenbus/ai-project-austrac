#!/usr/bin/env python3
"""
Clear Cache Script.

This script provides comprehensive management of both semantic cache and LRU cache systems,
including clearing all cache entries, selective invalidation, and cache statistics.

Cache Types Managed:
- Semantic Cache: PostgreSQL-based cache with vector similarity search
- LRU Cache: In-memory function-level caches (@lru_cache decorators)

Usage:
    python scripts/clear_cache.py --all                    # Clear all cache entries (semantic + LRU)
    python scripts/clear_cache.py --table users            # Clear semantic cache entries for specific table
    python scripts/clear_cache.py --ttl 24                 # Clear semantic cache entries older than 24 hours
    python scripts/clear_cache.py --stats                  # Show cache statistics only
    python scripts/clear_cache.py --low-confidence 0.8     # Clear semantic cache entries below confidence threshold
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from typing import Any

from loguru import logger


class CacheCleaner:
    """Utility class for managing semantic cache operations."""

    def __init__(self):
        """Initialize cache cleaner with semantic cache instance."""
        self.semantic_cache = None
        self._init_cache()

    def _init_cache(self):
        """Initialize semantic cache connection."""
        try:
            from src.cache.semantic_cache import SemanticCache
            from src.config.database_config import POSTGRES_CONFIG

            # Build PostgreSQL connection string
            connection_string = (
                f"host={POSTGRES_CONFIG['host']} "
                f"port={POSTGRES_CONFIG['port']} "
                f"dbname={POSTGRES_CONFIG['database']} "
                f"user={POSTGRES_CONFIG['user']} "
                f"password={POSTGRES_CONFIG['password']}"
            )

            # Initialize semantic cache
            self.semantic_cache = SemanticCache(
                connection_string=connection_string,
                similarity_threshold=0.92,
                max_cache_size=10000,
                enable_result_validation=True,
            )

            logger.info("✅ Semantic cache initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            sys.exit(1)

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            stats = await self.semantic_cache.get_cache_stats()

            # Get additional manual stats if the view doesn't exist
            if not stats:
                stats = await self._get_manual_stats()

            return stats
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e)}

    async def _get_manual_stats(self) -> dict[str, Any]:
        """Get manual cache statistics by querying the cache table directly."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            with psycopg2.connect(
                self.semantic_cache.connection_string, cursor_factory=RealDictCursor
            ) as conn:
                with conn.cursor() as cur:
                    # Get basic cache statistics
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) as total_entries,
                            AVG(cache_hits) as avg_cache_hits,
                            MAX(cache_hits) as max_cache_hits,
                            AVG(confidence_score) as avg_confidence,
                            MIN(created_at) as oldest_entry,
                            MAX(created_at) as newest_entry,
                            COUNT(CASE WHEN cache_hits > 0 THEN 1 END) as hit_entries,
                            SUM(cache_hits) as total_hits
                        FROM semantic_cache
                    """
                    )

                    result = cur.fetchone()

                    if result:
                        stats = dict(result)
                        # Calculate hit rate
                        if stats["total_entries"] > 0:
                            stats["hit_rate"] = (stats["hit_entries"] or 0) / stats[
                                "total_entries"
                            ]
                        else:
                            stats["hit_rate"] = 0.0

                        return stats
                    else:
                        return {"total_entries": 0}

        except Exception as e:
            logger.warning(f"Failed to get manual statistics: {e}")
            return {"error": str(e)}

    async def clear_all_cache(self) -> bool:
        """Clear all cache entries (both semantic cache and LRU caches)."""
        try:
            # Clear semantic cache
            success = self.semantic_cache.clear_cache()
            if not success:
                logger.error("Failed to clear semantic cache entries")
                return False

            if success:
                logger.success("🧹 All cache entries cleared successfully")
            else:
                logger.error("Failed to clear cache entries")

            return success
        except Exception as e:
            logger.error(f"Error clearing all cache entries: {e}")
            return False

    async def clear_by_table(self, table_name: str) -> int:
        """Clear cache entries that depend on a specific table."""
        try:
            count = await self.semantic_cache.invalidate_by_table(table_name)
            logger.success(f"🗑️ Cleared {count} cache entries for table: {table_name}")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache by table {table_name}: {e}")
            return 0

    async def clear_by_ttl(self, hours: int) -> int:
        """Clear cache entries older than specified hours."""
        try:
            count = await self.semantic_cache.invalidate_by_ttl(hours)
            logger.success(f"🗑️ Cleared {count} cache entries older than {hours} hours")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache by TTL {hours}h: {e}")
            return 0

    async def clear_by_confidence(self, min_confidence: float) -> int:
        """Clear cache entries below confidence threshold."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            with psycopg2.connect(
                self.semantic_cache.connection_string, cursor_factory=RealDictCursor
            ) as conn:
                with conn.cursor() as cur:
                    # Delete entries with low confidence
                    cur.execute(
                        """
                        DELETE FROM semantic_cache 
                        WHERE confidence_score < %s
                        RETURNING id
                    """,
                        (min_confidence,),
                    )

                    deleted_ids = cur.fetchall()
                    conn.commit()

                    count = len(deleted_ids)
                    logger.success(
                        f"🗑️ Cleared {count} cache entries with confidence < {min_confidence}"
                    )
                    return count

        except Exception as e:
            logger.error(f"Error clearing cache by confidence: {e}")
            return 0

    def display_stats(self, stats: dict[str, Any]):
        """Display cache statistics in a formatted way."""
        print("\n" + "=" * 60)
        print("📈 SEMANTIC CACHE STATISTICS")
        print("=" * 60)

        if "error" in stats:
            print(f"❌ Error: {stats['error']}")
            return

        # Basic statistics
        total_entries = stats.get("total_entries", 0)
        print(f"📊 Total Cache Entries:     {total_entries:,}")

        if total_entries > 0:
            print(f"🎯 Cache Hit Rate:          {stats.get('hit_rate', 0):.1%}")
            print(f"📈 Total Cache Hits:        {stats.get('total_hits', 0):,}")
            print(f"⭐ Average Confidence:      {stats.get('avg_confidence', 0):.3f}")
            print(f"🏆 Max Cache Hits:          {stats.get('max_cache_hits', 0):,}")

            # Dates
            if stats.get("oldest_entry"):
                print(f"📅 Oldest Entry:            {stats['oldest_entry']}")
            if stats.get("newest_entry"):
                print(f"🕐 Newest Entry:            {stats['newest_entry']}")
        else:
            print("📭 Cache is empty")

        print("=" * 60 + "\n")


def main():
    """Handle command line arguments and execute cache operations."""
    parser = argparse.ArgumentParser(
        description="Cache Management Tool (Semantic Cache + LRU Cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clear_cache.py --all                    # Clear all cache entries (semantic + LRU)
  python scripts/clear_cache.py --table users            # Clear semantic cache entries for specific table  
  python scripts/clear_cache.py --ttl 24                 # Clear semantic cache entries older than 24 hours
  python scripts/clear_cache.py --stats                  # Show cache statistics only
  python scripts/clear_cache.py --low-confidence 0.8     # Clear semantic cache entries below confidence threshold
  python scripts/clear_cache.py --stats --clear-all      # Show stats then clear all
        """,
    )

    # Action arguments (mutually exclusive for clearing operations)
    parser.add_argument(
        "--all",
        "--clear-all",
        action="store_true",
        help="Clear all cache entries (semantic cache + LRU caches)",
    )
    parser.add_argument(
        "--table",
        type=str,
        metavar="TABLE_NAME",
        help="Clear cache entries for specific table",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        metavar="HOURS",
        help="Clear cache entries older than specified hours",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        metavar="THRESHOLD",
        help="Clear cache entries below confidence threshold (0.0-1.0)",
    )

    # Information arguments
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.add(sys.stdout, level="INFO")

    # Initialize cache cleaner
    cleaner = CacheCleaner()

    async def run_operations():
        """Run the requested cache operations."""
        operations_performed = False

        # Show statistics if requested
        if args.stats:
            print("📊 Fetching cache statistics...")
            stats = await cleaner.get_cache_stats()
            cleaner.display_stats(stats)
            operations_performed = True

        # Perform clearing operations
        if args.all:
            print("🧹 Clearing all cache entries...")
            success = await cleaner.clear_all_cache()
            if not success:
                sys.exit(1)
            operations_performed = True

        elif args.table:
            print(f"🗑️ Clearing cache entries for table: {args.table}")
            count = await cleaner.clear_by_table(args.table)
            print(f"✅ Cleared {count} entries")
            operations_performed = True

        elif args.ttl:
            print(f"🕐 Clearing cache entries older than {args.ttl} hours...")
            count = await cleaner.clear_by_ttl(args.ttl)
            print(f"✅ Cleared {count} entries")
            operations_performed = True

        elif args.low_confidence:
            if not (0.0 <= args.low_confidence <= 1.0):
                logger.error("Confidence threshold must be between 0.0 and 1.0")
                sys.exit(1)
            print(f"⭐ Clearing cache entries with confidence < {args.low_confidence}")
            count = await cleaner.clear_by_confidence(args.low_confidence)
            print(f"✅ Cleared {count} entries")
            operations_performed = True

        # Show final stats if clearing was performed
        if (
            operations_performed
            and not args.stats
            and (args.all or args.table or args.ttl or args.low_confidence)
        ):
            print("\n📊 Updated cache statistics:")
            stats = await cleaner.get_cache_stats()
            cleaner.display_stats(stats)

        # If no operations specified, show help
        if not operations_performed:
            parser.print_help()
            print("\n💡 Use --stats to view current cache statistics")
            print("💡 Use --all to clear all cache entries (semantic + LRU)")
            print(
                "💡 Note: --all clears both semantic cache (PostgreSQL) and LRU caches (in-memory)"
            )

    # Run async operations
    try:
        asyncio.run(run_operations())
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
