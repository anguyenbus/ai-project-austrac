"""
Scripts module for nl2sql2vis project.

This module contains utility scripts for setup, training, and maintenance tasks.
"""

# Scripts are typically run directly, so we don't need to expose them as imports
# But we can document what's available:

__doc__ += """

Available scripts:
- setup_local_rag.py: Setup the local RAG system with pgvector for PostgreSQL
- generate_your_db_name_training.py: Generate training data from your_db_name database structure
- add_your_db_name_training.py: Add training data to the RAG system
- docker_helper.py: Docker and database helper utilities
- train.py: Training script for the RAG system
- clear_cache.py: Clear semantic cache and LRU caches

Usage:
    python scripts/script_name.py
"""
