"""
Langfuse configuration and decorator setup.

This module initializes Langfuse with environment variables and provides
the @observe decorator for automatic tracing.
"""

import os

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()


# Initialize Langfuse with environment variables
def initialize_langfuse():
    """
    Initialize Langfuse with environment variables.

    Returns:
        True if configuration successful, False otherwise

    """
    try:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST")

        if not all([secret_key, public_key, host]):
            logger.warning(
                "Langfuse configuration incomplete. Missing environment variables."
            )
            logger.info(
                "Required: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST"
            )
            return False

        # Configure Langfuse environment variables for decorators
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_HOST"] = host

        logger.info("✅ Langfuse configured successfully for decorator usage")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return False


# Initialize on import
langfuse_enabled = initialize_langfuse()

# Import observe decorator if Langfuse is configured
if langfuse_enabled:
    try:
        from langfuse.decorators import langfuse_context, observe

        logger.info("✅ Langfuse @observe decorator ready")
    except ImportError as e:
        logger.error(f"Failed to import Langfuse decorators: {e}")

        def observe(**kwargs):
            """
            No-op decorator when Langfuse import fails.

            Args:
                **kwargs: Arbitrary keyword arguments ignored by this fallback

            Returns:
                A decorator that returns the original function unchanged

            """

            def decorator(func):
                return func

            return decorator

        langfuse_context = None
else:
    # No-op decorator if Langfuse not configured
    def observe(**kwargs):
        """
        No-op decorator when Langfuse is not configured.

        Args:
            **kwargs: Arbitrary keyword arguments ignored by this fallback

        Returns:
            A decorator that returns the original function unchanged

        """

        def decorator(func):
            return func

        return decorator

    langfuse_context = None
