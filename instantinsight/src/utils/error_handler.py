"""
Error Handling Utilities.

Standardized error handling patterns to eliminate duplication across the codebase.
Provides consistent error handling with tuple returns (result, error).
"""

import traceback
from collections.abc import Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


class ErrorHandler:
    """Standardized error handling utilities."""

    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        *args,
        error_message_prefix: str = "Operation failed",
        log_errors: bool = True,
        return_none_on_error: bool = True,
        **kwargs,
    ) -> tuple[T | None, str | None]:
        """
        Execute a function safely with standardized error handling.

        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            error_message_prefix: Prefix for error messages
            log_errors: Whether to log errors
            return_none_on_error: Whether to return None on error or raise
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Tuple[Optional[T], Optional[str]]: (result, error_message)
            - On success: (result, None)
            - On failure: (None, error_message) if return_none_on_error=True

        """
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            error_msg = f"{error_message_prefix}: {str(e)}"

            if log_errors:
                logger.error(error_msg)
                logger.debug(f"Traceback: {traceback.format_exc()}")

            if return_none_on_error:
                return None, error_msg
            else:
                raise

    @staticmethod
    def safe_execute_with_default(
        func: Callable[..., T],
        default_value: T,
        *args,
        error_message_prefix: str = "Operation failed",
        log_errors: bool = True,
        **kwargs,
    ) -> tuple[T, str | None]:
        """
        Execute a function safely, returning a default value on error.

        Args:
            func: Function to execute
            default_value: Value to return if function fails
            *args: Arguments to pass to the function
            error_message_prefix: Prefix for error messages
            log_errors: Whether to log errors
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Tuple[T, Optional[str]]: (result_or_default, error_message)

        """
        result, error = ErrorHandler.safe_execute(
            func,
            *args,
            error_message_prefix=error_message_prefix,
            log_errors=log_errors,
            return_none_on_error=True,
            **kwargs,
        )

        if error is not None:
            return default_value, error
        return result, None

    @staticmethod
    def safe_database_operation(
        operation: Callable[..., T],
        *args,
        operation_name: str = "Database operation",
        **kwargs,
    ) -> tuple[T | None, str | None]:
        """
        Execute a database operation safely with specific error handling.

        Args:
            operation: Database operation function to execute
            *args: Arguments to pass to the operation
            operation_name: Name of the operation for error messages
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            Tuple[Optional[T], Optional[str]]: (result, error_message)

        """
        return ErrorHandler.safe_execute(
            operation,
            *args,
            error_message_prefix=f"{operation_name} failed",
            log_errors=True,
            return_none_on_error=True,
            **kwargs,
        )

    @staticmethod
    def safe_llm_operation(
        operation: Callable[..., T],
        *args,
        operation_name: str = "LLM operation",
        **kwargs,
    ) -> tuple[T | None, str | None]:
        """
        Execute an LLM operation safely with specific error handling.

        Args:
            operation: LLM operation function to execute
            *args: Arguments to pass to the operation
            operation_name: Name of the operation for error messages
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            Tuple[Optional[T], Optional[str]]: (result, error_message)

        """
        return ErrorHandler.safe_execute(
            operation,
            *args,
            error_message_prefix=f"{operation_name} failed",
            log_errors=True,
            return_none_on_error=True,
            **kwargs,
        )

    @staticmethod
    def validate_and_execute(
        func: Callable[..., T],
        validation_func: Callable[..., bool],
        validation_error_msg: str,
        *args,
        error_message_prefix: str = "Operation failed",
        **kwargs,
    ) -> tuple[T | None, str | None]:
        """
        Validate inputs before executing a function.

        Args:
            func: Function to execute
            validation_func: Function to validate inputs (should return bool)
            validation_error_msg: Error message if validation fails
            *args: Arguments to pass to both validation and main function
            error_message_prefix: Prefix for error messages
            **kwargs: Keyword arguments to pass to both functions

        Returns:
            Tuple[Optional[T], Optional[str]]: (result, error_message)

        """
        # First validate
        try:
            if not validation_func(*args, **kwargs):
                return None, validation_error_msg
        except Exception as e:
            return None, f"Validation failed: {str(e)}"

        # Then execute
        return ErrorHandler.safe_execute(
            func, *args, error_message_prefix=error_message_prefix, **kwargs
        )


class ServiceError(Exception):
    """Base exception for service-level errors."""


class DatabaseError(ServiceError):
    """Exception for database-related errors."""


class LLMError(ServiceError):
    """Exception for LLM-related errors."""


class ConfigurationError(ServiceError):
    """Exception for configuration-related errors."""
